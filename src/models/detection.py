"""Componentes compartidos por el modelo 3D (`src/models/model3d.py`): dispositivo, semilla y precisión mixta, caché
del cubo isótropo, atención CBAM y squeeze-excitation con γ, anclas, cabeza con anclas, IoU, NMS propio, codificación de
cajas y AP.

γ: cada capa con atención calcula y = x + γ · atención(x), con γ aprendible que empieza en 0. Si al final γ queda
cerca de 0, la red no usó la atención de esa capa.

Todos los parámetros están en `src/config.py`.

Comandos (desde la raíz del proyecto):
    uv run python -m src.models.detection cache      # cubo 256³ de los 100 casos en data/cache_cubo_400mm_256/
    uv run python -m src.models.detection anchors    # estadísticas de las cajas de train para las anclas
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.dataset import preprocessing as pre

VIEWS = config.VIEWS
PX_COLS = [f"px_{r}" for r in config.REGION_NAMES]
CLASSES = {v: tuple(config.VIEW_CLASSES[v]) for v in VIEWS}

# Código de fragmento (0-255) -> clase de cada vista (0 fondo, 1..n en el orden de VIEW_CLASSES).
# En sagital, los códigos de los dos coxales van a la misma clase "hipbone".
VIEW_LUT = {}
for _v in VIEWS:
    _lut = np.zeros(256, np.uint8)
    for _r, _codes in config.REGION_CODES.items():
        _name = "hipbone" if (_v == "sagittal" and _r.endswith("hipbone")) else _r
        _lut[list(_codes)] = CLASSES[_v].index(_name) + 1
    VIEW_LUT[_v] = _lut


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = config.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, enabled: bool = config.AMP):
    """Precisión mixta. CUDA: float16 (con GradScaler en el entrenamiento, como torch.cuda.amp). MPS: bfloat16,
    que tiene el rango de float32 y no se desborda (en MPS no hay GradScaler que detecte los desbordes)."""
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=enabled and device.type != "cpu")


# ======================================================================================
# Datos
# ======================================================================================

def _slice_counts(cube_regions: np.ndarray, view: str) -> np.ndarray:
    """Píxeles de cada región (1..3) en cada corte de la vista -> (256, 3)."""
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[view]
    return np.stack([(cube_regions == c + 1).sum(axis=axes) for c in range(len(config.REGION_NAMES))], 1)


def _bone_flags(hu_cube: np.ndarray, view: str) -> np.ndarray:
    """Corte con hueso según los HU (sin máscara) en cada corte de la vista -> (256,) bool."""
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[view]
    return (hu_cube >= config.BONE_HU).mean(axis=axes) >= config.MIN_BONE_FRACTION


def build_cache(overwrite: bool = False) -> pd.DataFrame:
    """Preprocesa los 100 casos al cubo y guarda en `config.CACHE_DIR`:
    `<caso>_img.npy` (uint8, ventana HU a 0-255) y `<caso>_msk.npy` (códigos de fragmento).

    `index.csv` tiene una fila por corte de cada vista (3 × 256 por caso) con los píxeles de cada
    región y si el corte tiene hueso por HU. `cases.csv` guarda la geometría del cubo y la fracción
    del hueso etiquetado que queda fuera del cubo.
    """
    from tqdm.auto import tqdm

    meta = {"cube_mm": config.CUBE_MM, "input_size": config.INPUT_SIZE, "hu_window": list(config.HU_WINDOW),
            "orientation": config.TARGET_ORIENTATION, "bone_hu": config.BONE_HU,
            "min_bone_fraction": config.MIN_BONE_FRACTION}
    cache = config.CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "cache_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta and not overwrite:
        raise RuntimeError(f"El caché se hizo con otros parámetros ({meta_path}). Usa --overwrite.")

    splits = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})
    img_paths = {p.stem: p for d in config.IMAGE_DIRS for p in d.glob("*.mha")}
    rows, cases = [], []
    old_cases = cache / "cases.csv"
    prev = pd.read_csv(old_cases, dtype={"case_id": str}).set_index("case_id") if old_cases.exists() and not overwrite else None
    region_lut = VIEW_LUT["axial"]
    for cid, split in tqdm(list(zip(splits.case_id, splits.split)), desc="Caché del cubo"):
        fi, fm, fb = cache / f"{cid}_img.npy", cache / f"{cid}_msk.npy", cache / f"{cid}_bone.npy"
        if overwrite or not (fi.exists() and fm.exists() and fb.exists()) or prev is None or cid not in prev.index:
            img, msk = pre.load_volume(img_paths[cid]), pre.load_volume(config.LABEL_DIR / f"{cid}.mha")
            res = pre.preprocess_volume(img, msk)
            np.save(fi, np.round(np.clip(res["image"], 0, 1) * 255).astype(np.uint8))
            np.save(fm, res["mask"])
            np.save(fb, np.stack([_bone_flags(res["hu"], v) for v in VIEWS]))
            fuera, _ = pre.fraction_outside_cube(pre.reorient(msk), res["cube"])
            g = res["cube"]
            cases.append({"case_id": cid, "split": split, "bone_outside_frac": fuera,
                          **{f"origin_{a}": o for a, o in zip("xyz", g["origin"])},
                          "direction": " ".join(f"{d:g}" for d in g["direction"])})
        reg = region_lut[np.load(fm)]
        bone = np.load(fb)
        for vi, view in enumerate(VIEWS):
            counts = _slice_counts(reg, view)
            for s in range(config.INPUT_SIZE):
                rows.append({"case_id": cid, "split": split, "view": view, "slice": s, "bone": bool(bone[vi, s]),
                             **{col: int(n) for col, n in zip(PX_COLS, counts[s])}})
    index = pd.DataFrame(rows)
    index.to_csv(cache / "index.csv", index=False)
    new = pd.DataFrame(cases)
    if prev is not None:
        new = pd.concat([prev.reset_index()[~prev.index.isin(new.case_id if len(new) else [])], new])
    if len(new):
        new.sort_values("case_id").to_csv(old_cases, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    c = pd.read_csv(old_cases, dtype={"case_id": str})
    print(f"Caché listo: {index.case_id.nunique()} casos, {len(index)} cortes (3 vistas). Hueso fuera del cubo: "
          f"{int((c.bone_outside_frac > 0).sum())} casos, máximo {100 * c.bone_outside_frac.max():.3f} %")
    return index


def load_index() -> pd.DataFrame:
    path = config.CACHE_DIR / "index.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: python -m src.models.detection cache")
    index = pd.read_csv(path, dtype={"case_id": str})
    index["px_hipbone"] = index.px_left_hipbone + index.px_right_hipbone
    sag = index.view == "sagittal"
    index["positive"] = (index[PX_COLS] >= config.MIN_REGION_PX).any(axis=1)
    index.loc[sag, "positive"] = ((index.loc[sag, ["px_sacrum", "px_hipbone"]] >= config.MIN_REGION_PX).any(axis=1))
    return index


def region_boxes(m: torch.Tensor, num_classes: int, min_px: int = config.MIN_REGION_PX):
    """Caja (x0, y0, x1, y1) en píxeles de cada clase del mapa (B, H, W) -> boxes (B, C, 4), valid (B, C).

    x1 e y1 son exclusivos (la caja de un solo píxel en (i, j) es (j, i, j+1, i+1)).
    """
    b, h, w = m.shape
    oh = torch.stack([m == c + 1 for c in range(num_classes)], 1)          # (B, C, H, W)
    cols, rows = oh.any(2), oh.any(3)
    ax = torch.arange(w, device=m.device)
    ay = torch.arange(h, device=m.device)
    x0 = torch.where(cols, ax, w).amin(-1)
    x1 = torch.where(cols, ax, -1).amax(-1) + 1
    y0 = torch.where(rows, ay, h).amin(-1)
    y1 = torch.where(rows, ay, -1).amax(-1) + 1
    valid = oh.sum((2, 3)) >= min_px
    return torch.stack([x0, y0, x1, y1], -1).float(), valid


# ======================================================================================
# Modelo
# ======================================================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018).

    Canal: avg-pool y max-pool globales -> MLP compartido -> sigmoide: qué mapas aportan.
    Espacial: media y máximo entre canales -> conv 7x7 -> sigmoide: qué zonas importan.
    Guarda el último mapa de atención espacial en `spatial_map` para inspeccionarlo.
    """

    def __init__(self, channels: int, reduction: int = config.CBAM_REDUCTION, kernel: int = config.CBAM_KERNEL):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2)
        self.spatial_map = None

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        self.spatial_map = sa.detach()
        return x * sa


class GatedCBAM(nn.Module):
    """y = x + γ · CBAM(x), con γ aprendible que empieza en `config.GAMMA_INIT` (0).

    Con γ = 0 la capa deja pasar x sin cambios: la atención empieza sin aportar y la red aprende
    cuánto usarla. El valor final de γ (y su evolución por época) mide el aporte de la atención.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.cbam = CBAM(channels)
        self.gamma = nn.Parameter(torch.tensor(float(config.GAMMA_INIT)))

    def forward(self, x):
        return x + self.gamma * self.cbam(x)


class SE(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018): solo atención de canal. avg-pool global -> MLP -> sigmoide."""

    def __init__(self, channels: int, reduction: int = config.CBAM_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)))


class GatedSE(nn.Module):
    """y = x + γ · SE(x), con γ aprendible desde `config.GAMMA_INIT` (igual que GatedCBAM)."""

    def __init__(self, channels: int):
        super().__init__()
        self.se = SE(channels)
        self.gamma = nn.Parameter(torch.tensor(float(config.GAMMA_INIT)))

    def forward(self, x):
        return x + self.gamma * self.se(x)


def generar_anclas_celda(cx, cy, escalas, aspect_ratios):
    """Anclas (x0, y0, x1, y1) centradas en (cx, cy): una por escala y proporción ancho/alto."""
    anclas = []
    for escala in escalas:
        for ar in aspect_ratios:
            w = escala * np.sqrt(ar)
            h = escala / np.sqrt(ar)
            anclas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(anclas)


def generar_anclas_grilla(tam_mapa, stride, escalas, aspect_ratios):
    """Anclas de todas las celdas del mapa, en orden (fila, columna, ancla)."""
    anclas = []
    for fila in range(tam_mapa):
        for col in range(tam_mapa):
            cx = (col + 0.5) * stride
            cy = (fila + 0.5) * stride
            anclas.append(generar_anclas_celda(cx, cy, escalas, aspect_ratios))
    return np.concatenate(anclas, axis=0)


class AnchorHead(nn.Module):
    """Cabeza con anclas, estilo RPN: conv 3x3 + ReLU compartida, atención
    (CBAM + γ, si `attention`) y, por cada ancla de cada celda, un logit por clase y 4 offsets."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int, attention):
        super().__init__()
        mid = config.HEAD_CHANNELS
        self.num_classes, self.num_anchors = num_classes, num_anchors
        self.conv = nn.Conv2d(in_channels, mid, 3, padding=1)
        # attention: True / "cbam" -> CBAM + γ; "se" -> squeeze-excitation + γ; False / "none" -> sin atención
        kind = {True: "cbam", False: "none"}.get(attention, attention)
        self.attn = {"cbam": GatedCBAM, "se": GatedSE}[kind](mid) if kind != "none" else nn.Identity()
        self.cls = nn.Conv2d(mid, num_anchors * num_classes, 1)
        self.reg = nn.Conv2d(mid, num_anchors * 4, 1)

    def forward(self, x):
        x = self.attn(torch.relu(self.conv(x)))
        b, _, h, w = x.shape
        # (B, K·C, H, W) -> (B, H·W·K, C): mismo orden (fila, columna, ancla) que generar_anclas_grilla
        cls = self.cls(x).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, self.num_classes)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, 4)
        return cls, reg


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ======================================================================================
# Cajas, NMS y codificación
# ======================================================================================

def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def box_iou(a, b):
    """IoU entre todas las cajas de a (N, 4) y b (M, 4) -> (N, M)."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter).clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = config.NMS_IOU) -> torch.Tensor:
    """Supresión de no máximos: toma la caja de mayor puntaje, elimina las que la solapan con
    IoU > `iou_thr` y repite con las que quedan. Devuelve los índices conservados, de mayor a menor puntaje.
    """
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thr]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long, device=boxes.device)


def calcular_offsets(anclas: torch.Tensor, cajas: torch.Tensor) -> torch.Tensor:
    """Offsets (tx, ty, tw, th) que llevan cada ancla a su caja real, para N pares."""
    aw, ah = anclas[:, 2] - anclas[:, 0], anclas[:, 3] - anclas[:, 1]
    acx, acy = anclas[:, 0] + aw / 2, anclas[:, 1] + ah / 2
    gw, gh = cajas[:, 2] - cajas[:, 0], cajas[:, 3] - cajas[:, 1]
    gcx, gcy = cajas[:, 0] + gw / 2, cajas[:, 1] + gh / 2
    return torch.stack([(gcx - acx) / aw, (gcy - acy) / ah, torch.log(gw / aw), torch.log(gh / ah)], 1)


def decode_boxes(anclas: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Inversa de `calcular_offsets`: anclas (A, 4) + offsets (A, 4) -> cajas (A, 4)."""
    aw, ah = anclas[:, 2] - anclas[:, 0], anclas[:, 3] - anclas[:, 1]
    acx, acy = anclas[:, 0] + aw / 2, anclas[:, 1] + ah / 2
    tw, th = deltas[:, 2].clamp(max=4.0), deltas[:, 3].clamp(max=4.0)    # evita exp() desbordado
    pcx, pcy = deltas[:, 0] * aw + acx, deltas[:, 1] * ah + acy
    pw, ph = torch.exp(tw) * aw, torch.exp(th) * ah
    return torch.stack([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2], 1)


def sample_anchors(labels: torch.Tensor, batch_size: int = config.ANCHOR_BATCH, pos_fraction: float = config.POS_FRACTION):
    """Muestra hasta `batch_size` anclas, como máximo `pos_fraction` positivas.
    Devuelve los índices seleccionados (positivas + negativas) y solo las positivas."""
    positivos = torch.where(labels == 1)[0]
    negativos = torch.where(labels == 0)[0]
    n_pos = min(len(positivos), int(batch_size * pos_fraction))
    n_neg = min(len(negativos), batch_size - n_pos)
    positivos = positivos[torch.randperm(len(positivos), device=labels.device)[:n_pos]]
    negativos = negativos[torch.randperm(len(negativos), device=labels.device)[:n_neg]]
    return torch.cat([positivos, negativos]), positivos


# ======================================================================================
# Métricas
# ======================================================================================

def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP interpolada en 101 puntos de recall (como COCO)."""
    prec = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


# ======================================================================================
# Anclas
# ======================================================================================

def anchors(args) -> dict:
    """Estadísticas de las cajas de train en cada vista para elegir las anclas: p25 / p50 / p75 de la
    raíz del área y p20 / p50 / p80 de la proporción ancho/alto. Guarda results/detection/anchors.json."""
    index = load_index()
    out = {}
    for v in VIEWS:
        d = index[(index.split == "train") & (index.view == v) & index.positive]
        side, ratio = [], []
        for cid, g in d.groupby("case_id"):
            msk = np.load(config.CACHE_DIR / f"{cid}_msk.npy", mmap_mode="r")
            stack = torch.from_numpy(VIEW_LUT[v][pre.view_slices(msk, v, g["slice"].tolist())])
            b, ok = region_boxes(stack, len(CLASSES[v]))
            b = b[ok]
            w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            side += torch.sqrt(w * h).tolist(); ratio += (w / h).tolist()
        side, ratio = np.array(side), np.array(ratio)
        out[v] = {"scales": [int(round(x)) for x in np.percentile(side, (25, 50, 75))],
                  "ratios": [round(float(x), 2) for x in np.percentile(ratio, (20, 50, 80))], "n_boxes": len(side)}
        print(f"{v:8s} escalas {out[v]['scales']}  proporciones {out[v]['ratios']}  ({len(side)} cajas)")
    config.DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    (config.DETECTION_DIR / "anchors.json").write_text(json.dumps(out, indent=2))
    print("Copia estos valores en config.ANCHORS.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Caché del cubo y estadísticas de las cajas para las anclas.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="preprocesa los 100 casos al cubo y guarda el índice de cortes")
    c.add_argument("--overwrite", action="store_true")
    sub.add_parser("anchors", help="estadísticas de las cajas para elegir las anclas")
    args = p.parse_args()
    if args.cmd == "cache":
        build_cache(args.overwrite)
    else:
        anchors(args)


if __name__ == "__main__":
    main()
