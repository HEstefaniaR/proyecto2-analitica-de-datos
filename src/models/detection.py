"""Detección de regiones pélvicas con tres vistas (axial, coronal y sagital), estilo QuickNAT.

Pipeline:
  CT 3D -> reorientación LPS -> cubo isótropo de 400 mm y 256³ vóxeles (1,5625 mm), con relleno de
  aire -> ventana HU -> cortes de las tres vistas -> entrada 2.5D (corte y vecinos como 3 canales)
  -> backbone FundidoraFPN COMPARTIDO por las tres vistas (batches con cortes de las tres), con
  CBAM + γ en los bloques 3 y 4
  -> cabeza con anclas propia de cada vista (con CBAM + γ, como la RPN del taller 3) sobre p3
  -> NMS propio -> una caja por región y corte -> fusión de las tres vistas en una caja 3D por región.

Clases: axial y coronal predicen sacro, coxal izquierdo y coxal derecho. En un corte sagital los dos
coxales se ven iguales, así que la vista sagital predice sacro y "coxal", y el lado se asigna por la
posición del corte respecto a la línea media (como QuickNAT con los hemisferios).

γ: cada capa con atención calcula y = x + γ · CBAM(x), con γ aprendible que empieza en 0. Si al
final γ queda cerca de 0, la red no usó la atención de esa capa. Se guarda en cada época.

Todos los parámetros están en `src/config.py`.

Comandos (desde la raíz del proyecto):
    uv run python -m src.models.detection cache
    uv run python -m src.models.detection anchors
    uv run python -m src.models.detection overfit
    uv run python -m src.models.detection train
    uv run python -m src.models.detection train --no-cbam
    uv run python -m src.models.detection evaluate --run tres_vistas_cbam
    uv run python -m src.models.detection evaluate --run tres_vistas_sin_cbam
    uv run python -m src.models.detection probe --run tres_vistas_cbam
    uv run python -m src.models.detection summary
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.dataset import preprocessing as pre

VIEWS = config.VIEWS
VIEW_ID = {v: i for i, v in enumerate(VIEWS)}
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


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


class SliceDataset(torch.utils.data.Dataset):
    """Devuelve (K cortes uint8 (K, H, W), mapa de clases de la vista uint8 (H, W), fila, vista).

    Los cubos se abren con memmap al primer uso. En los bordes del cubo el vecino que falta repite
    el corte más cercano.
    """

    def __init__(self, index: pd.DataFrame, k_slices: int = config.K_SLICES):
        self.case = index.case_id.to_numpy()
        self.slice = index["slice"].to_numpy()
        self.view = index.view.map(VIEW_ID).to_numpy()
        self.k = k_slices
        self._vols = {}

    def _arrays(self, cid):
        if cid not in self._vols:
            self._vols[cid] = (np.load(config.CACHE_DIR / f"{cid}_img.npy", mmap_mode="r"),
                               np.load(config.CACHE_DIR / f"{cid}_msk.npy", mmap_mode="r"))
        return self._vols[cid]

    def __len__(self):
        return len(self.case)

    def __getitem__(self, i):
        img, msk = self._arrays(self.case[i])
        view, s, h = VIEWS[self.view[i]], int(self.slice[i]), self.k // 2
        zs = np.clip(np.arange(s - h, s + h + 1), 0, config.INPUT_SIZE - 1).tolist()
        x = pre.view_slices(img, view, zs)
        y = VIEW_LUT[view][pre.view_slices(msk, view, s)]
        return torch.from_numpy(x), torch.from_numpy(y), i, self.view[i]


class EpochSampler(torch.utils.data.Sampler):
    """Índices de una época: `per_view` cortes de cada vista, mezclados entre vistas.

    Por vista: una fracción `neg_frac` sin regiones (de preferencia con hueso por HU, p. ej. fémur o
    columna, que son los negativos difíciles) y el resto con alguna. Los batches quedan con cortes de
    las tres vistas, así BatchNorm ve en entrenamiento la misma mezcla que guarda en sus estadísticas
    para la evaluación. La semilla depende de la época (`set_epoch`).
    """

    def __init__(self, index: pd.DataFrame, per_view: int, neg_frac: float, seed: int = config.SEED):
        self.groups = {}
        for v in VIEWS:
            sel = (index.view == v).to_numpy()
            pos = np.flatnonzero(sel & index.positive.to_numpy())
            neg = np.flatnonzero(sel & ~index.positive.to_numpy() & index.bone.to_numpy())
            if len(neg) == 0:
                neg = np.flatnonzero(sel & ~index.positive.to_numpy())
            self.groups[v] = (pos, neg)
        self.per_view, self.neg_frac, self.seed, self.epoch = per_view, neg_frac, seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        parts = []
        for pos, neg in self.groups.values():
            n_neg = int(round(self.per_view * self.neg_frac)) if len(neg) else 0
            parts.append(rng.choice(pos, self.per_view - n_neg, replace=self.per_view - n_neg > len(pos)))
            if n_neg:
                parts.append(rng.choice(neg, n_neg, replace=n_neg > len(neg)))
        return iter(rng.permutation(np.concatenate(parts)).tolist())

    def __len__(self):
        return len(VIEWS) * self.per_view


def to_device(x_u8, m_u8, device, augment=False):
    """Pasa un batch a `device`: imagen a float [0, 1] y mapa de clases a long, con aumento opcional.

    Aumento (mismo muestreo para imagen y máscara; parámetros en config): rotación, escala,
    traslación, contraste, brillo y gamma de intensidad. Sin volteo horizontal: cambiaría el lado
    del coxal sin cambiar su etiqueta.
    """
    x = x_u8.to(device, non_blocking=True).float().div_(255)
    m = m_u8.to(device, non_blocking=True)
    if augment:
        b = x.shape[0]
        u = lambda: torch.rand(b, device=device) * 2 - 1
        ang, sc = u() * math.radians(config.AUG_ROTATION_DEG), 1 + u() * config.AUG_SCALE
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, u() * config.AUG_SHIFT], 1),
                             torch.stack([sin, cos, u() * config.AUG_SHIFT], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.grid_sample(m[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        m = m[:, 0].round().long()
        gamma = torch.exp(u() * math.log(config.AUG_GAMMA)).view(b, 1, 1, 1)
        x = (x.clamp(0, 1) ** gamma * (1 + u() * config.AUG_CONTRAST).view(b, 1, 1, 1)
             + (u() * config.AUG_BRIGHTNESS).view(b, 1, 1, 1)).clamp(0, 1)
        x = degrade(x)
    return x, m.long()


def degrade(x: torch.Tensor) -> torch.Tensor:
    """Simula CT de peor calidad. A cada corte, por separado y con probabilidad `config.AUG_DEGRADE_PROB`:
    baja resolución (reducir y volver a 256), desenfoque gaussiano y ruido gaussiano. Solo cambia la
    imagen: las cajas siguen siendo las mismas."""
    b, c, h, w = x.shape
    dev = x.device
    pick = lambda: (torch.rand(b, device=dev) < config.AUG_DEGRADE_PROB).nonzero(as_tuple=True)[0]
    unif = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1))
    idx = pick()
    if idx.numel():                                         # baja resolución
        f = unif(*config.AUG_LOWRES_FACTOR)
        small = F.interpolate(x[idx], size=(max(int(h / f), 8), max(int(w / f), 8)), mode="bilinear", align_corners=False)
        x[idx] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    idx = pick()
    if idx.numel():                                         # desenfoque gaussiano separable
        sigma = unif(*config.AUG_BLUR_SIGMA)
        r = int(math.ceil(3 * sigma))
        k = torch.exp(-torch.arange(-r, r + 1, device=dev, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
        k = k / k.sum()
        y = F.conv2d(F.pad(x[idx], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        x[idx] = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    idx = pick()
    if idx.numel():                                         # ruido gaussiano
        std = torch.rand(len(idx), 1, 1, 1, device=dev) * config.AUG_NOISE_STD
        x[idx] = x[idx] + torch.randn_like(x[idx]) * std
    return x.clamp(0, 1)


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

def conv_bn_relu(in_c: int, out_c: int, k: int = 3, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))


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


class UpBlock(nn.Module):
    """Interpolación bilineal ×2 + conv 1x1 (mitad de canales) -> concat skip -> 2 x (Conv3x3+BN+ReLU).

    La subida bilineal evita el patrón de rayas cada 8 px que dejaba la ConvTranspose 2x2.
    """

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                nn.Conv2d(in_c, in_c // 2, 1))
        self.conv = nn.Sequential(conv_bn_relu(in_c // 2 + skip_c, out_c), conv_bn_relu(out_c, out_c))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class FundidoraFPN(nn.Module):
    """FundidoraPC + CBAM con γ en los bloques `config.CBAM_BLOCKS` + cuello dilatado + decoder U-Net.

    Devuelve {"f1": 1/2, "p2": 1/4, "p3": 1/8, "p4": 1/16}. p3 alimenta la detección; p2 y f1 quedan
    para el decoder de segmentación.
    """

    def __init__(self, cbam: bool = True, in_channels: int = config.K_SLICES, widths=config.WIDTHS):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks.append(nn.Sequential(conv_bn_relu(c, w), nn.MaxPool2d(2)))
            c = w
        self.blocks = nn.ModuleList(blocks)
        self.attn = nn.ModuleList(GatedCBAM(w) if cbam and i + 1 in config.CBAM_BLOCKS else nn.Identity()
                                  for i, w in enumerate(widths))
        w1, w2, w3, w4 = widths
        d1, d2 = config.BOTTLENECK_DILATIONS
        self.bottleneck = nn.Sequential(conv_bn_relu(w4, w4, dilation=d1), conv_bn_relu(w4, w4, dilation=d2))
        self.up3 = UpBlock(w4, w3, w3)                        # 1/16 -> 1/8
        self.up2 = UpBlock(w3, w2, w2)                        # 1/8 -> 1/4
        self.widths = tuple(widths)

    def forward(self, x):
        feats = []
        for block, attn in zip(self.blocks, self.attn):
            x = attn(block(x))
            feats.append(x)
        f1, f2, f3, f4 = feats
        p4 = self.bottleneck(f4)
        p3 = self.up3(p4, f3)
        p2 = self.up2(p3, f2)
        return {"f1": f1, "p2": p2, "p3": p3, "p4": p4}


def generar_anclas_celda(cx, cy, escalas, aspect_ratios):
    """Anclas (x0, y0, x1, y1) centradas en (cx, cy): una por escala y proporción ancho/alto (taller 3)."""
    anclas = []
    for escala in escalas:
        for ar in aspect_ratios:
            w = escala * np.sqrt(ar)
            h = escala / np.sqrt(ar)
            anclas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(anclas)


def generar_anclas_grilla(tam_mapa, stride, escalas, aspect_ratios):
    """Anclas de todas las celdas del mapa, en orden (fila, columna, ancla) (taller 3)."""
    anclas = []
    for fila in range(tam_mapa):
        for col in range(tam_mapa):
            cx = (col + 0.5) * stride
            cy = (fila + 0.5) * stride
            anclas.append(generar_anclas_celda(cx, cy, escalas, aspect_ratios))
    return np.concatenate(anclas, axis=0)


class AnchorHead(nn.Module):
    """Cabeza con anclas, como la RPNHead del taller 3: conv 3x3 + ReLU compartida, atención
    (CBAM + γ, si `attention`) y, por cada ancla de cada celda, un logit por clase y 4 offsets."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int, attention: bool):
        super().__init__()
        mid = config.HEAD_CHANNELS
        self.num_classes, self.num_anchors = num_classes, num_anchors
        self.conv = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.attn = GatedCBAM(mid) if attention else nn.Identity()
        self.cls = nn.Conv2d(mid, num_anchors * num_classes, 1)
        self.reg = nn.Conv2d(mid, num_anchors * 4, 1)

    def forward(self, x):
        x = self.attn(torch.relu(self.conv(x)))
        b, _, h, w = x.shape
        # (B, K·C, H, W) -> (B, H·W·K, C): mismo orden (fila, columna, ancla) que generar_anclas_grilla
        cls = self.cls(x).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, self.num_classes)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, 4)
        return cls, reg


class ThreeViewDetector(nn.Module):
    """Backbone FundidoraFPN compartido por las tres vistas + una cabeza con anclas por vista (p3).

    `forward(x, view)`:
      - `view` str: todo el batch es de esa vista. Devuelve "cls" (B, A, C) logits, "reg" (B, A, 4)
        offsets, "anchors" (A, 4) las anclas de la vista en píxeles y "feats".
      - `view` tensor (B,) con el índice de vista de cada corte: el backbone procesa el batch completo
        y cada cabeza recibe solo sus cortes. Devuelve "feats" y "views" = {vista: {"cls", "reg",
        "anchors", "idx"}}, donde "idx" son las posiciones de esos cortes en el batch.
    """

    def __init__(self, cbam: bool = True):
        super().__init__()
        self.backbone = FundidoraFPN(cbam)
        tam = config.INPUT_SIZE // config.STRIDE
        self.heads = nn.ModuleDict()
        for v in VIEWS:
            a = config.ANCHORS[v]
            self.heads[v] = AnchorHead(self.backbone.widths[2], len(CLASSES[v]), len(a["scales"]) * len(a["ratios"]),
                                       attention=cbam and config.HEAD_ATTENTION)
            anclas = generar_anclas_grilla(tam, config.STRIDE, a["scales"], a["ratios"])
            self.register_buffer(f"anchors_{v}", torch.tensor(anclas, dtype=torch.float32), persistent=False)

    def forward(self, x, view):
        feats = self.backbone(x)
        if isinstance(view, str):
            cls, reg = self.heads[view](feats["p3"])
            return {"cls": cls, "reg": reg, "anchors": getattr(self, f"anchors_{view}"), "feats": feats, "view": view}
        out = {"feats": feats, "views": {}}
        for vi, v in enumerate(VIEWS):
            idx = (view == vi).nonzero(as_tuple=True)[0]
            if idx.numel():
                cls, reg = self.heads[v](feats["p3"][idx])
                out["views"][v] = {"cls": cls, "reg": reg, "anchors": getattr(self, f"anchors_{v}"), "idx": idx}
        return out

    def gammas(self) -> dict:
        """γ de cada capa con atención: {"enc3": ..., "enc4": ..., "head_axial": ...}."""
        out = {}
        for i, m in enumerate(self.backbone.attn, start=1):
            if isinstance(m, GatedCBAM):
                out[f"enc{i}"] = float(m.gamma.detach())
        for v, h in self.heads.items():
            if isinstance(h.attn, GatedCBAM):
                out[f"head_{v}"] = float(h.attn.gamma.detach())
        return out

    def attention_modules(self) -> dict:
        mods = {f"enc{i}": m for i, m in enumerate(self.backbone.attn, start=1) if isinstance(m, GatedCBAM)}
        mods |= {f"head_{v}": h.attn for v, h in self.heads.items() if isinstance(h.attn, GatedCBAM)}
        return mods


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ======================================================================================
# Cajas, NMS y pérdida
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
    """Offsets (tx, ty, tw, th) que llevan cada ancla a su caja real (taller 3), para N pares."""
    aw, ah = anclas[:, 2] - anclas[:, 0], anclas[:, 3] - anclas[:, 1]
    acx, acy = anclas[:, 0] + aw / 2, anclas[:, 1] + ah / 2
    gw, gh = cajas[:, 2] - cajas[:, 0], cajas[:, 3] - cajas[:, 1]
    gcx, gcy = cajas[:, 0] + gw / 2, cajas[:, 1] + gh / 2
    return torch.stack([(gcx - acx) / aw, (gcy - acy) / ah, torch.log(gw / aw), torch.log(gh / ah)], 1)


def decode_boxes(anclas: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Inversa de `calcular_offsets`: anclas (A, 4) + offsets (A, 4) -> cajas (A, 4) (taller 3)."""
    aw, ah = anclas[:, 2] - anclas[:, 0], anclas[:, 3] - anclas[:, 1]
    acx, acy = anclas[:, 0] + aw / 2, anclas[:, 1] + ah / 2
    tw, th = deltas[:, 2].clamp(max=4.0), deltas[:, 3].clamp(max=4.0)    # evita exp() desbordado
    pcx, pcy = deltas[:, 0] * aw + acx, deltas[:, 1] * ah + acy
    pw, ph = torch.exp(tw) * aw, torch.exp(th) * ah
    return torch.stack([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2], 1)


def assign_targets(anchors: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
    """Etiqueta las anclas de una imagen comparándolas por IoU con sus cajas reales (taller 3).

    Cada ancla se asigna a la caja real con la que tiene mayor IoU: positiva si IoU >= POS_IOU,
    negativa si < NEG_IOU, ignorada en medio. La mejor ancla de cada caja real siempre es positiva.
    Devuelve labels (A,) en {1, 0, −1}, objetivo de clase (A, C) y offsets objetivo (A, 4).
    """
    a, c = anchors.shape[0], gt.shape[0]
    labels = torch.full((a,), -1, dtype=torch.long, device=anchors.device)
    cls_t = torch.zeros(a, c, device=anchors.device)
    reg_t = torch.zeros(a, 4, device=anchors.device)
    clases = valid.nonzero(as_tuple=True)[0]
    if clases.numel() == 0:                       # corte sin regiones: todas las anclas son negativas
        labels[:] = 0
        return labels, cls_t, reg_t
    cajas = gt[clases]
    ious = box_iou(anchors, cajas)                # (A, G)
    max_iou, g = ious.max(1)
    labels[max_iou < config.NEG_IOU] = 0
    labels[max_iou >= config.POS_IOU] = 1
    mejor = ious.argmax(0)                        # al menos un ancla positiva por caja real
    labels[mejor] = 1
    g[mejor] = torch.arange(len(clases), device=anchors.device)
    pos = labels == 1
    cls_t[pos, clases[g[pos]]] = 1.0
    reg_t[pos] = calcular_offsets(anchors[pos], cajas[g[pos]])
    return labels, cls_t, reg_t


def sample_anchors(labels: torch.Tensor, batch_size: int = config.ANCHOR_BATCH, pos_fraction: float = config.POS_FRACTION):
    """Muestra hasta `batch_size` anclas, como máximo `pos_fraction` positivas (taller 3).
    Devuelve los índices seleccionados (positivas + negativas) y solo las positivas."""
    positivos = torch.where(labels == 1)[0]
    negativos = torch.where(labels == 0)[0]
    n_pos = min(len(positivos), int(batch_size * pos_fraction))
    n_neg = min(len(negativos), batch_size - n_pos)
    positivos = positivos[torch.randperm(len(positivos), device=labels.device)[:n_pos]]
    negativos = negativos[torch.randperm(len(negativos), device=labels.device)[:n_neg]]
    return torch.cat([positivos, negativos]), positivos


def detection_loss(out: dict, gt: torch.Tensor, valid: torch.Tensor, lambdas: dict = config.LAMBDAS):
    """L = λ_cls · BCE(logits de las anclas muestreadas) + λ_box · SmoothL1(offsets de las positivas).

    Igual que la pérdida de la RPN del taller 3, con un logit por clase en lugar de uno de objectness.
    Se promedia por imagen y luego sobre el batch. Devuelve (total, términos sin λ).
    """
    cls, reg, anchors = out["cls"].float(), out["reg"].float(), out["anchors"]
    l_cls, l_box = [], []
    for b in range(cls.shape[0]):
        labels, cls_t, reg_t = assign_targets(anchors, gt[b], valid[b])
        sel, pos = sample_anchors(labels)
        l_cls.append(F.binary_cross_entropy_with_logits(cls[b, sel], cls_t[sel]))
        l_box.append(F.smooth_l1_loss(reg[b, pos], reg_t[pos]) if len(pos) else reg[b].sum() * 0)
    terms = {"cls": torch.stack(l_cls).mean(), "box": torch.stack(l_box).mean()}
    total = sum(lambdas[k] * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}


def mixed_loss(out: dict, m: torch.Tensor, lambdas: dict = config.LAMBDAS):
    """Pérdida de un batch con cortes de varias vistas: la de cada vista, ponderada por su número de
    cortes. Devuelve (total, {vista: términos})."""
    total, per_view = 0.0, {}
    for v, o in out["views"].items():
        gt, valid = region_boxes(m[o["idx"]], len(CLASSES[v]))
        loss, terms = detection_loss(o, gt, valid, lambdas)
        total = total + loss * (len(o["idx"]) / m.shape[0])
        per_view[v] = terms | {"n": len(o["idx"])}
    return total, per_view


@torch.no_grad()
def postprocess(out: dict, score_thr: float = config.SCORE_THR, iou_thr: float = config.NMS_IOU,
                pre_nms: int = config.PRE_NMS, max_det: int = config.MAX_DET):
    """Salida de la red -> por imagen, {"boxes" (N, 4), "scores" (N,), "labels" (N,)} en CPU.

    Por clase: puntaje = sigmoide del logit de cada ancla, umbral, top-`pre_nms`, NMS propio y como
    máximo `max_det` cajas (1: anatómicamente hay a lo sumo una de cada región por corte).
    """
    # En CPU: el NMS es un bucle secuencial y en GPU cada iteración obligaría a sincronizar
    scores = torch.sigmoid(out["cls"].float()).cpu()
    reg, anchors = out["reg"].float().cpu(), out["anchors"].cpu()
    results = []
    for i in range(scores.shape[0]):
        cajas = decode_boxes(anchors, reg[i]).clamp(0, config.INPUT_SIZE)
        bx, sc, lb = [], [], []
        for k in range(scores.shape[2]):
            s = scores[i, :, k]
            keep = (s > score_thr).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            keep = keep[s[keep].argsort(descending=True)[:pre_nms]]
            k_nms = nms(cajas[keep], s[keep], iou_thr)[:max_det]
            bx.append(cajas[keep][k_nms]); sc.append(s[keep][k_nms]); lb.append(torch.full((k_nms.numel(),), k))
        if bx:
            results.append({"boxes": torch.cat(bx), "scores": torch.cat(sc), "labels": torch.cat(lb)})
        else:
            results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
    return results


# ======================================================================================
# Métricas
# ======================================================================================

def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP interpolada en 101 puntos de recall (como COCO)."""
    prec = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def detection_metrics(preds: list, gt_boxes: np.ndarray, gt_valid: np.ndarray, class_names,
                      thresholds=tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))) -> dict:
    """AP por clase a cada umbral de IoU, mAP@0.5, mAP@[0.5:0.95] e IoU promedio.

    Hay a lo sumo una caja real por clase y corte. «IoU promedio»: para cada caja real, IoU con la
    predicción de mayor puntaje de su clase en ese corte (0 si no hay ninguna).
    """
    res = {"ap": {}, "mean_iou": {}}
    for c, name in enumerate(class_names):
        scores, ious, top_iou = [], [], []
        n_gt = int(gt_valid[:, c].sum())
        for i, p in enumerate(preds):
            sel = (p["labels"] == c).numpy()
            if not sel.any():
                if gt_valid[i, c]:
                    top_iou.append(0.0)
                continue
            s, bx = p["scores"].numpy()[sel], p["boxes"][sel]
            if gt_valid[i, c]:
                iou = box_iou(bx, torch.as_tensor(gt_boxes[i, c:c + 1], dtype=torch.float32))[:, 0].numpy()
                top_iou.append(float(iou[np.argmax(s)]))
            else:
                iou = np.zeros(len(s))
            order = np.argsort(-s)
            scores += list(s[order]); ious.append(iou[order])
        scores = np.asarray(scores)
        ap = {}
        if n_gt:
            for thr in thresholds:
                # En cada imagen solo la primera predicción (por puntaje) con IoU >= thr es acierto
                tp_list = []
                for iou in ious:
                    hit = iou >= thr
                    first = np.zeros_like(hit)
                    if hit.any():
                        first[np.argmax(hit)] = True
                    tp_list.append(first)
                tp = np.concatenate(tp_list) if tp_list else np.zeros(0, bool)
                order = np.argsort(-scores, kind="stable")
                tp = tp[order].astype(float)
                ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
                recall = ctp / n_gt
                precision = ctp / np.maximum(ctp + cfp, 1e-9)
                ap[float(thr)] = average_precision(recall, precision)
        res["ap"][name] = ap
        res["mean_iou"][name] = float(np.mean(top_iou)) if top_iou else float("nan")
    names = [n for n in class_names if res["ap"][n]]
    nan = float("nan")
    res["AP50"] = {n: res["ap"][n][0.5] for n in names}
    res["mAP50"] = float(np.mean([res["ap"][n][0.5] for n in names])) if names else nan
    res["mAP50_95"] = float(np.mean([np.mean(list(res["ap"][n].values())) for n in names])) if names else nan
    res["mAP75"] = float(np.mean([res["ap"][n][0.75] for n in names])) if names else nan
    res["mIoU"] = float(np.nanmean(list(res["mean_iou"].values()))) if names else nan
    return res


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC ROC por rangos (Mann-Whitney), con empates promediados."""
    y = y.astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y: np.ndarray, prob: np.ndarray, class_names, thr: float = 0.5) -> dict:
    """F1 y AUC de la presencia de cada clase en el corte (puntaje = máximo entre sus anclas)."""
    out = {"f1": {}, "auc": {}}
    for c, name in enumerate(class_names):
        pred = prob[:, c] >= thr
        tp, fp, fn = (pred & y[:, c]).sum(), (pred & ~y[:, c]).sum(), (~pred & y[:, c]).sum()
        out["f1"][name] = float(2 * tp / max(2 * tp + fp + fn, 1))
        out["auc"][name] = roc_auc(y[:, c], prob[:, c])
    out["macro_f1"] = float(np.mean(list(out["f1"].values())))
    out["macro_auc"] = float(np.nanmean(list(out["auc"].values())))
    return out


@torch.no_grad()
def run_inference(model, loader, device, view: str, amp: bool = config.AMP, lambdas=config.LAMBDAS):
    """Pasa por el modelo un loader de cortes de UNA vista. Devuelve predicciones tras NMS, cajas
    reales, presencia real, probabilidad de cada clase en el corte, pérdida media y filas."""
    model.eval()
    nc = len(CLASSES[view])
    preds, gts, valids, probs, rows, losses = [], [], [], [], [], []
    for x_u8, m_u8, idx, _ in loader:
        x, m = to_device(x_u8, m_u8, device)
        with autocast(device, amp):
            out = model(x, view)
        gt, valid = region_boxes(m, nc)
        _, terms = detection_loss(out, gt, valid, lambdas)
        losses.append(terms)
        preds += postprocess(out)
        gts.append(gt.cpu()); valids.append(valid.cpu())
        probs.append(torch.sigmoid(out["cls"].float()).amax(1).cpu()); rows.append(idx)
    return {"preds": preds, "gt": torch.cat(gts).numpy(), "valid": torch.cat(valids).numpy(),
            "prob": torch.cat(probs).numpy(), "rows": torch.cat(rows).numpy(),
            "loss": {k: float(np.mean([t[k] for t in losses])) for k in losses[0]}}


def evaluate_views(model, subsets: dict, device, workers: int = config.EVAL_WORKERS, lambdas=config.LAMBDAS,
                   batch_size: int = 32):
    """Métricas por vista sobre `subsets` {vista: DataFrame}. Devuelve {vista: {...}} con la
    pérdida, las métricas de detección y de clasificación y la salida cruda de `run_inference`."""
    out = {}
    for view, df in subsets.items():
        loader = make_loader(SliceDataset(df), batch_size, workers=workers)
        r = run_inference(model, loader, device, view, lambdas=lambdas)
        out[view] = {"raw": r, "loss": r["loss"],
                     "det": detection_metrics(r["preds"], r["gt"], r["valid"], CLASSES[view]),
                     "cls": classification_metrics(r["valid"], r["prob"], CLASSES[view])}
    return out


def _flat_metrics(ev: dict, prefix: str) -> dict:
    """Aplana las métricas por vista a columnas `{prefix}_{métrica}_{vista}` más la media de las vistas."""
    row = {}
    keys = {"cls": lambda e: e["loss"]["cls"], "box": lambda e: e["loss"]["box"],
            "mAP50": lambda e: e["det"]["mAP50"], "mAP50_95": lambda e: e["det"]["mAP50_95"],
            "mIoU": lambda e: e["det"]["mIoU"], "macro_f1": lambda e: e["cls"]["macro_f1"],
            "macro_auc": lambda e: e["cls"]["macro_auc"]}
    for k, f in keys.items():
        vals = {v: f(e) for v, e in ev.items()}
        row |= {f"{prefix}_{k}_{v}": x for v, x in vals.items()}
        row[f"{prefix}_{k}"] = float(np.nanmean(list(vals.values())))
    return row


# ======================================================================================
# Fusión 3D de las tres vistas
# ======================================================================================

def _view_box_3d(view: str, slices: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Caja 3D (z0, y0, x0, z1, y1, x1) en vóxeles del cubo a partir de las cajas 2D de una vista
    (unión de los cortes donde se detectó la región)."""
    n = config.INPUT_SIZE
    s0, s1 = slices.min(), slices.max() + 1
    c0, r0 = boxes[:, 0].min(), boxes[:, 1].min()
    c1, r1 = boxes[:, 2].max(), boxes[:, 3].max()
    if view == "axial":          # filas = y, columnas = x, cortes = z
        return np.array([s0, r0, c0, s1, r1, c1], float)
    if view == "coronal":        # filas = z invertido, columnas = x, cortes = y
        return np.array([n - r1, s0, c0, n - r0, s1, c1], float)
    return np.array([n - r1, c0, s0, n - r0, c1, s1], float)    # sagital: columnas = y, cortes = x


def iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.prod(np.clip(np.minimum(a[3:], b[3:]) - np.maximum(a[:3], b[:3]), 0, None))
    va, vb = np.prod(a[3:] - a[:3]), np.prod(b[3:] - b[:3])
    return float(inter / max(va + vb - inter, 1e-9))


def fuse_3d(per_view: dict, thr: float = config.FUSION_SCORE_THR) -> dict:
    """Caja 3D de cada región a partir de las detecciones 2D de las tres vistas de un caso.

    `per_view[vista]` = lista de (corte, predicción) de todos los cortes de esa vista. Cada vista
    da una caja 3D (unión de sus cortes con la región, puntaje >= thr); la caja fusionada es la media
    ponderada de las coordenadas con `config.VIEW_WEIGHTS` (QuickNAT: 0,4 / 0,4 / 0,2).
    En sagital el coxal izquierdo son los cortes con x por encima de la línea media (LPS: x crece
    hacia la izquierda) y el derecho los de abajo; la línea media es el centro del sacro detectado.
    """
    found = {}
    for view, items in per_view.items():
        names = CLASSES[view]
        dets = {n: [] for n in names}
        for s, p in items:
            for bx, sc, lb in zip(p["boxes"].numpy(), p["scores"].numpy(), p["labels"].numpy()):
                if sc >= thr:
                    dets[names[lb]].append((s, bx))
        if view == "sagittal":
            sac = [s for s, _ in dets["sacrum"]]
            mid = np.mean(sac) if sac else config.INPUT_SIZE / 2
            hip = dets.pop("hipbone")
            dets["left_hipbone"] = [(s, b) for s, b in hip if s > mid]
            dets["right_hipbone"] = [(s, b) for s, b in hip if s <= mid]
        for r, d in dets.items():
            if d:
                found.setdefault(r, {})[view] = _view_box_3d(view, np.array([s for s, _ in d]), np.stack([b for _, b in d]))
    fused = {}
    for r, boxes in found.items():
        w = np.array([config.VIEW_WEIGHTS[v] for v in boxes])
        fused[r] = {"views": boxes, "fused": (np.stack(list(boxes.values())) * w[:, None]).sum(0) / w.sum()}
    return fused


def gt_boxes_3d(cid: str) -> dict:
    """Caja 3D real de cada región en el cubo (vóxeles), a partir de la máscara del caché."""
    reg = VIEW_LUT["axial"][np.load(config.CACHE_DIR / f"{cid}_msk.npy", mmap_mode="r")]
    out = {}
    for c, r in enumerate(config.REGION_NAMES, start=1):
        idx = np.argwhere(reg == c)
        if len(idx):
            out[r] = np.concatenate([idx.min(0), idx.max(0) + 1]).astype(float)
    return out


# ======================================================================================
# Grad-CAM, latencia y prueba de reconstrucción
# ======================================================================================

def grad_cam(model, x: torch.Tensor, view: str, cls: int, layer: str = "p3") -> torch.Tensor:
    """Grad-CAM (Selvaraju et al., 2017) sobre el mapa `layer` del backbone para la clase `cls`.

    Objetivo: el logit máximo de esa clase entre todas las anclas. Pesos = media espacial del
    gradiente; mapa = ReLU(Σ pesos·activaciones), interpolado al tamaño de la entrada y normalizado
    a [0, 1] por imagen. Devuelve (B, H, W).
    """
    model.eval()
    with torch.enable_grad():
        out = model(x, view)
        f = out["feats"][layer]
        f.retain_grad()
        model.zero_grad(set_to_none=True)
        out["cls"][:, :, cls].amax(1).sum().backward()
        w = f.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * f).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam / cam.flatten(1).amax(1).clamp(min=1e-8)[:, None, None]


def cam_box_scores(cam: torch.Tensor, box: torch.Tensor) -> tuple[float, float, float]:
    """Energía del CAM dentro de la caja real, fracción del área que ocupa la caja y si el máximo
    del CAM cae dentro (pointing game). Energía ≈ área: el mapa no se concentra en la región."""
    h, w = cam.shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    inside = torch.zeros_like(cam, dtype=torch.bool)
    inside[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    energy = float(cam[inside].sum() / cam.sum().clamp(min=1e-8))
    peak = int(cam.argmax())
    return energy, float(inside.float().mean()), float(inside.flatten()[peak])


@torch.no_grad()
def measure_latency(model, device: torch.device, view: str = "coronal") -> dict:
    """Latencia por imagen (batch 1, entrada ya preprocesada): red + decodificación + NMS, en ms."""
    model = model.to(device).eval()
    x = torch.rand(1, config.K_SLICES, config.INPUT_SIZE, config.INPUT_SIZE, device=device)
    times = []
    for i in range(config.LATENCY_WARMUP + config.LATENCY_RUNS):
        sync(device)
        t0 = time.perf_counter()
        postprocess(model(x, view))
        sync(device)
        if i >= config.LATENCY_WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    return {"device": device.type, "median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


class ReconstructionProbe(nn.Module):
    """Prueba de la información que conserva un mapa de características.

    El mapa (C, h, w) se lleva a la resolución de la entrada y se estandariza por canal, se comprime
    con una conv 9x9 de stride 6 a ≤ 15 canales, se interpola de vuelta y una conv 3x3 reconstruye el
    corte central. Con el backbone congelado, un error bajo indica que las características conservan
    la estructura de la imagen.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        k = config.PROBE_KERNEL
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.enc = nn.Conv2d(in_channels, config.PROBE_BOTTLENECK, k, stride=config.PROBE_STRIDE, padding=k // 2)
        self.dec = nn.Conv2d(config.PROBE_BOTTLENECK, 1, 3, padding=1)

    def forward(self, f, size):
        f = F.interpolate(f, size=size, mode="bilinear", align_corners=False) if f.shape[-2:] != size else f
        z = self.enc(self.norm(f))
        return self.dec(F.interpolate(z, size=size, mode="bilinear", align_corners=False))


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM medio por imagen para tensores (B, 1, H, W) en [0, 1] (ventana gaussiana)."""
    g = torch.exp(-((torch.arange(window, device=a.device) - window // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    k = (g[:, None] * g[None, :])[None, None]
    blur = lambda t: F.conv2d(t, k, padding=window // 2)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.flatten(1).mean(1)


# ======================================================================================
# Figuras
# ======================================================================================

def plot_detections(images, gts, valids, preds, titles, class_names, path, ncols: int = 4):
    """Corte central con cajas reales (discontinuas) y predichas (continuas, con puntaje)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.0 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, img, gt, val, p, title in zip(axes.ravel(), images, gts, valids, preds, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for c, name in enumerate(class_names):
            col = config.REGION_COLORS[name]
            if val[c]:
                x0, y0, x1, y1 = gt[c]
                ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.2, ls="--"))
        for bx, s, lb in zip(p["boxes"].numpy(), p["scores"].numpy(), p["labels"].numpy()):
            if s < config.PLOT_SCORE_THR:
                continue
            col = config.REGION_COLORS[class_names[lb]]
            x0, y0, x1, y1 = bx
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.8))
            ax.text(x0, y0 - 2, f"{s:.2f}", color=col, fontsize=7, va="bottom")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    names = list(dict.fromkeys(n for v in VIEWS for n in CLASSES[v]))
    handles = [patches.Patch(color=config.REGION_COLORS[r], label=r) for r in names]
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(names), fontsize=8, frameon=False, title_fontsize=8,
               title=f"discontinua = caja real · continua = predicción (puntaje ≥ {config.PLOT_SCORE_THR:.2f})")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# Entrenamiento y evaluación
# ======================================================================================

def build_model(cfg: dict) -> ThreeViewDetector:
    """Modelo de tres vistas. Un checkpoint de otra arquitectura (p. ej. el archivado) no se carga."""
    if cfg.get("arch") != "three_view_anchors":
        raise ValueError(f"El checkpoint es de otra arquitectura ({cfg.get('arch', 'desconocida')}); "
                         "este código solo carga el modelo de tres vistas")
    return ThreeViewDetector(cbam=cfg["cbam"])


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"]


def make_loader(ds, batch_size, sampler=None, workers=config.EVAL_WORKERS):
    """DataLoader. Con workers > 0 los procesos son persistentes (se lanzan una vez, no en cada época)."""
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=workers,
                                       persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def split_views(index: pd.DataFrame, split: str, stride: int = 1) -> dict:
    """{vista: cortes del split} tomando 1 de cada `stride` cortes."""
    d = index[(index.split == split) & (index["slice"] % stride == 0)]
    return {v: d[d.view == v].reset_index(drop=True) for v in VIEWS}


def train_eval_views(index: pd.DataFrame) -> dict:
    """Subconjunto fijo de cortes de train por vista para medir métricas de train en cada época
    (misma proporción con/sin regiones que el entrenamiento)."""
    out = {}
    for v in VIEWS:
        d = index[(index.split == "train") & (index.view == v)]
        n_neg = int(round(config.TRAIN_EVAL_SLICES * config.NEG_FRAC))
        pos = d[d.positive].sample(config.TRAIN_EVAL_SLICES - n_neg, random_state=config.SEED)
        neg = d[~d.positive & d.bone]
        neg = neg.sample(min(n_neg, len(neg)), random_state=config.SEED)
        out[v] = pd.concat([pos, neg]).reset_index(drop=True)
    return out


def run_config(cbam: bool, epochs: int, samples: int) -> dict:
    return {"arch": "three_view_anchors", "cbam": cbam, "cbam_blocks": list(config.CBAM_BLOCKS) if cbam else [],
            "head_attention": cbam and config.HEAD_ATTENTION, "gamma_init": config.GAMMA_INIT,
            "views": list(VIEWS), "view_classes": {v: list(c) for v, c in CLASSES.items()},
            "k_slices": config.K_SLICES, "input_size": config.INPUT_SIZE, "cube_mm": config.CUBE_MM,
            "hu_window": list(config.HU_WINDOW), "stride": config.STRIDE,
            "anchors": {v: {k: list(x) for k, x in a.items()} for v, a in config.ANCHORS.items()},
            "pos_iou": config.POS_IOU, "neg_iou": config.NEG_IOU, "lambdas": dict(config.LAMBDAS),
            "epochs": epochs, "samples_per_view": samples, "batch_size": config.BATCH_SIZE, "lr": config.LR,
            "weight_decay": config.WEIGHT_DECAY, "warmup_epochs": config.WARMUP_EPOCHS,
            "early_stop_patience": config.EARLY_STOP_PATIENCE, "early_stop_start": config.EARLY_STOP_START, "neg_frac": config.NEG_FRAC, "val_stride": config.VAL_STRIDE,
            "augment": {k.removeprefix("AUG_").lower(): getattr(config, k) for k in dir(config) if k.startswith("AUG_")},
            "amp": config.AMP, "seed": config.SEED, "backbone_init": None, "backbone_frozen": False}


def _optimizer(model):
    """AdamW con weight decay solo en los pesos de las convoluciones. Sin decay en γ (lo empujaría a 0
    y sesgaría la medida del aporte de la atención), ni en BatchNorm ni en los sesgos (práctica estándar)."""
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.WEIGHT_DECAY},
                              {"params": no_decay, "weight_decay": 0.0}], lr=config.LR)


def train(args) -> Path:
    seed_everything()
    device = get_device(args.device)
    cbam = not args.no_cbam
    cfg = run_config(cbam, args.epochs, args.samples)
    name = args.name or (config.RUN_CBAM if cbam else config.RUN_NO_CBAM)
    out_dir = config.DETECTION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index()
    tr = index[index.split == "train"].reset_index(drop=True)
    sampler = EpochSampler(tr, args.samples, config.NEG_FRAC)
    train_loader = make_loader(SliceDataset(tr), config.BATCH_SIZE, sampler=sampler, workers=args.workers)
    val_sets = split_views(index, "val", config.VAL_STRIDE)
    tr_eval = train_eval_views(index)

    model = ThreeViewDetector(cbam).to(device)
    opt = _optimizer(model)
    steps_per_epoch = math.ceil(len(sampler) / config.BATCH_SIZE)
    total, warm = args.epochs * steps_per_epoch, config.WARMUP_EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")

    start, best, history = 0, -1.0, []
    last_path = out_dir / "last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        torch.set_rng_state(ck["rng"])
        print(f"Reanudando {name} desde la época {start}")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{name}: {count_params(model):,} parámetros | {device} | {steps_per_epoch} pasos por época "
          f"({args.samples} cortes por vista) | val {sum(len(d) for d in val_sets.values())} cortes")
    if not history:
        history.append({"epoch": -1, **{f"gamma_{k}": v for k, v in model.gammas().items()}})   # γ antes de entrenar

    for epoch in range(start, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        t0, acc, skipped = time.time(), [], 0
        for x_u8, m_u8, _, vid in train_loader:
            x, m = to_device(x_u8, m_u8, device, augment=True)
            with autocast(device):
                out = model(x, vid.to(device))
            loss, per_view = mixed_loss(out, m, config.LAMBDAS)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):                     # paso con pérdida no finita: se salta
                skipped += 1; sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            if not torch.isfinite(gnorm) and not scaler.is_enabled():
                skipped += 1; opt.zero_grad(set_to_none=True); sched.step()   # gradiente no finito: se salta
                continue
            scaler.step(opt); scaler.update(); sched.step()  # en CUDA el GradScaler ya salta los pasos con inf
            acc += [{"view": v, **t} for v, t in per_view.items()]
        t_train = time.time() - t0
        acc = pd.DataFrame(acc)
        wmean = lambda g, k: float(np.average(g[k], weights=g["n"]))
        ev_val = evaluate_views(model, val_sets, device)
        ev_tr = evaluate_views(model, tr_eval, device)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0, "train_time_s": t_train,
               "skipped_steps": skipped,
               "train_cls": wmean(acc, "cls"), "train_box": wmean(acc, "box"),
               **{f"train_{k}_{v}": wmean(g, k) for v, g in acc.groupby("view") for k in ("cls", "box")},
               **{k: v for k, v in _flat_metrics(ev_tr, "train").items() if not k.startswith(("train_cls", "train_box"))},
               **_flat_metrics(ev_val, "val"),
               **{f"gamma_{k}": v for k, v in model.gammas().items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if row["val_mAP50_95"] > best:
            best = row["val_mAP50_95"]
            torch.save({**state, "val_mAP50_95": best}, out_dir / "best.pth")
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "history": history, "rng": torch.get_rng_state()}, last_path)
        gam = "  ".join(f"γ {k} {v:+.3f}" for k, v in model.gammas().items())
        row["train_loss"] = config.LAMBDAS["cls"] * row["train_cls"] + config.LAMBDAS["box"] * row["train_box"]
        print(f"[{name}] época {epoch + 1}/{args.epochs}  loss {row['train_loss']:.3f}  "
              f"val mAP50 {row['val_mAP50']:.3f}  mAP50-95 {row['val_mAP50_95']:.3f}  mIoU {row['val_mIoU']:.3f}  "
              f"F1 {row['val_macro_f1']:.3f}  ({row['time_s']:.0f} s)  {gam}"
              + (f"  pasos saltados {skipped}" if skipped else ""), flush=True)
        # Parada temprana: épocas desde la mejor val mAP50-95
        vals = [h["val_mAP50_95"] for h in history if h.get("epoch", -1) >= 0]
        sin_mejora = len(vals) - 1 - int(np.argmax(vals))
        if epoch + 1 >= config.EARLY_STOP_START and sin_mejora >= config.EARLY_STOP_PATIENCE:
            print(f"[{name}] parada temprana: {sin_mejora} épocas sin mejorar val mAP50-95 "
                  f"(mejor {max(vals):.3f} en la época {int(np.argmax(vals)) + 1})", flush=True)
            break
    return out_dir


def overfit(args) -> Path:
    """Prueba de correctitud: sobreajustar un batch fijo pequeño de cada vista. Si el modelo, la
    asignación de anclas, la pérdida, la decodificación y el NMS están bien, debe quedar casi perfecto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_everything()
    device = get_device(args.device)
    cbam = not args.no_cbam
    cfg = run_config(cbam, 0, 0)
    name = f"overfit_{config.RUN_CBAM if cbam else config.RUN_NO_CBAM}"
    out_dir = config.DETECTION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index()
    tr = index[(index.split == "train") & index.positive]
    n = config.OVERFIT_SLICES
    picks = []
    for v in VIEWS:
        d = tr[tr.view == v]
        cols = [f"px_{c}" for c in CLASSES[v]]
        full = d[(d[cols] >= 200).all(axis=1)]
        pick = full.groupby("case_id").sample(1, random_state=config.SEED).sample(n - n // 4, random_state=config.SEED)
        picks.append(pd.concat([pick, d.drop(full.index).sample(n // 4, random_state=config.SEED)]))
    pick = pd.concat(picks).reset_index(drop=True)                      # un batch con las tres vistas
    ds = SliceDataset(pick)
    x_u8, m_u8, _, vid = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds))))
    x, m = to_device(x_u8, m_u8, device)
    vid = vid.to(device)

    model = ThreeViewDetector(cbam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.OVERFIT_LR)
    hist = []
    for it in range(args.iters):
        model.train()
        loss, per_view = mixed_loss(model(x, vid), m)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        hist.append({"iter": it, "loss": float(loss.detach()),
                     **{f"{k}_{v}": t[k] for v, t in per_view.items() for k in ("cls", "box")},
                     **{f"gamma_{k}": g for k, g in model.gammas().items()}})
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {hist[-1]['loss']:.4f}", flush=True)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

    model.eval()
    metrics = {"n_slices_per_view": n, "iters": args.iters, "final_loss": hist[-1]["loss"],
               "gammas": model.gammas(), "views": {}}
    imgs, gts, vals, preds, titles = [], [], [], [], []
    with torch.no_grad():
        out = model(x, vid)
    for v in VIEWS:
        o = out["views"][v]
        gt, valid = region_boxes(m[o["idx"]], len(CLASSES[v]))
        p = postprocess(o)
        det = detection_metrics(p, gt.cpu().numpy(), valid.cpu().numpy(), CLASSES[v])
        pv = pick.iloc[o["idx"].cpu().numpy()]
        metrics["views"][v] = {"mAP50": det["mAP50"], "mAP50_95": det["mAP50_95"], "mIoU": det["mIoU"],
                               "slices": [f"{c}:{s}" for c, s in zip(pv.case_id, pv["slice"])]}
        imgs += list(x[o["idx"][:4], config.K_SLICES // 2].cpu().numpy())
        gts.append(gt[:4].cpu().numpy()); vals.append(valid[:4].cpu().numpy()); preds += p[:4]
        titles += [f"{v} · caso {c} · corte {s}" for c, s in zip(pv.case_id[:4], pv["slice"][:4])]
    metrics["mAP50"] = float(np.mean([m["mAP50"] for m in metrics["views"].values()]))
    metrics["mAP50_95"] = float(np.mean([m["mAP50_95"] for m in metrics["views"].values()]))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(h.iter, h.loss, color="#333"); ax[0].set_yscale("log"); ax[0].set_title("Pérdida total (3 vistas)")
    for v in VIEWS:
        ax[1].plot(h.iter, h[f"cls_{v}"] + h[f"box_{v}"], label=v)
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Pérdida por vista")
    for a in ax:
        a.set_xlabel("iteración"); a.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=130); plt.close(fig)
    # Una figura por vista (las clases cambian en sagital)
    for k, v in enumerate(VIEWS):
        sl = slice(4 * k, 4 * k + 4)
        plot_detections(imgs[sl], gts[k], vals[k], preds[sl], titles[sl], CLASSES[v], out_dir / f"predictions_{v}.png")
    print(f"Overfit {name}: " + "  ".join(f"{v} mAP50 {m['mAP50']:.3f}" for v, m in metrics["views"].items()) + f" -> {out_dir}")
    return out_dir


def evaluate(args) -> dict:
    """Evalúa `best.pth` de un run sobre todos los cortes de validación de las tres vistas y guarda
    en su carpeta: metrics.json, per_slice.csv, fusion_3d.csv, predictions_<vista>.png,
    gradcam_<vista>.png, gradcam_scores.csv y la latencia."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_everything()
    device = get_device(args.device)
    run_dir = config.DETECTION_DIR / args.run
    model, cfg = load_model(run_dir / "best.pth", device)
    k = cfg["k_slices"]
    index = load_index()
    val_sets = split_views(index, "val", 1)
    ev = evaluate_views(model, val_sets, device, lambdas=cfg.get("lambdas", config.LAMBDAS))

    # Resultado por corte
    rows = []
    for v, e in ev.items():
        r, va = e["raw"], val_sets[v]
        for i, p in enumerate(r["preds"]):
            row = {"view": v, "case_id": va.case_id[i], "slice": int(va["slice"][i])}
            for c, nm in enumerate(CLASSES[v]):
                sel = p["labels"] == c
                iou = np.nan
                if r["valid"][i, c]:
                    iou = 0.0
                    if sel.any():
                        j = int(p["scores"][sel].argmax())
                        iou = float(box_iou(p["boxes"][sel][j:j + 1], torch.as_tensor(r["gt"][i, c:c + 1]))[0, 0])
                row |= {f"present_{nm}": bool(r["valid"][i, c]), f"score_{nm}": float(p["scores"][sel].max()) if sel.any() else 0.0,
                        f"iou_{nm}": iou, f"prob_{nm}": float(r["prob"][i, c])}
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "per_slice.csv", index=False)

    # Fusión 3D por caso: caja 3D de cada vista y fusionada, contra la caja 3D real
    fus_rows = []
    for cid in val_sets["axial"].case_id.unique():
        per_view = {v: [(int(val_sets[v]["slice"][i]), ev[v]["raw"]["preds"][i])
                        for i in np.flatnonzero(val_sets[v].case_id.to_numpy() == cid)] for v in VIEWS}
        fused, gt3 = fuse_3d(per_view), gt_boxes_3d(cid)
        for r, g in gt3.items():
            row = {"case_id": cid, "region": r}
            f = fused.get(r)
            for v in VIEWS:
                row[f"iou3d_{v}"] = iou_3d(f["views"][v], g) if f and v in f["views"] else 0.0
            row["iou3d_fused"] = iou_3d(f["fused"], g) if f else 0.0
            if f:
                cen_err = np.abs((f["fused"][:3] + f["fused"][3:]) / 2 - (g[:3] + g[3:]) / 2) * config.VOXEL_MM
                row["center_error_mm"] = float(np.linalg.norm(cen_err))
            fus_rows.append(row)
    fus = pd.DataFrame(fus_rows)
    fus.to_csv(run_dir / "fusion_3d.csv", index=False)

    # Ejemplos por vista: cortes con regiones de varios casos de validación
    rng = np.random.default_rng(config.SEED)
    examples = {}
    for v in VIEWS:
        va = val_sets[v]
        pos = va[va.positive]
        ex = pos.groupby("case_id").sample(1, random_state=config.SEED)
        ex = ex.iloc[rng.permutation(len(ex))[:8]].sort_values(["case_id", "slice"])
        ex_rows = [int(np.flatnonzero((va.case_id == c) & (va["slice"] == s))[0]) for c, s in zip(ex.case_id, ex["slice"])]
        examples[v] = ex_rows
        ds = SliceDataset(va)
        imgs = np.stack([ds[i][0][k // 2].numpy() / 255 for i in ex_rows])
        r = ev[v]["raw"]
        plot_detections(imgs, r["gt"][ex_rows], r["valid"][ex_rows], [r["preds"][i] for i in ex_rows],
                        [f"{v} · caso {va.case_id[i]} · corte {va['slice'][i]}" for i in ex_rows], CLASSES[v],
                        run_dir / f"predictions_{v}.png")

    # Grad-CAM por vista: energía dentro de la caja real y pointing game
    cam_rows = []
    for v in VIEWS:
        va, r = val_sets[v], ev[v]["raw"]
        ds = SliceDataset(va)
        pos = va[va.positive]
        sample = pos.sample(min(config.CAM_SAMPLES, len(pos)), random_state=config.SEED)
        s_rows = [int(np.flatnonzero((va.case_id == c) & (va["slice"] == s))[0]) for c, s in zip(sample.case_id, sample["slice"])]
        for layer in config.CAM_LAYERS:
            for st in range(0, len(s_rows), 16):
                chunk = s_rows[st:st + 16]
                x = torch.stack([ds[i][0] for i in chunk]).to(device).float() / 255
                for c, nm in enumerate(CLASSES[v]):
                    cams = grad_cam(model, x, v, c, layer).cpu()
                    for j, i in enumerate(chunk):
                        if r["valid"][i, c]:
                            e, a, hit = cam_box_scores(cams[j], torch.as_tensor(r["gt"][i, c]))
                            cam_rows.append({"view": v, "layer": layer, "region": nm, "energy_in_box": e,
                                             "box_area_frac": a, "pointing_hit": hit})
        # Figura: 4 ejemplos × clases × capas
        show = examples[v][:4]
        x = torch.stack([ds[i][0] for i in show]).to(device).float() / 255
        nc = len(CLASSES[v])
        cams = {(l, c): grad_cam(model, x, v, c, l).cpu().numpy() for l in config.CAM_LAYERS for c in range(nc)}
        ncols = 1 + len(config.CAM_LAYERS) * nc
        fig, axes = plt.subplots(len(show), ncols, figsize=(2.3 * ncols, 2.4 * len(show)), squeeze=False)
        for j, i in enumerate(show):
            base = x[j, k // 2].cpu().numpy()
            axes[j, 0].imshow(base, cmap="gray"); axes[j, 0].set_title(f"{va.case_id[i]} corte {va['slice'][i]}", fontsize=7)
            col = 1
            for l in config.CAM_LAYERS:
                for c, nm in enumerate(CLASSES[v]):
                    a = axes[j, col]
                    a.imshow(base, cmap="gray"); a.imshow(cams[(l, c)][j], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                    if r["valid"][i, c]:
                        x0, y0, x1, y1 = r["gt"][i, c]
                        a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1, ls="--"))
                    if j == 0:
                        a.set_title(f"{nm} · {l}", fontsize=7)
                    col += 1
        for a in axes.ravel():
            a.axis("off")
        plt.tight_layout(); fig.savefig(run_dir / f"gradcam_{v}.png", dpi=110); plt.close(fig)
    cam_df = pd.DataFrame(cam_rows)
    cam_df.to_csv(run_dir / "gradcam_scores.csv", index=False)
    cam_summary = (cam_df.assign(ratio=cam_df.energy_in_box / cam_df.box_area_frac)
                   .groupby(["view", "layer", "region"])[["energy_in_box", "box_area_frac", "ratio", "pointing_hit"]].mean())

    latency = [measure_latency(model, torch.device("cpu"))]
    if device.type != "cpu":
        latency.append(measure_latency(model, device))

    per_view = {v: {"loss": e["loss"], "detection": e["det"], "classification": e["cls"], "n_slices": len(val_sets[v])}
                for v, e in ev.items()}
    metrics = {"run": args.run, "config": cfg, "params": count_params(model), "gammas": model.gammas(),
               "views": per_view,
               "mean": {k: float(np.nanmean([per_view[v]["detection"][k] for v in VIEWS])) for k in ("mAP50", "mAP50_95", "mIoU")}
                       | {"macro_f1": float(np.mean([per_view[v]["classification"]["macro_f1"] for v in VIEWS])),
                          "macro_auc": float(np.nanmean([per_view[v]["classification"]["macro_auc"] for v in VIEWS]))},
               "fusion_3d": {c: float(fus[c].mean()) for c in fus.columns if c.startswith(("iou3d", "center"))},
               "gradcam": {f"{v}/{l}/{nm}": d for (v, l, nm), d in cam_summary.to_dict("index").items()},
               "latency": latency}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    m = metrics["mean"]
    print(f"{args.run}: mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50_95']:.3f}  mIoU {m['mIoU']:.3f}  F1 {m['macro_f1']:.3f}  "
          f"| IoU 3D fusionado {metrics['fusion_3d']['iou3d_fused']:.3f}  | "
          + "  ".join(f"{d['device']} {d['median_ms']:.1f} ms" for d in latency))
    for v in VIEWS:
        d = per_view[v]["detection"]
        print(f"  {v:8s} mAP50 {d['mAP50']:.3f}  mAP50-95 {d['mAP50_95']:.3f}  mIoU {d['mIoU']:.3f}")
    print("  γ:", {k: round(g, 3) for k, g in model.gammas().items()})
    return metrics


def probe(args) -> dict:
    """Prueba de reconstrucción (backbone congelado) sobre el mapa `config.PROBE_LAYER` del run,
    comparada con el mismo backbone sin entrenar, con la imagen de entrada y con predecir todo negro.
    Métrica principal: MSE en píxeles de hueso (el fondo domina el MSE global y el SSIM)."""
    seed_everything()
    device = get_device(args.device)
    run_dir = config.DETECTION_DIR / args.run
    model, cfg = load_model(run_dir / "best.pth", device)
    k, layer = cfg["k_slices"], config.PROBE_LAYER
    index = load_index()
    tr = index[(index.split == "train") & index.positive].reset_index(drop=True)
    va = index[(index.split == "val") & index.positive & (index["slice"] % 4 == 0)].reset_index(drop=True)
    va = va.groupby("view").sample(256, random_state=config.SEED).reset_index(drop=True)
    size = (config.INPUT_SIZE, config.INPUT_SIZE)

    torch.manual_seed(config.SEED)
    random_model = build_model(cfg).to(device).eval()
    sources = {"entrenado": lambda x: model.backbone(x)[layer],
               "sin_entrenar": lambda x: random_model.backbone(x)[layer],
               "imagen": lambda x: x}
    rng = np.random.default_rng(config.SEED)
    tr_rows = rng.choice(len(tr), config.PROBE_ITERS * 16, replace=True)
    tr_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(SliceDataset(tr), tr_rows.tolist()), batch_size=16,
                                            num_workers=args.workers)
    va_loader = make_loader(SliceDataset(va), 32)
    results, recon = {}, {}
    for name, feat in sources.items():
        torch.manual_seed(config.SEED)
        with torch.no_grad():
            x0 = next(iter(va_loader))[0][:2].to(device).float() / 255
            c_in = feat(x0).shape[1]
        pr = ReconstructionProbe(c_in).to(device)
        opt = torch.optim.Adam(pr.parameters(), lr=config.PROBE_LR)
        for x_u8, _, _, _ in tr_loader:
            x = x_u8.to(device).float() / 255
            with torch.no_grad():
                f = feat(x)
            loss = F.mse_loss(pr(f, size), x[:, k // 2:k // 2 + 1])
            opt.zero_grad(); loss.backward(); opt.step()
        pr.eval()
        results[name], recon[name] = _probe_eval(lambda x: pr(feat(x), size), va_loader, device, k)
        print(f"probe {args.run} [{name}] {results[name]}", flush=True)
    results["ceros"], recon["ceros"] = _probe_eval(lambda x: torch.zeros_like(x[:, :1]), va_loader, device, k)
    print(f"probe {args.run} [ceros] {results['ceros']}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1 + len(recon), figsize=(2.6 * (1 + len(recon)), 10))
    for i in range(4):
        axes[i, 0].imshow(recon["imagen"][0][i], cmap="gray", vmin=0, vmax=1); axes[i, 0].set_title("original", fontsize=8)
        for j, (name, (_, y)) in enumerate(recon.items(), start=1):
            axes[i, j].imshow(y[i], cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{name} · MSE hueso {results[name]['mse_bone']:.3f}", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    plt.tight_layout(); fig.savefig(run_dir / f"probe_{layer}.png", dpi=110); plt.close(fig)
    out = {"layer": layer, "bottleneck": config.PROBE_BOTTLENECK, "kernel": config.PROBE_KERNEL,
           "stride": config.PROBE_STRIDE, "iters": config.PROBE_ITERS, "results": results}
    (run_dir / f"probe_{layer}.json").write_text(json.dumps(out, indent=2))
    return out


@torch.no_grad()
def _probe_eval(predict, loader, device, k):
    """MSE global, MSE en hueso, PSNR y SSIM de `predict` sobre `loader`, y 4 ejemplos."""
    mse, mse_bone, ss, ejemplo = [], [], [], None
    for x_u8, _, _, _ in loader:
        x = x_u8.to(device).float() / 255
        t = x[:, k // 2:k // 2 + 1]
        y = predict(x).clamp(0, 1)
        err = (y - t) ** 2
        bone = (t > config.PROBE_BONE_THR).float()
        mse.append(err.flatten(1).mean(1).cpu())
        mse_bone.append(((err * bone).flatten(1).sum(1) / bone.flatten(1).sum(1).clamp(min=1)).cpu())
        ss.append(ssim(y, t).cpu())
        if ejemplo is None:
            ejemplo = (t[:4, 0].cpu().numpy(), y[:4, 0].cpu().numpy())
    mse = torch.cat(mse)
    return ({"mse": float(mse.mean()), "mse_bone": float(torch.cat(mse_bone).mean()),
             "psnr_db": float((10 * torch.log10(1 / mse.clamp(min=1e-10))).mean()), "ssim": float(torch.cat(ss).mean())},
            ejemplo)


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


def summary(args) -> pd.DataFrame:
    """Tabla comparativa de los runs evaluados -> results/detection/comparison.csv."""
    rows = []
    for mpath in sorted(config.DETECTION_DIR.glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        if m.get("config", {}).get("arch") != "three_view_anchors" or "views" not in m:
            continue
        lat = {d["device"]: d["median_ms"] for d in m["latency"]}
        row = {"run": m["run"], "cbam": m["config"]["cbam"], "params": m["params"], **m["mean"],
               **{f"{k}_{v}": m["views"][v]["detection"][k] for v in VIEWS for k in ("mAP50", "mAP50_95", "mIoU")},
               **m["fusion_3d"], **{f"gamma_{k}": g for k, g in m["gammas"].items()},
               "latency_cpu_ms": lat.get("cpu"), "latency_gpu_ms": lat.get("cuda", lat.get("mps"))}
        ppath = mpath.parent / f"probe_{config.PROBE_LAYER}.json"
        if ppath.exists():
            row |= {f"probe_mse_bone_{k}": v.get("mse_bone") for k, v in json.loads(ppath.read_text())["results"].items()}
        hpath = mpath.parent / "history.csv"
        if hpath.exists():
            row["train_min"] = pd.read_csv(hpath).time_s.sum() / 60
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("mAP50_95", ascending=False)
        df.to_csv(config.DETECTION_DIR / "comparison.csv", index=False)
        print(df.round(3).to_string(index=False))
    else:
        print("No hay runs evaluados.")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Detección de regiones pélvicas con tres vistas (FundidoraFPN + CBAM con γ + anclas).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="preprocesa los 100 casos al cubo y guarda el índice de cortes")
    c.add_argument("--overwrite", action="store_true")
    sub.add_parser("anchors", help="estadísticas de las cajas para elegir las anclas")

    def common(sp):
        sp.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
        sp.add_argument("--workers", type=int, default=config.WORKERS)

    t = sub.add_parser("train", help="entrena el modelo de tres vistas")
    common(t)
    t.add_argument("--no-cbam", action="store_true", help="variante sin atención (ablación)")
    t.add_argument("--epochs", type=int, default=config.EPOCHS)
    t.add_argument("--samples", type=int, default=config.SAMPLES_PER_VIEW, help="cortes por vista y época")
    t.add_argument("--name", default=None, help="carpeta del run (por defecto config.RUN_CBAM o RUN_NO_CBAM)")
    t.add_argument("--resume", action="store_true", help="continúa desde last.pth")

    o = sub.add_parser("overfit", help="prueba de correctitud sobre un batch fijo por vista")
    common(o)
    o.add_argument("--no-cbam", action="store_true")
    o.add_argument("--iters", type=int, default=config.OVERFIT_ITERS)

    e = sub.add_parser("evaluate", help="métricas, fusión 3D, Grad-CAM y latencia de un run")
    common(e)
    e.add_argument("--run", required=True)

    pr = sub.add_parser("probe", help="prueba de reconstrucción del extractor")
    common(pr)
    pr.add_argument("--run", required=True)

    sub.add_parser("summary", help="tabla comparativa de los runs evaluados")

    args = p.parse_args()
    if args.cmd == "cache":
        build_cache(args.overwrite)
    else:
        {"train": train, "overfit": overfit, "evaluate": evaluate, "probe": probe, "anchors": anchors,
         "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
