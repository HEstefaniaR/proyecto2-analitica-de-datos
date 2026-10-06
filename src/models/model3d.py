"""Fundidora 3D con detección 2D por corte (backbone 3D, cabeza de detección del modelo 2.5D).

Propuesta: `C.cambios/propuesta_red_3d.md`. Por ahora solo la DETECCIÓN: la cabeza de segmentación (y los
niveles P1 y P0 del decoder que la alimentarían) todavía no se hacen.

Reutiliza de `detection.py` el caché del cubo, las anclas, la cabeza con anclas (AnchorHead, con CBAM 2D + γ),
la asignación, la pérdida, el NMS y las métricas, así que las métricas por corte son comparables con la línea
base 2.5D.

Pipeline (un paso = un CT completo):
  cubo 256³ (1,5625 mm, ventana HU -> [0, 1]) -> aumento 3D en la GPU
  -> ENCODER FundidoraPC 3D (Conv3d + InstanceNorm + ReLU, MaxPool3d), CBAM 3D + γ en los bloques 3 y 4
     -> cuello dilatado -> DECODER U-Net 3D hasta P3 (1/8) y P2 (1/4)
  -> DETECCIÓN 2D por corte: para el corte s de la vista v se cortan P3 y P2 en esa vista (interpolación lineal
     entre los dos cortes vecinos del mapa), se llevan a la grilla de 32 × 32 y la cabeza con anclas de la vista
     predice cajas (como el modelo 2.5D). NMS propio. La presencia de cada región en el corte (clasificación) es
     el máximo puntaje de sus anclas, como en el 2.5D.
  Pérdida: la del 2.5D, λ_cls · BCE + λ_box · SmoothL1 (`config.LAMBDAS`).

Comandos (desde la raíz del proyecto):
    uv run python -m src.models.detection cache            # caché del cubo (compartido con el 2.5D)
    uv run python -m src.models.model3d bench              # memoria y tiempo de un paso
    uv run python -m src.models.model3d overfit
    uv run python -m src.models.model3d train
    uv run python -m src.models.model3d train --resume --epochs 140 --no-early-stop     # seguir entrenando
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

# Con 6 GB de GPU (RTX 3060 Laptop) la fragmentación del asignador puede hacer que lo reservado pase de 6 GB
# y Windows desborde a memoria compartida (mucho más lento). Debe fijarse antes de importar torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src import config
from src.models import detection as det
from src.models.detection import CLASSES, VIEWS, AnchorHead, GatedCBAM

N = config.INPUT_SIZE                                   # lado del cubo del caché (coordenadas de las cajas)
REGION_LUT = torch.from_numpy(det.VIEW_LUT["axial"]).long()          # código -> 0 fondo, 1..3 región
VIEW_LUT_T = {v: torch.from_numpy(det.VIEW_LUT[v]).long() for v in VIEWS}
BONE_THR = (config.BONE_HU - config.HU_WINDOW[0]) / (config.HU_WINDOW[1] - config.HU_WINDOW[0])


# ======================================================================================
# Datos
# ======================================================================================

class VolumeDataset(torch.utils.data.Dataset):
    """Un CT del caché: (imagen uint8 (Z, Y, X), códigos de fragmento uint8, caso)."""

    def __init__(self, case_ids):
        self.cases = list(case_ids)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        img = np.load(config.CACHE_DIR / f"{cid}_img.npy")
        msk = np.load(config.CACHE_DIR / f"{cid}_msk.npy")
        return torch.from_numpy(img), torch.from_numpy(msk), cid


def case_ids(split: str) -> list:
    s = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})
    ids = s.case_id[s.split == split].tolist()
    missing = [c for c in ids if not (config.CACHE_DIR / f"{c}_img.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} casos en el caché {config.CACHE_DIR}. "
                                "Ejecuta: python -m src.models.detection cache")
    return ids


def _rot(axis: int, a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    r = torch.eye(3, device=a.device)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def augment_3d(x: torch.Tensor, m: torch.Tensor):
    """Aumento 3D en la GPU. x (1, 1, Z, Y, X) float [0, 1], m (Z, Y, X) long.

    Geometría (mismo muestreo para imagen y máscara, así las cajas siguen a la imagen): rotación pequeña en los
    tres ejes, escala, traslación. Intensidad: gamma, contraste y brillo. Degradación (cada una con probabilidad
    `config.AUG_DEGRADE_PROB`): baja resolución, desenfoque gaussiano y ruido. Sin espejo izquierda-derecha
    (cambiaría el lado del coxal sin cambiar su etiqueta).
    """
    dev = x.device
    u = lambda: float(torch.rand(1)) * 2 - 1
    ang = math.radians(config.M3D_AUG_ROTATION_DEG)
    r = _rot(0, torch.tensor(u() * ang, device=dev)) @ _rot(1, torch.tensor(u() * ang, device=dev)) \
        @ _rot(2, torch.tensor(u() * ang, device=dev))
    r = r / (1 + u() * config.AUG_SCALE)
    t = torch.tensor([u() * config.AUG_SHIFT for _ in range(3)], device=dev)
    theta = torch.cat([r, t[:, None]], 1)[None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.grid_sample(m[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    m = m[0, 0].round().long()
    del grid
    gamma = math.exp(u() * math.log(config.AUG_GAMMA))
    x = (x.clamp(0, 1) ** gamma * (1 + u() * config.AUG_CONTRAST) + u() * config.AUG_BRIGHTNESS).clamp(0, 1)

    p = config.AUG_DEGRADE_PROB
    unif = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1))
    size = x.shape[-3:]
    if float(torch.rand(1)) < p:                                    # baja resolución
        f = unif(*config.AUG_LOWRES_FACTOR)
        small = F.interpolate(x, size=[max(int(s / f), 8) for s in size], mode="trilinear", align_corners=False)
        x = F.interpolate(small, size=size, mode="trilinear", align_corners=False)
    if float(torch.rand(1)) < p:                                    # desenfoque gaussiano separable
        sigma = unif(*config.AUG_BLUR_SIGMA)
        rad = int(math.ceil(3 * sigma))
        k = torch.exp(-torch.arange(-rad, rad + 1, device=dev, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
        k = k / k.sum()
        for d in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[2 + d] = -1
            pad = [0] * 6
            pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = rad
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    if float(torch.rand(1)) < p:                                    # ruido gaussiano
        x = x + torch.randn_like(x) * float(torch.rand(1)) * config.AUG_NOISE_STD
    return x.clamp(0, 1), m


def take(vol: torch.Tensor, view: str, idx: torch.Tensor) -> torch.Tensor:
    """Cortes `idx` de un volumen (C, Z, Y, X) en la vista `view` -> (k, C, H, W), con la misma
    orientación que `preprocessing.view_slices` (coronal y sagital con la cabeza arriba)."""
    if view == "axial":
        return vol[:, idx].transpose(0, 1)
    if view == "coronal":
        return vol[:, :, idx].permute(2, 0, 1, 3).flip(2)
    return vol[:, :, :, idx].permute(3, 0, 1, 2).flip(2)


def view_counts(reg: torch.Tensor, view: str) -> torch.Tensor:
    """Píxeles de cada región (1..3) por corte de la vista -> (N, 3)."""
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[view]
    return torch.stack([(reg == c).sum(axes) for c in (1, 2, 3)], 1)


def sample_slices(reg: torch.Tensor, x: torch.Tensor, view: str, n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` cortes de la vista para la cabeza de detección: fracción `config.NEG_FRAC` sin regiones (de
    preferencia con hueso por intensidad: negativos difíciles) y el resto con alguna región (como el 2.5D)."""
    counts = view_counts(reg, view)
    if view == "sagittal":
        counts = torch.stack([counts[:, 0], counts[:, 1] + counts[:, 2]], 1)
    pos = (counts >= config.MIN_REGION_PX).any(1)
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[view]
    bone = (x[0, 0] >= BONE_THR).float().mean(axes) >= config.MIN_BONE_FRACTION
    p_idx = pos.nonzero(as_tuple=True)[0].cpu()
    n_idx = (~pos & bone).nonzero(as_tuple=True)[0].cpu()
    if len(n_idx) == 0:
        n_idx = (~pos).nonzero(as_tuple=True)[0].cpu()
    n_neg = min(int(round(n * config.NEG_FRAC)), len(n_idx))
    n_pos = min(n - n_neg, len(p_idx))
    pick = [p_idx[torch.randperm(len(p_idx), generator=gen)[:n_pos]], n_idx[torch.randperm(len(n_idx), generator=gen)[:n_neg]]]
    return torch.cat(pick).to(reg.device)


# ======================================================================================
# Modelo
# ======================================================================================

def conv_in_relu(in_c: int, out_c: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.InstanceNorm3d(out_c, affine=True), nn.ReLU(inplace=True))


class CBAM3d(nn.Module):
    """CBAM (Woo et al., 2018) en 3D: atención de canal (avg/max-pool global -> MLP) y espacial
    (media y máximo entre canales -> Conv3d 7×7×7)."""

    def __init__(self, channels: int, reduction: int = config.CBAM_REDUCTION, kernel: int = config.CBAM_KERNEL):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv3d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv3d(hidden, channels, 1))
        self.spatial = nn.Conv3d(2, 1, kernel, padding=kernel // 2)
        self.spatial_map = None

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(F.adaptive_avg_pool3d(x, 1)) + self.mlp(F.adaptive_max_pool3d(x, 1)))
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        self.spatial_map = sa.detach()
        return x * sa


class GatedCBAM3d(nn.Module):
    """y = x + γ · CBAM3d(x), con γ aprendible que empieza en `config.GAMMA_INIT` (0)."""

    def __init__(self, channels: int):
        super().__init__()
        self.cbam = CBAM3d(channels)
        self.gamma = nn.Parameter(torch.tensor(float(config.GAMMA_INIT)))

    def forward(self, x):
        return x + self.gamma * self.cbam(x)


class Up3d(nn.Module):
    """Conv 1×1×1 (reduce canales) + subida trilineal ×2 -> concat skip -> 2 × (Conv3d 3×3×3 + IN + ReLU).
    La conv 1×1×1 va antes de la subida (son operaciones lineales que conmutan) para ahorrar memoria."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.reduce = nn.Conv3d(in_c, out_c, 1)
        self.conv = nn.Sequential(conv_in_relu(out_c + skip_c, out_c), conv_in_relu(out_c, out_c))

    def forward(self, x, skip):
        x = F.interpolate(self.reduce(x), size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class Fundidora3DDetector(nn.Module):
    """Backbone 3D (encoder FundidoraPC 3D + CBAM 3D con γ + cuello + decoder hasta P2) y una cabeza con
    anclas 2D por vista sobre los cortes de P3 y P2."""

    def __init__(self, cbam: bool = True):
        super().__init__()
        w = config.M3D_WIDTHS
        d3, d2 = config.M3D_DECODER
        self.blocks, c = nn.ModuleList(), 1
        for wi in w:
            self.blocks.append(nn.Sequential(conv_in_relu(c, wi), nn.MaxPool3d(2)))
            c = wi
        self.attn = nn.ModuleList(GatedCBAM3d(wi) if cbam and i + 1 in config.M3D_CBAM_BLOCKS else nn.Identity()
                                  for i, wi in enumerate(w))
        dl1, dl2 = config.BOTTLENECK_DILATIONS
        self.neck = nn.Sequential(conv_in_relu(w[3], config.M3D_NECK, dl1), conv_in_relu(config.M3D_NECK, config.M3D_NECK, dl2))
        self.up3 = Up3d(config.M3D_NECK, w[2], d3)       # 1/16 -> 1/8   (skip: bloque 3)
        self.up2 = Up3d(d3, w[1], d2)                    # 1/8  -> 1/4   (skip: bloque 2)
        self.grid = N // config.STRIDE
        self.heads = nn.ModuleDict()
        for v in VIEWS:
            a = config.ANCHORS[v]
            self.heads[v] = AnchorHead(d3 + d2, len(CLASSES[v]), len(a["scales"]) * len(a["ratios"]),
                                       attention=cbam and config.HEAD_ATTENTION)
            anclas = det.generar_anclas_grilla(self.grid, config.STRIDE, a["scales"], a["ratios"])
            self.register_buffer(f"anchors_{v}", torch.tensor(anclas, dtype=torch.float32), persistent=False)
        self.use_checkpoint = config.M3D_CHECKPOINT

    def _ck(self, fn, *xs):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *xs, use_reentrant=False)
        return fn(*xs)

    def _encode_block(self, i):
        return lambda x: self.attn[i](self.blocks[i](x))

    def forward(self, x):
        """x (1, 1, 256, 256, 256); se reduce a `config.M3D_INPUT`³ si es menor. Devuelve los mapas 3D."""
        if x.shape[-1] != config.M3D_INPUT:
            x = F.interpolate(x, size=(config.M3D_INPUT,) * 3, mode="trilinear", align_corners=False)
        f1 = self._ck(self._encode_block(0), x)
        f2 = self._ck(self._encode_block(1), f1)
        f3 = self.attn[2](self.blocks[2](f2))
        f4 = self.attn[3](self.blocks[3](f3))
        neck = self.neck(f4)
        p3 = self.up3(neck, f3)
        p2 = self._ck(self.up2, p3, f2)
        return {"p3": p3, "p2": p2, "neck": neck}

    def slice_features(self, feats: dict, view: str, slices: torch.Tensor) -> torch.Tensor:
        """Características 2D de los cortes `slices` (coordenadas del cubo de 256) en la vista `view`:
        P3 y P2 cortados con interpolación lineal entre los dos cortes vecinos del mapa, llevados a la
        grilla de 32 × 32 y concatenados -> (k, C3 + C2, 32, 32)."""
        outs = []
        for name in ("p3", "p2"):
            f = feats[name][0]
            n = f.shape[-1]
            p = ((slices.float() + 0.5) * n / N - 0.5).clamp(0, n - 1)
            i0 = p.floor().long()
            i1 = (i0 + 1).clamp(max=n - 1)
            w = (p - i0.float()).view(-1, 1, 1, 1).to(f.dtype)
            s = take(f, view, i0) * (1 - w) + take(f, view, i1) * w
            if s.shape[-1] > self.grid:
                s = F.adaptive_avg_pool2d(s, self.grid)
            elif s.shape[-1] < self.grid:
                s = F.interpolate(s, size=(self.grid,) * 2, mode="bilinear", align_corners=False)
            outs.append(s)
        return torch.cat(outs, 1)

    def detect(self, feats: dict, view: str, slices: torch.Tensor) -> dict:
        """Cabeza de detección de la vista sobre los cortes pedidos (mismo formato que ThreeViewDetector)."""
        cls, reg = self.heads[view](self.slice_features(feats, view, slices))
        return {"cls": cls, "reg": reg, "anchors": getattr(self, f"anchors_{view}"), "view": view}

    def gammas(self) -> dict:
        out = {f"enc{i}": float(m.gamma.detach()) for i, m in enumerate(self.attn, start=1) if isinstance(m, GatedCBAM3d)}
        out |= {f"head_{v}": float(h.attn.gamma.detach()) for v, h in self.heads.items() if isinstance(h.attn, GatedCBAM)}
        return out


# ======================================================================================
# Pérdida
# ======================================================================================

def step_loss(model, x, m, gen, n_slices=config.M3D_DET_SLICES):
    """Pérdida de detección de un CT: media de las tres vistas de λ_cls · BCE + λ_box · SmoothL1 (como el 2.5D)
    sobre `n_slices` cortes por vista. Devuelve (total, términos por vista)."""
    with det.autocast(x.device):
        feats = model(x)
        reg = REGION_LUT.to(m.device)[m]
        losses, terms = [], {}
        for v in VIEWS:
            sl = sample_slices(reg, x, v, n_slices, gen)
            out = model.detect(feats, v, sl)
            mv = VIEW_LUT_T[v].to(m.device)[take(m[None], v, sl)[:, 0]]
            gt, valid = det.region_boxes(mv, len(CLASSES[v]))
            loss, t = det.detection_loss(out, gt, valid, config.LAMBDAS)
            losses.append(loss)
            terms |= {f"{k}_{v}": val for k, val in t.items()}
    terms["cls"] = float(np.mean([terms[f"cls_{v}"] for v in VIEWS]))
    terms["box"] = float(np.mean([terms[f"box_{v}"] for v in VIEWS]))
    return torch.stack(losses).mean(), terms


def to_gpu(img_u8, msk_u8, device, augment=False):
    x = img_u8.to(device, non_blocking=True).float().div_(255)[None]       # (1, 1, Z, Y, X) del batch de 1
    m = msk_u8.to(device, non_blocking=True)[0].long()
    if augment:
        x, m = augment_3d(x, m)
    return x, m


# ======================================================================================
# Validación
# ======================================================================================

@torch.no_grad()
def validate(model, cases, device, stride: int = config.M3D_VAL_STRIDE) -> dict:
    """Las mismas métricas por corte del modelo 2.5D (mAP50, mAP50-95, mIoU, F1 y AUC por vista y su media)
    sobre 1 de cada `stride` cortes de los casos."""
    model.eval()
    acc = {v: {"preds": [], "gt": [], "valid": [], "prob": [], "loss": []} for v in VIEWS}
    ds = VolumeDataset(cases)
    for i in range(len(ds)):
        img, msk, _ = ds[i]
        x, m = to_gpu(img[None], msk[None], device)
        with det.autocast(device):
            feats = model(x)
        sl = torch.arange(0, N, stride, device=device)
        for v in VIEWS:
            mv = VIEW_LUT_T[v].to(device)[take(m[None], v, sl)[:, 0]]
            gt, valid = det.region_boxes(mv, len(CLASSES[v]))
            for c0 in range(0, len(sl), 64):
                s = slice(c0, c0 + 64)
                with det.autocast(device):
                    out = model.detect(feats, v, sl[s])
                _, t = det.detection_loss(out, gt[s], valid[s])
                a = acc[v]
                a["loss"].append(t)
                a["preds"] += det.postprocess(out)
                a["gt"].append(gt[s].cpu()); a["valid"].append(valid[s].cpu())
                a["prob"].append(torch.sigmoid(out["cls"].float()).amax(1).cpu())
        del feats
    ev = {}
    for v, a in acc.items():
        gt, valid, prob = torch.cat(a["gt"]).numpy(), torch.cat(a["valid"]).numpy(), torch.cat(a["prob"]).numpy()
        ev[v] = {"loss": {k: float(np.mean([t[k] for t in a["loss"]])) for k in a["loss"][0]},
                 "det": det.detection_metrics(a["preds"], gt, valid, CLASSES[v]),
                 "cls": det.classification_metrics(valid, prob, CLASSES[v])}
    return det._flat_metrics(ev, "val")


# ======================================================================================
# Entrenamiento
# ======================================================================================

def run_config(cbam: bool, epochs: int) -> dict:
    keys = [k for k in dir(config) if k.startswith("M3D_")]
    return {"arch": "fundidora3d_det2d", "cbam": cbam, "head_attention": cbam and config.HEAD_ATTENTION,
            **{k.lower(): getattr(config, k) for k in keys}, "epochs": epochs,
            "input_cube": N, "cube_mm": config.CUBE_MM, "hu_window": list(config.HU_WINDOW),
            "anchors": {v: {k: list(x) for k, x in a.items()} for v, a in config.ANCHORS.items()},
            "stride": config.STRIDE, "view_classes": {v: list(c) for v, c in CLASSES.items()},
            "pos_iou": config.POS_IOU, "neg_iou": config.NEG_IOU, "lambdas": dict(config.LAMBDAS),
            "neg_frac": config.NEG_FRAC, "grad_clip": config.GRAD_CLIP, "amp": config.AMP, "seed": config.SEED,
            "augment": {k.removeprefix("AUG_").lower(): getattr(config, k) for k in dir(config) if k.startswith("AUG_")}}


def _optimizer(model):
    """AdamW con weight decay solo en los pesos de las convoluciones (no en γ, normas ni sesgos)."""
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.M3D_WEIGHT_DECAY},
                              {"params": no_decay, "weight_decay": 0.0}], lr=config.M3D_LR)


def train(args):
    det.seed_everything()
    device = det.get_device(args.device)
    torch.backends.cudnn.benchmark = False   # las formas cambian (aumento): re-afinar costaba ~10 s
    cbam = not args.no_cbam
    name = args.name or (config.M3D_RUN_CBAM if cbam else config.M3D_RUN_NO_CBAM)
    out_dir = config.DETECTION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_config(cbam, args.epochs)

    tr_ids, va_ids = case_ids("train"), case_ids("val")
    loader = torch.utils.data.DataLoader(VolumeDataset(tr_ids), batch_size=1, shuffle=True, num_workers=args.workers,
                                         persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
                                         generator=torch.Generator().manual_seed(config.SEED))
    model = Fundidora3DDetector(cbam).to(device)
    opt = _optimizer(model)
    steps = len(loader)
    # Coseno sobre `args.epochs`. Al reanudar con más épocas (--resume --epochs 140) el coseno se recalcula sobre
    # el total nuevo: el lr vuelve a subir desde ~0 hasta el punto del coseno largo (como un reinicio, SGDR).
    total, warm = args.epochs * steps, config.M3D_WARMUP_EPOCHS * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    gen = torch.Generator().manual_seed(config.SEED)

    start, best, history, n_val_no_improve = 0, -1.0, [], 0
    last_path = out_dir / "last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        n_val_no_improve = ck.get("n_val_no_improve", 0)
        torch.set_rng_state(ck["rng"])
        print(f"Reanudando {name} desde la época {start + 1} hasta la {args.epochs}", flush=True)
    else:
        history.append({"epoch": -1, **{f"gamma_{k}": v for k, v in model.gammas().items()}})   # γ antes de entrenar
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"{name}: {det.count_params(model):,} parámetros | {device} | {steps} CT por época | val {len(va_ids)} CT",
          flush=True)

    for epoch in range(start, args.epochs):
        model.train()
        t0, acc, skipped = time.time(), [], 0
        for img, msk, _ in loader:
            x, m = to_gpu(img, msk, device, augment=True)
            loss, terms = step_loss(model, x, m, gen)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1; sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            if not torch.isfinite(gnorm) and not scaler.is_enabled():
                skipped += 1; opt.zero_grad(set_to_none=True); sched.step()
                continue
            scaler.step(opt); scaler.update(); sched.step()
            acc.append({"loss": float(loss.detach()), **terms})
        acc = pd.DataFrame(acc)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_time_s": time.time() - t0, "skipped_steps": skipped,
               **{f"train_{k}": float(acc[k].mean()) for k in acc.columns},
               **{f"gamma_{k}": v for k, v in model.gammas().items()}}
        if device.type == "cuda":
            row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        is_val = (epoch + 1) % config.M3D_VAL_EVERY == 0 or epoch + 1 == args.epochs
        msg = (f"[{name}] época {epoch + 1}/{args.epochs}  loss {row['train_loss']:.3f} (cls {row['train_cls']:.3f} "
               f"box {row['train_box']:.3f})  {row['train_time_s']:.0f} s")
        if is_val:
            t1 = time.time()
            row |= validate(model, va_ids, device)
            row["val_time_s"] = time.time() - t1
            msg += (f"  | val mAP50 {row['val_mAP50']:.3f} mAP50-95 {row['val_mAP50_95']:.3f} mIoU {row['val_mIoU']:.3f} "
                    f"F1 {row['val_macro_f1']:.3f}")
        row["time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if is_val:
            if row["val_mAP50_95"] > best:
                best, n_val_no_improve = row["val_mAP50_95"], 0
                torch.save({**state, "val_mAP50_95": best}, out_dir / "best.pth")
                msg += "  * mejor"
            else:
                n_val_no_improve += 1
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "history": history, "rng": torch.get_rng_state(),
                    "n_val_no_improve": n_val_no_improve}, last_path)
        gam = "  ".join(f"γ {k} {v:+.3f}" for k, v in model.gammas().items())
        print(msg + f"  {gam}" + (f"  pasos saltados {skipped}" if skipped else ""), flush=True)
        if (not args.no_early_stop and is_val and epoch + 1 >= config.M3D_EARLY_STOP_START
                and n_val_no_improve >= config.M3D_EARLY_STOP_PATIENCE):
            print(f"[{name}] parada temprana: {n_val_no_improve} validaciones sin mejorar val mAP50-95 (mejor {best:.3f})",
                  flush=True)
            break
    return out_dir


def overfit(args):
    """Prueba de correctitud: sobreajustar `config.M3D_OVERFIT_CASES` CT sin aumento. La detección de esos CT
    debe quedar casi perfecta."""
    det.seed_everything()
    device = det.get_device(args.device)
    cbam = not args.no_cbam
    ids = case_ids("train")[:config.M3D_OVERFIT_CASES]
    ds = VolumeDataset(ids)
    data = [ds[i] for i in range(len(ds))]
    model = Fundidora3DDetector(cbam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.OVERFIT_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    gen = torch.Generator().manual_seed(config.SEED)
    t0 = time.time()
    for it in range(args.iters):
        model.train()
        img, msk, _ = data[it % len(data)]
        x, m = to_gpu(img[None], msk[None], device)
        loss, terms = step_loss(model, x, m, gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if it % 25 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {float(loss):.4f}  cls {terms['cls']:.4f}  box {terms['box']:.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    row = validate(model, ids, device, stride=2)
    out_dir = config.DETECTION_DIR / f"overfit_{config.M3D_RUN_CBAM if cbam else config.M3D_RUN_NO_CBAM}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    print("Overfit: " + "  ".join(f"{k} {row[k]:.3f}" for k in ("val_mAP50", "val_mAP50_95", "val_mIoU", "val_macro_f1")))


def bench(args):
    """Memoria máxima y tiempo de un paso de entrenamiento (ida y vuelta) con un volumen aleatorio."""
    device = det.get_device(args.device)
    torch.backends.cudnn.benchmark = False
    model = Fundidora3DDetector(not args.no_cbam).to(device)
    opt = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    gen = torch.Generator().manual_seed(0)
    img = (torch.rand(1, N, N, N) * 255).to(torch.uint8)
    msk = torch.zeros(1, N, N, N, dtype=torch.uint8)
    msk[:, 80:180, 60:190, 40:120] = 11; msk[:, 80:180, 60:190, 130:210] = 21; msk[:, 100:170, 120:180, 110:140] = 1
    print(f"{det.count_params(model):,} parámetros, entrada {config.M3D_INPUT}³, checkpointing {config.M3D_CHECKPOINT}",
          flush=True)
    for i in range(args.iters):
        torch.cuda.synchronize(); t0 = time.time()
        x, m = to_gpu(img, msk, device, augment=True)
        loss, _ = step_loss(model, x, m, gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        print(f"paso {i}: {time.time() - t0:.2f} s  memoria máx {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB "
              f"(reservada {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Fundidora 3D con detección 2D por corte.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default="auto")
        sp.add_argument("--no-cbam", action="store_true")

    t = sub.add_parser("train"); common(t)
    t.add_argument("--epochs", type=int, default=config.M3D_EPOCHS)
    t.add_argument("--workers", type=int, default=config.M3D_WORKERS)
    t.add_argument("--name", default=None)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--no-early-stop", action="store_true",
                   help="sin parada temprana (p. ej. al reanudar con más épocas: el lr vuelve a subir y val baja al inicio)")
    o = sub.add_parser("overfit"); common(o)
    o.add_argument("--iters", type=int, default=config.M3D_OVERFIT_ITERS)
    b = sub.add_parser("bench"); common(b)
    b.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    {"train": train, "overfit": overfit, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
