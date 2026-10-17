"""PelvicU-3D: modelo 3D multitarea para fracturas pélvicas PENGWIN.

Pipeline:
  cubo CT 256³ (dos canales: ventana de hueso y ventana trabecular)
  -> encoder U-Net 3D con atención (CBAM o squeeze-excitation, con γ) en el bloque 4
  -> cuello dilatado
  -> decoder U-Net 3D hasta P0, con atención en P3

Cabezas:
  - detección class-agnostic 2D por corte axial sobre P3 + P2, con anclas y NMS propio
  - segmentación semántica sobre P0: fondo, sacro, coxal izquierdo y coxal derecho
  - máscara por caja (ROI Align de P2, P1 y del CT a doble resolución) con su rama de clase

Inferencia (`infer_ct`):
  detección axial -> NMS -> regiones 3D -> segmentación restringida a las regiones -> máscara y clase de cada caja
  -> fusión de la segmentación y la máscara (`fuse_heads`) -> vuelta a la rejilla original del CT.
"""

import argparse
import json
import math
import os
import pathlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Con una GPU de 6 GB la fragmentación del asignador puede hacer que lo reservado pase de 6 GB
# y Windows desborde a memoria compartida (mucho más lento). Debe fijarse antes de importar torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # una operación del CBAM 3D no existe en la GPU del Mac
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import roi_align

from src import config
from src.models import detection as det
from src.models.detection import CLASSES, VIEWS, AnchorHead
from src.models.segmentation import (
    SegmentationHead3D,
    _binary_scores,
    bone_instances,
    bone_targets,
    boxes_to_region_mask_3d,
    head_metrics,
    segmentation_metrics,
    segmentation_prediction,
    separation_distances,
)
from src.models.classification import (
    ClassificationHead3D,
    classification_loss,
    classification_metrics,
    extract_gt_instance_features,
    extract_pred_instance_features,
)

N = config.INPUT_SIZE                                   # lado del cubo del caché (coordenadas de las cajas)
REGION_LUT = torch.from_numpy(det.VIEW_LUT["axial"]).long()          # código -> 0 fondo, 1..3 región
VIEW_LUT_T = {v: torch.from_numpy(det.VIEW_LUT[v]).long() for v in VIEWS}
BONE_THR = (config.BONE_HU - config.HU_WINDOW[0]) / (config.HU_WINDOW[1] - config.HU_WINDOW[0])


# ======================================================================================
# Datos
# ======================================================================================

class VolumeDataset(torch.utils.data.Dataset):
    """Un CT del caché: (imagen uint8 (Z, Y, X), códigos de fragmento uint8, caso). Con `hires=True` agrega la imagen
    y los códigos a doble resolución en el plano (Z, 2Y, 2X) para la cabeza de máscara:
    (img, msk, img_hr, msk_hr, caso)."""

    def __init__(self, case_ids, hires: bool = False):
        self.cases = list(case_ids)
        self.hires = hires

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        img = np.load(config.CACHE_DIR / f"{cid}_img.npy")
        msk = np.load(config.CACHE_DIR / f"{cid}_msk.npy")
        if not self.hires:
            return torch.from_numpy(img), torch.from_numpy(msk), cid
        img_hr, msk_hr = load_hires(cid)
        return torch.from_numpy(img), torch.from_numpy(msk), img_hr, msk_hr, cid


def load_hires(cid: str):
    """Imagen (uint8) y códigos de fragmento del caché a doble resolución en el plano: tensores (Z, 2Y, 2X)."""
    d = config.M3D_HIRES_DIR
    if not (d / f"{cid}_img.npy").exists():
        raise FileNotFoundError(f"Falta {cid} en {d}. Ejecuta: python -m src.models.model3d cache-hires")
    return torch.from_numpy(np.load(d / f"{cid}_img.npy")), torch.from_numpy(np.load(d / f"{cid}_msk.npy"))


def build_hires_cache(overwrite: bool = False) -> None:
    """Caché de la cabeza de máscara: cada CT y su máscara en la rejilla del cubo con el doble de resolución en
    el plano axial (512 × 512 a 0,78 mm, los mismos 256 cortes), `preprocessing.hires_geometry`. Mismo ventaneo."""
    from tqdm.auto import tqdm
    from src.dataset import preprocessing as pre
    out = config.M3D_HIRES_DIR
    out.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})
    img_paths = {q.stem: q for d in config.IMAGE_DIRS for q in d.glob("*.mha")}
    for cid in tqdm(splits.case_id.tolist(), desc="Caché a doble resolución"):
        fi, fm = out / f"{cid}_img.npy", out / f"{cid}_msk.npy"
        if fi.exists() and fm.exists() and not overwrite:
            continue
        img = pre.reorient(pre.load_volume(img_paths[cid]))
        geom = pre.hires_geometry(pre.cube_geometry(img), config.M3D_HIRES_FACTOR)
        hu = pre.sitk.GetArrayFromImage(pre.to_cube(img, geom))
        np.save(fi, np.round(pre.window_hu(hu) * 255).astype(np.uint8))
        lab = pre.reorient(pre.load_volume(config.LABEL_DIR / f"{cid}.mha"))
        np.save(fm, pre.sitk.GetArrayFromImage(pre.to_cube(lab, geom, is_mask=True)).astype(np.uint8))


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


def augment_3d(x: torch.Tensor, m: torch.Tensor, params: dict | None = None):
    """Aumento 3D en la GPU. x (1, 1, Z, Y, X) float [0, 1], m (Z, Y, X) long.

    Geometría (mismo muestreo para imagen y máscara, así las cajas siguen a la imagen): rotación pequeña en los
    tres ejes, escala, traslación. Intensidad: gamma, contraste y brillo. Degradación (cada una con probabilidad
    `config.AUG_DEGRADE_PROB`): baja resolución, desenfoque gaussiano y ruido. Sin espejo izquierda-derecha
    (cambiaría el lado del coxal sin cambiar su etiqueta).
    Si se pasa `params`, guarda ahí la transformación (theta, gamma, contraste y brillo) para aplicarla igual a los
    cortes a doble resolución de la cabeza de máscara (`hires_slices`).
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
    contrast, bright = u() * config.AUG_CONTRAST, u() * config.AUG_BRIGHTNESS
    x = (x.clamp(0, 1) ** gamma * (1 + contrast) + bright).clamp(0, 1)
    if params is not None:
        params.update(theta=theta, gamma=gamma, contrast=contrast, bright=bright)

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
    preferencia con hueso por intensidad: negativos difíciles) y el resto con alguna región."""
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

def conv_in_relu(in_c: int, out_c: int, dilation: int = 1, leaky: bool = False) -> nn.Sequential:
    act = nn.LeakyReLU(config.M3D_LEAKY_SLOPE, inplace=True) if leaky else nn.ReLU(inplace=True)
    return nn.Sequential(nn.Conv3d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.InstanceNorm3d(out_c, affine=True), act)


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


class SE3d(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018) en 3D: solo atención de canal (avg-pool global -> MLP -> sigmoide)."""

    def __init__(self, channels: int, reduction: int = config.CBAM_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv3d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv3d(hidden, channels, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(F.adaptive_avg_pool3d(x, 1)))


class GatedSE3d(nn.Module):
    """y = x + γ · SE3d(x), con γ aprendible desde `config.GAMMA_INIT` (igual que GatedCBAM3d)."""

    def __init__(self, channels: int):
        super().__init__()
        self.se = SE3d(channels)
        self.gamma = nn.Parameter(torch.tensor(float(config.GAMMA_INIT)))

    def forward(self, x):
        return x + self.gamma * self.se(x)


ATTENTION_3D = {"cbam": GatedCBAM3d, "se": GatedSE3d}


def current_arch() -> dict:
    """Arquitectura que fija `config` hoy (la guarda `run_config` en cada entrenamiento)."""
    return {"cbam_blocks": tuple(config.M3D_CBAM_BLOCKS), "decoder_attention": tuple(config.M3D_DECODER_ATTENTION),
            "trab_window": config.M3D_TRAB_WINDOW, "mask_head": config.M3D_MASK_HEAD,
            "decoder_act": config.M3D_DECODER_ACT, "seg_aux": config.M3D_SEG_AUX}


def arch_from_config(cfg: dict) -> tuple[str, dict]:
    """(atención, arquitectura) de un checkpoint. Los checkpoints que no guardan estas claves usan:
    CBAM en los bloques 3 y 4, sin atención en el decoder, un solo canal de entrada, sin cabeza de máscara, ReLU en el
    decoder y sin salida semántica auxiliar."""
    attention = cfg.get("attention") or ("cbam" if cfg.get("cbam", True) else "none")
    trab = cfg.get("m3d_trab_window")
    return attention, {"cbam_blocks": tuple(cfg.get("m3d_cbam_blocks", (3, 4))),
                       "decoder_attention": tuple(cfg.get("m3d_decoder_attention", ())),
                       "trab_window": tuple(trab) if trab else None, "mask_head": bool(cfg.get("m3d_mask_head", False)),
                       "decoder_act": cfg.get("m3d_decoder_act", "relu"), "seg_aux": bool(cfg.get("m3d_seg_aux", False))}


def load_checkpoint(path) -> dict:
    """`torch.load` de un checkpoint; los entrenados en Windows pueden guardar rutas `WindowsPath` en
    la configuración, que no se pueden abrir en macOS/Linux: se leen como rutas puras."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (pathlib.UnsupportedOperation, NotImplementedError):
        orig = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PureWindowsPath
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            pathlib.WindowsPath = orig


def load_model(ck: dict, device) -> "Fundidora3DDetector":
    """Modelo de un checkpoint (`best.pth`), con su arquitectura. Ignora la cabeza de instancia (`inst_head`) si el
    checkpoint la trae."""
    attention, arch = arch_from_config(ck["config"])
    model = Fundidora3DDetector(attention, arch).to(device).eval()
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("inst_head.")})
    return model


class Up3d(nn.Module):
    """Conv 1×1×1 (reduce canales) + subida trilineal ×2 -> concat skip -> 2 × (Conv3d 3×3×3 + IN + ReLU).
    La conv 1×1×1 va antes de la subida (son operaciones lineales que conmutan) para ahorrar memoria."""

    def __init__(self, in_c: int, skip_c: int, out_c: int, leaky: bool = False):
        super().__init__()
        self.reduce = nn.Conv3d(in_c, out_c, 1)
        self.conv = nn.Sequential(conv_in_relu(out_c + skip_c, out_c, leaky=leaky), conv_in_relu(out_c, out_c, leaky=leaky))

    def forward(self, x, skip):
        x = F.interpolate(self.reduce(x), size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))

class RoiMaskHead(nn.Module):
    """Cabeza tipo Mask R-CNN sobre cada caja axial.

    Entra el ROI Align de P2 y P1 del corte (`config.M3D_ROI_FEAT`²) y el del CT a doble resolución
    (`config.M3D_MASK_SIZE`²). Rama de máscara: 2 conv 3×3 -> deconvolución ×2 -> concat con el CT -> 2 conv 3×3 ->
    una máscara por clase (sacro, coxal izq., coxal der.), como Mask R-CNN. Devuelve también el vector del ROI
    (promedio de las características) para la rama de clase (la cabeza de clasificación)."""

    def __init__(self, in_feat: int, in_img: int, mid: int = config.M3D_MASK_CHANNELS, classes: int = 3):
        super().__init__()
        relu = lambda: nn.ReLU(inplace=True)
        self.feat = nn.Sequential(nn.Conv2d(in_feat, mid, 3, padding=1), relu(), nn.Conv2d(mid, mid, 3, padding=1), relu())
        self.up = nn.Sequential(nn.ConvTranspose2d(mid, mid, 2, stride=2), relu())
        self.fuse = nn.Sequential(nn.Conv2d(mid + in_img, mid // 2, 3, padding=1), relu(),
                                  nn.Conv2d(mid // 2, mid // 2, 3, padding=1), relu())
        self.mask = nn.Conv2d(mid // 2, classes, 1)

    def forward(self, f: torch.Tensor, img: torch.Tensor):
        h = self.feat(f)
        m = self.fuse(torch.cat([self.up(h), img], 1))
        return self.mask(m), h.mean((2, 3))


def mask_loss(logits: torch.Tensor, cls: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Máscara de la clase real de cada ROI (como Mask R-CNN): BCE + Dice suave, promedio sobre los ROI."""
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    lg = logits[torch.arange(len(cls), device=logits.device), cls].float()
    bce = F.binary_cross_entropy_with_logits(lg, target)
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * target).sum((1, 2)) + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
    return bce + dice.mean()


def jitter_boxes(boxes: torch.Tensor, frac: float = config.M3D_MASK_JITTER) -> torch.Tensor:
    """Mueve cada borde de la caja hasta ±`frac` de su lado (simula una caja predicha imperfecta)."""
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
    d = (torch.rand(len(boxes), 4, device=boxes.device) * 2 - 1) * frac * torch.stack([w, h, w, h], 1)
    b = (boxes + d).clamp(0, N)
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1)
    return b


class Fundidora3DDetector(nn.Module):
    """Backbone 3D (encoder FundidoraPC 3D + CBAM 3D con γ + cuello + decoder hasta P2) y una cabeza con
    anclas 2D por vista sobre los cortes de P3 y P2."""

    def __init__(self, attention="cbam", arch: dict | None = None):
        """attention: "cbam" (o True), "se" (solo canal) o "none" (o False). arch: `current_arch()` por defecto;
        para un checkpoint, `arch_from_config(ck["config"])`."""
        super().__init__()
        w = config.M3D_WIDTHS
        d3, d2, d1, d0 = config.M3D_DECODER
        self.attention = {True: "cbam", False: "none"}.get(attention, attention)
        self.arch = arch = arch or current_arch()
        gated = ATTENTION_3D.get(self.attention)
        # Segundo canal de entrada, la ventana trabecular, calculada dentro de la red a partir del primero
        self.trab_window = arch["trab_window"]
        self.in_channels = 2 if self.trab_window else 1

        # Encoder 3D
        self.blocks = nn.ModuleList()
        c = self.in_channels

        for wi in w:
            self.blocks.append(nn.Sequential(conv_in_relu(c, wi), nn.MaxPool3d(2)))
            c = wi

        # Atención (CBAM o SE, con γ) en los bloques configurados (`config.M3D_CBAM_BLOCKS`)
        self.attn = nn.ModuleList(
            gated(wi) if gated and i + 1 in arch["cbam_blocks"] else nn.Identity()
            for i, wi in enumerate(w)
        )

        # Cuello de botella
        dl1, dl2 = config.BOTTLENECK_DILATIONS
        self.neck = nn.Sequential(
            conv_in_relu(w[3], config.M3D_NECK, dl1),
            conv_in_relu(config.M3D_NECK, config.M3D_NECK, dl2),
        )

        # Decoder U-Net 3D
        lk = arch.get("decoder_act", "relu") == "leaky"
        self.up3 = Up3d(config.M3D_NECK, w[2], d3, lk)
        self.up2 = Up3d(d3, w[1], d2, lk)
        self.up1 = Up3d(d2, w[0], d1, lk)
        self.up0 = Up3d(d1, self.in_channels, d0, lk)  # 1/2 -> 1/1   (skip: la imagen de entrada)
        # Atención (con γ) a la salida de las primeras capas del decoder; alimenta segmentación y detección
        self.dec_attn = nn.ModuleDict({lvl: gated({"p3": d3, "p2": d2}[lvl])
                                       for lvl in arch["decoder_attention"]} if gated else {})

        # Segmentación: una cabeza, tres máscaras (sacro, coxal izq., coxal der.) + fondo; cada hueso es una
        # instancia con todos sus fragmentos.
        self.sem_head = SegmentationHead3D(in_channels=d0, num_classes=config.M3D_SEM_CLASSES)
        # Supervisión profunda: salida semántica auxiliar sobre P1 (solo para la pérdida de entrenamiento)
        self.aux_head = SegmentationHead3D(d1, config.M3D_SEM_CLASSES) if arch.get("seg_aux", False) else None

        # Clasificación anatómica de cada instancia segmentada.
        # Con cabeza de máscara, la clasificación es la rama de clase de cada caja (ROI), como en Mask R-CNN: recibe
        # el vector del ROI. Sin ella, el promedio de P1 en cada instancia.
        self.mask_head = arch.get("mask_head", False)
        self.class_head = ClassificationHead3D(
            in_channels=config.M3D_MASK_CHANNELS if self.mask_head else d1,
            hidden_channels=config.M3D_CLS_HIDDEN,
            num_classes=config.M3D_CLS_CLASSES,
        )
        if self.mask_head:
            self.roi_head = RoiMaskHead(d2 + d1, self.in_channels)

        # Configuración de la cabeza de detección existente
        self.grid = N // config.STRIDE
        self.heads = nn.ModuleDict()

        for v in config.M3D_DET_VIEWS:
            a = config.ANCHORS[v]
            self.heads[v] = AnchorHead(
                d3 + d2,
                config.M3D_DET_CLASSES,
                len(a["scales"]) * len(a["ratios"]),
                attention=self.attention if config.HEAD_ATTENTION else "none",
            )

            anclas = det.generar_anclas_grilla(self.grid, config.STRIDE, a["scales"], a["ratios"])
            self.register_buffer(
                f"anchors_{v}",
                torch.tensor(anclas, dtype=torch.float32),
                persistent=False,
            )

        self.use_checkpoint = config.M3D_CHECKPOINT

    def _ck(self, fn, *xs):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *xs, use_reentrant=False)
        return fn(*xs)

    def _up0_input(self, p1, x):
        u = F.interpolate(self.up0.reduce(p1), size=x.shape[-3:], mode="trilinear", align_corners=False)
        return torch.cat([u, x], 1)

    def _encode_block(self, i):
        return lambda x: self.attn[i](self.blocks[i](x))

    def _inputs(self, x):
        """Canal 0: la ventana de `config.HU_WINDOW` en [0, 1] (el caché). Si `trab_window`, agrega la ventana
        trabecular calculada desde el canal 0 (la ventana trabecular cabe dentro de la de hueso)."""
        return input_channels(x, self.trab_window)

    def forward(self, x):
        if x.shape[-1] != config.M3D_INPUT:
            x = F.interpolate(
                x,
                size=(config.M3D_INPUT,) * 3,
                mode="trilinear",
                align_corners=False,
            )
        x = self._inputs(x)

        # Encoder
        f1 = self._ck(
            self._encode_block(0),
            x,
        )
        f2 = self._ck(
            self._encode_block(1),
            f1,
        )
        f3 = self.attn[2](
            self.blocks[2](f2)
        )
        f4 = self.attn[3](
            self.blocks[3](f3)
        )

        # Bottleneck
        neck = self.neck(f4)

        # Decoder
        p3 = self.up3(
            neck,
            f3,
        )
        if "p3" in self.dec_attn:
            p3 = self.dec_attn["p3"](p3)
        p2 = self._ck(
            self.up2,
            p3,
            f2,
        )
        if "p2" in self.dec_attn:
            p2 = self.dec_attn["p2"](p2)
        p1 = self._ck(
            self.up1,
            p2,
            f1,
        )
        # P0 (256³) es el nivel más pesado: checkpoint por partes (subida + concat, y cada conv) para que el
        # recálculo no tenga todo el bloque en memoria a la vez (GPU de 6 GB).
        u = self._ck(self._up0_input, p1, x)
        u = self._ck(self.up0.conv[0], u)
        p0 = self._ck(self.up0.conv[1], u)

        # Segmentación: un hueso por vóxel
        sem_logits = self.sem_head(p0)
        aux = {"sem_logits_aux": self.aux_head(p1)} if self.aux_head is not None else {}

        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "neck": neck,
            "p3": p3,
            "p2": p2,
            "p1": p1,
            "p0": p0,
            "sem_logits": sem_logits,
            **aux,
        }
            

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

    def detect(
            self,
            feats: dict,
            view: str,
            slices: torch.Tensor,
        ) -> dict:
            obj, reg = self.heads[view](
                self.slice_features(
                    feats,
                    view,
                    slices,
                )
            )

            return {
                "obj": obj,
                "reg": reg,
                "anchors": getattr(
                    self,
                    f"anchors_{view}",
                ),
                "view": view,
            }

    def classify_gt_instances(
        self,
        feats: dict,
        fragment_mask: torch.Tensor,
    ):
        """Clasificación usando instancias reales durante entrenamiento."""

        instance_features, targets, metadata = (
            extract_gt_instance_features(
                feats["p1"],
                fragment_mask,
            )
        )

        logits = self.class_head(
            instance_features
        )

        return {
            "logits": logits,
            "targets": targets,
            "metadata": metadata,
        }


    @torch.no_grad()
    def classify_pred_instances(
        self,
        feats: dict,
        regions,
        min_voxels: int = config.M3D_MIN_INSTANCE_VOXELS,
    ):
        """Segmentación en dos etapas y clasificación de cada instancia.
        Etapa 1: regiones detectadas (si `M3D_RESTRICT_TO_DETECTION`). Etapa 2: la máscara de cada hueso dentro de
        ellas; cada hueso es una instancia con todos sus fragmentos."""
        sem = segmentation_prediction(feats["sem_logits"])[0].cpu().numpy()
        region = regions_mask(regions, sem.shape) if config.M3D_RESTRICT_TO_DETECTION else None
        inst_np, seg_meta = bone_instances(sem, region)
        instances = torch.from_numpy(inst_np.astype(np.int64)).to(feats["p1"].device)[None]
        instance_features, cls_metadata = extract_pred_instance_features(feats["p1"], instances)
        if instance_features.shape[0] == 0:
            logits = feats["p1"].new_empty((0, config.M3D_CLS_CLASSES))
        elif self.mask_head:
            # Con cabeza de máscara la clasificación es por caja (`roi_predict`); aquí la clase es el hueso de la
            # instancia (la semántica)
            ids = torch.tensor([md["instance_id"] for md in cls_metadata], device=feats["p1"].device)
            logits = F.one_hot(ids - 1, config.M3D_CLS_CLASSES).float() * 20 - 10
        else:
            logits = self.class_head(instance_features)
        by_id = {item["instance_id"]: item for item in seg_meta}
        metadata = [{**item, **by_id.get(item["instance_id"], {})} for item in cls_metadata]
        return {"semantic_mask": torch.from_numpy(inst_np), "instances": instances, "logits": logits, "metadata": metadata}

    def axial_maps(self, feats: dict, slices: torch.Tensor):
        """Cortes axiales de P2 (k, C2, 64, 64) y P1 (k, C1, 128, 128), interpolados entre los dos cortes vecinos."""
        outs = []
        for name in ("p2", "p1"):
            f = feats[name][0]
            n = f.shape[1]
            p = ((slices.float() + 0.5) * n / N - 0.5).clamp(0, n - 1)
            i0 = p.floor().long()
            i1 = (i0 + 1).clamp(max=n - 1)
            w = (p - i0.float()).view(-1, 1, 1, 1).to(f.dtype)
            s = take(f, "axial", i0) * (1 - w) + take(f, "axial", i1) * w
            outs.append(s.detach() if config.M3D_MASK_DETACH else s)
        return outs

    def roi_forward(self, feats: dict, slices: torch.Tensor, rois: torch.Tensor, img_hr: torch.Tensor):
        """Cabeza de máscara y de clase sobre `rois` (L, 5) = (posición del corte en `slices`, x0, y0, x1, y1) en
        coordenadas del cubo. `img_hr`: (k, 1, 2N, 2N) en [0, 1]. -> logits de máscara (L, 3, M, M), de clase (L, 3)."""
        p2, p1 = self.axial_maps(feats, slices)
        r = rois.float()
        f = torch.cat([roi_align(p2.float(), r, config.M3D_ROI_FEAT, p2.shape[-1] / N, 2, True),
                       roi_align(p1.float(), r, config.M3D_ROI_FEAT, p1.shape[-1] / N, 2, True)], 1)
        img = input_channels(img_hr.float(), self.trab_window)
        ct = roi_align(img, r, config.M3D_MASK_SIZE, img.shape[-1] / N, 2, True)
        mask_logits, vec = self.roi_head(f, ct)
        return mask_logits, self.class_head(vec)

    @torch.no_grad()
    def roi_predict(self, feats: dict, preds: list, slices: torch.Tensor, img_hr, score_thr=config.M3D_MASK_SCORE_THR):
        """Inferencia de la cabeza de máscara: para cada corte `slices[j]`, las cajas de `preds[j]` con puntaje ≥
        `score_thr` -> clase de cada caja y su máscara (la de la clase predicha), pegada en el corte a doble
        resolución. Devuelve (etiquetas uint8 (k, 2N, 2N): 0 fondo, 1 sacro, 2 coxal izq., 3 coxal der.;
        lista de ROI con corte, caja, puntaje, clase y probabilidades)."""
        dev = feats["p1"].device
        H = img_hr.shape[-1]
        f = H / N
        labels = np.zeros((len(slices), H, H), np.uint8)
        rois_out = []
        todo = [j for j, p in enumerate(preds) if (p["scores"] >= score_thr).any()]
        for c0 in range(0, len(todo), config.M3D_MASK_CHUNK):
            js = todo[c0:c0 + config.M3D_MASK_CHUNK]
            sl = slices[js].to(dev)
            im = img_hr[sl.cpu()].to(dev).float().div_(255)[:, None]
            rows, meta = [], []
            for q, j in enumerate(js):
                keep = preds[j]["scores"] >= score_thr
                for b, sc in zip(preds[j]["boxes"][keep], preds[j]["scores"][keep]):
                    rows.append([q, *b.tolist()]); meta.append((j, b.tolist(), float(sc)))
            rois = torch.tensor(rows, device=dev, dtype=torch.float32)
            with det.autocast(dev):
                ml, cl = self.roi_forward(feats, sl, rois, im)
            prob = torch.softmax(cl.float(), 1)
            cls = prob.argmax(1)
            mp = torch.sigmoid(ml.float()[torch.arange(len(cls), device=dev), cls])          # (L, M, M)
            best = torch.zeros(len(js), 3, H, H, device=dev)
            for i, (q, x0, y0, x1, y1) in enumerate(rows):
                a0, b0 = int(math.floor(x0 * f)), int(math.floor(y0 * f))
                a1, b1 = min(int(math.ceil(x1 * f)), H), min(int(math.ceil(y1 * f)), H)
                if a1 <= a0 or b1 <= b0:
                    continue
                m = F.interpolate(mp[i][None, None], size=(b1 - b0, a1 - a0), mode="bilinear", align_corners=False)[0, 0]
                c = int(cls[i])
                best[q, c, b0:b1, a0:a1] = torch.maximum(best[q, c, b0:b1, a0:a1], m)
            mx, arg = best.max(1)
            lab = torch.where(mx >= config.M3D_MASK_THR, arg + 1, torch.zeros_like(arg))
            labels[js] = lab.cpu().numpy().astype(np.uint8)
            for i, (j, box, sc) in enumerate(meta):
                rois_out.append({"slice": int(slices[j]), "box": box, "score": sc, "class": int(cls[i]),
                                 "prob": prob[i].cpu().numpy()})
        return labels, rois_out

    def gammas(self) -> dict:
        out = {f"enc{i}": float(m.gamma.detach()) for i, m in enumerate(self.attn, start=1) if hasattr(m, "gamma")}
        out |= {f"dec_{k}": float(m.gamma.detach()) for k, m in self.dec_attn.items()}
        out |= {f"head_{v}": float(h.attn.gamma.detach()) for v, h in self.heads.items() if hasattr(h.attn, "gamma")}
        return out

# ======================================================================================
# Pérdida
# ======================================================================================
def assign_objectness_targets(
    anchors: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
):
    """Asigna anclas sin utilizar la categoría anatómica."""

    num_anchors = anchors.shape[0]

    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
    obj_target = torch.zeros(num_anchors, device=anchors.device)
    reg_target = torch.zeros(num_anchors, 4, device=anchors.device)

    boxes = gt[valid]

    if boxes.shape[0] == 0:
        labels[:] = 0
        return labels, obj_target, reg_target

    ious = det.box_iou(anchors, boxes)
    max_iou, matched = ious.max(dim=1)

    labels[max_iou < config.NEG_IOU] = 0
    labels[max_iou >= config.POS_IOU] = 1

    best_anchor = ious.argmax(dim=0)
    labels[best_anchor] = 1
    matched[best_anchor] = torch.arange(boxes.shape[0], device=anchors.device)

    positive = labels == 1
    obj_target[positive] = 1.0
    reg_target[positive] = det.calcular_offsets(
        anchors[positive], boxes[matched[positive]]
    )

    return labels, obj_target, reg_target


def objectness_detection_loss(
    out: dict,
    gt: torch.Tensor,
    valid: torch.Tensor,
):
    """Objectness BCE + regresión Smooth L1."""

    obj = out["obj"].float()
    reg = out["reg"].float()
    anchors = out["anchors"]

    obj_losses = []
    box_losses = []

    for batch_id in range(obj.shape[0]):
        labels, obj_target, reg_target = assign_objectness_targets(
            anchors, gt[batch_id], valid[batch_id]
        )

        selected, positive = det.sample_anchors(labels)

        obj_loss = F.binary_cross_entropy_with_logits(
            obj[batch_id, selected, 0],
            obj_target[selected],
        )

        if positive.numel():
            box_loss = F.smooth_l1_loss(
                reg[batch_id, positive],
                reg_target[positive],
            )
        else:
            box_loss = reg[batch_id].sum() * 0.0

        obj_losses.append(obj_loss)
        box_losses.append(box_loss)

    obj_loss = torch.stack(obj_losses).mean()
    box_loss = torch.stack(box_losses).mean()

    total = (
        config.M3D_LOSS_WEIGHTS["det"] * (obj_loss + config.M3D_BOX_WEIGHT * box_loss)
    )

    terms = {
        "obj": float(obj_loss.detach()),
        "box": float(box_loss.detach()),
    }

    return total, terms

def step_loss(model, x, m, gen, n_slices=config.M3D_DET_SLICES, hr=None):
    """Pérdida multitarea de detección, segmentación y clasificación (y máscara por caja).
    `hr` = (imagen y códigos a doble resolución, parámetros del aumento o None): necesario si el modelo tiene cabeza
    de máscara. Entonces la clasificación es la rama de clase de cada caja (ROI) y se suma la pérdida de máscara."""
    with det.autocast(x.device):
        feats = model(x)

        from src.models.segmentation import segmentation_loss
        from src.models.classification import classification_loss

        seg_loss, seg_terms = segmentation_loss(            # tres huesos + fondo: CE + Dice
            feats["sem_logits"], m,
            ce_weight=config.M3D_SEG_CE_WEIGHT, dice_weight=config.M3D_SEG_DICE_WEIGHT)
        if "sem_logits_aux" in feats:                        # supervisión profunda sobre P1 (128³)
            aux_loss, _ = segmentation_loss(feats["sem_logits_aux"], m[::2, ::2, ::2],
                                            ce_weight=config.M3D_SEG_CE_WEIGHT, dice_weight=config.M3D_SEG_DICE_WEIGHT)
            seg_loss = seg_loss + config.M3D_SEG_AUX_WEIGHT * aux_loss
            seg_terms["seg_aux"] = float(aux_loss.detach())

        if not model.mask_head:
            cls_out = model.classify_gt_instances(feats, m)
            anat_loss = classification_loss(cls_out["logits"], cls_out["targets"])

        region_map = REGION_LUT.to(m.device)[m]
        det_losses, terms = [], {}

        for view in config.M3D_DET_VIEWS:
            slices = sample_slices(region_map, x, view, n_slices, gen)
            out = model.detect(feats, view, slices)
            mask_view = VIEW_LUT_T[view].to(m.device)[take(m[None], view, slices)[:, 0]]
            gt, valid = det.region_boxes(mask_view, len(CLASSES[view]))

            det_loss, det_terms = objectness_detection_loss(out, gt, valid)
            det_losses.append(det_loss)

            terms[f"obj_{view}"] = det_terms["obj"]
            terms[f"box_{view}"] = det_terms["box"]

            if model.mask_head and view == "axial":
                mask_l, anat_loss = roi_losses(model, feats, slices, gt, valid, hr)

        detection_loss = torch.stack(det_losses).mean()
        total = (
            detection_loss
            + config.M3D_LOSS_WEIGHTS["seg"] * seg_loss
            + config.M3D_LOSS_WEIGHTS["anat"] * anat_loss
        )
        if model.mask_head:
            total = total + config.M3D_LOSS_WEIGHTS["mask"] * mask_l
            terms["mask"] = float(mask_l.detach())

    terms["obj"] = float(np.mean([terms[f"obj_{v}"] for v in config.M3D_DET_VIEWS]))
    terms["box"] = float(np.mean([terms[f"box_{v}"] for v in config.M3D_DET_VIEWS]))
    terms.update(seg_terms)
    terms["seg"] = float(seg_loss.detach())
    terms["anat"] = float(anat_loss.detach())

    return total, terms


def roi_losses(model, feats, slices, gt, valid, hr):
    """Pérdidas de la cabeza de máscara y de la rama de clase sobre las cajas reales de los cortes `slices`, movidas
    un poco (`jitter_boxes`). -> (máscara, clasificación)."""
    img_hr, msk_hr, aug = hr
    j, c = valid.nonzero(as_tuple=True)
    if len(j) == 0:
        z = feats["p1"].sum() * 0.0
        return z, z
    boxes = jitter_boxes(gt[j, c].float())
    rois = torch.cat([j.float()[:, None], boxes], 1)
    im, bone = hires_slices(img_hr, msk_hr, slices, gt.device, aug)
    ml, cl = model.roi_forward(feats, slices, rois, im)
    onehot = torch.stack([bone == k + 1 for k in range(3)], 1).float()                # (k, 3, 2N, 2N)
    tm = roi_align(onehot, rois, config.M3D_MASK_SIZE, onehot.shape[-1] / N, 2, True)
    target = (tm[torch.arange(len(c), device=tm.device), c] > 0.5).float()
    return mask_loss(ml, c, target), classification_loss(cl.float(), c)


def to_gpu(img_u8, msk_u8, device, augment=False, params: dict | None = None):
    x = img_u8.to(device, non_blocking=True).float().div_(255)[None]       # (1, 1, Z, Y, X) del batch de 1
    m = msk_u8.to(device, non_blocking=True)[0].long()
    if augment:
        x, m = augment_3d(x, m, params)
    return x, m


def input_channels(x: torch.Tensor, trab_window) -> torch.Tensor:
    """(B, 1, ...) en la ventana de `config.HU_WINDOW` -> agrega la ventana trabecular como segundo canal."""
    if not trab_window:
        return x
    lo, hi = config.HU_WINDOW
    a, b = trab_window
    hu = x[:, :1] * (hi - lo) + lo
    return torch.cat([x[:, :1], ((hu - a) / (b - a)).clamp(0, 1)], 1)


def hires_slices(img_hr, msk_hr, slices: torch.Tensor, device, aug: dict | None = None):
    """Cortes axiales `slices` (índices del cubo) del caché a doble resolución -> imagen (k, 1, 2N, 2N) en [0, 1] y
    hueso (k, 2N, 2N) (0 fondo, 1 sacro, 2 coxal izq., 3 coxal der.). Con `aug` (de `augment_3d`), aplica la misma
    transformación geométrica e de intensidad que al cubo, muestreando solo esos cortes."""
    if aug is None:
        idx = slices.cpu()
        im = img_hr[idx].to(device).float().div_(255)[:, None]
        ms = msk_hr[idx].to(device).long()
    else:
        Z, H, W = img_hr.shape
        zn = (2 * slices.to(device).float() + 1) / Z - 1
        yn = (2 * torch.arange(H, device=device).float() + 1) / H - 1
        xn = (2 * torch.arange(W, device=device).float() + 1) / W - 1
        zz, yy, xx = torch.meshgrid(zn, yn, xn, indexing="ij")
        pts = torch.stack([xx, yy, zz, torch.ones_like(xx)], -1)                  # (k, H, W, 4) en orden (x, y, z)
        grid = (pts @ aug["theta"][0].T)[None]                                        # (1, k, H, W, 3)
        del pts, zz, yy, xx
        vol = img_hr.to(device)[None, None].float().div_(255)
        im = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0].transpose(0, 1)
        del vol
        vol = msk_hr.to(device)[None, None].float()
        ms = F.grid_sample(vol, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0].round().long()
        del vol, grid
        im = (im.clamp(0, 1) ** aug["gamma"] * (1 + aug["contrast"]) + aug["bright"]).clamp(0, 1)
    bone = torch.where(ms > 0, (ms - 1) // 10 + 1, torch.zeros_like(ms))
    return im, bone


# ======================================================================================
# Validación
# ======================================================================================
@torch.no_grad()
def postprocess_objectness(
    out: dict,
    score_thr: float = config.SCORE_THR,
    iou_thr: float = config.NMS_IOU,
    pre_nms: int = config.PRE_NMS,
    max_det: int = config.M3D_MAX_DET,
):
    """Convierte objectness + regresión en cajas finales tras NMS."""
    scores = torch.sigmoid(out["obj"].float()).squeeze(-1).cpu()
    reg, anchors = out["reg"].float().cpu(), out["anchors"].cpu()

    results = []
    for i in range(scores.shape[0]):
        boxes = det.decode_boxes(anchors, reg[i]).clamp(0, config.INPUT_SIZE)
        s = scores[i]

        keep = (s > score_thr).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)})
            continue

        keep = keep[s[keep].argsort(descending=True)[:pre_nms]]
        keep_nms = det.nms(boxes[keep], s[keep], iou_thr)[:max_det]

        results.append({
            "boxes": boxes[keep][keep_nms],
            "scores": s[keep][keep_nms],
        })

    return results

@torch.no_grad()
def group_boxes_across_slices(
    predictions,
    slices,
    iou_threshold: float = 0.3,
):
    """Agrupa cajas axiales de cortes consecutivos para formar regiones 3D."""

    regions = []

    for prediction, slice_id in zip(predictions, slices.tolist()):
        boxes = prediction["boxes"]

        for box in boxes:
            best_region = None
            best_iou = 0.0

            for region in regions:
                last_slice = region["last_slice"]

                if slice_id - last_slice > 1 or slice_id < last_slice:
                    continue

                last_box = region["last_box"]

                iou = det.box_iou(
                    box.unsqueeze(0),
                    last_box.unsqueeze(0),
                )[0, 0].item()

                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_region = region

            if best_region is None:
                regions.append({
                    "boxes_by_slice": {slice_id: box.unsqueeze(0)},
                    "last_slice": slice_id,
                    "last_box": box,
                })
                continue

            best_region["boxes_by_slice"][slice_id] = box.unsqueeze(0)
            best_region["last_slice"] = slice_id
            best_region["last_box"] = box

    return regions

def regions_mask(regions, shape) -> np.ndarray:
    """Unión de las regiones 3D detectadas (cajas axiales agrupadas entre cortes) -> máscara bool (Z, Y, X)."""
    out = np.zeros(shape, bool)
    for r in regions:
        out |= boxes_to_region_mask_3d(r["boxes_by_slice"], shape, "cpu").numpy()
    return out


@torch.no_grad()
def infer_instances_axial(
    model,
    x,
    score_thr: float = config.SCORE_THR,
    nms_iou: float = config.NMS_IOU,
    track_iou: float = config.M3D_TRACK_IOU,
    min_voxels: int = config.M3D_MIN_INSTANCE_VOXELS,
    img_hr=None,
):
    """Pipeline axial: detección -> regiones 3D -> segmentación -> clasificación. Con cabeza de máscara e `img_hr`
    (el CT a doble resolución), además la máscara y la clase de cada caja (`mask_labels`, `rois`)."""

    model.eval()

    with det.autocast(x.device):
        feats = model(x)

    slices = torch.arange(
        0,
        N,
        device=x.device,
    )

    with det.autocast(x.device):            # los mapas vienen en float16; las cabezas tienen pesos float32
        det_out = model.detect(
            feats,
            "axial",
            slices,
        )

    predictions = postprocess_objectness(
        det_out,
        score_thr=score_thr,
        iou_thr=nms_iou,
    )

    regions = group_boxes_across_slices(
        predictions,
        slices,
        iou_threshold=track_iou,
    )

    with det.autocast(x.device):
        result = model.classify_pred_instances(
            feats,
            regions,
            min_voxels=min_voxels,
        )

    result["detections"] = predictions
    result["regions"] = regions
    if model.mask_head and img_hr is not None:
        result["mask_labels"], result["rois"] = model.roi_predict(feats, predictions, slices, img_hr)

    return result

def object_detection_metrics(predictions, gt_objects, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)

    aps, matched_ious = {}, []

    for threshold in thresholds:
        scores_all, tp_all = [], []
        total_gt = sum(len(gt) for gt in gt_objects)

        for pred, gt in zip(predictions, gt_objects):
            boxes_pred, scores = pred["boxes"], pred["scores"]
            order = scores.argsort(descending=True)
            boxes_pred, scores = boxes_pred[order], scores[order]

            used = set()
            for box, score in zip(boxes_pred, scores):
                scores_all.append(float(score))

                if len(gt) == 0:
                    tp_all.append(0)
                    continue

                ious = det.box_iou(box.unsqueeze(0), gt.float())[0]
                best_iou, best_idx = ious.max(dim=0)
                best_iou, best_idx = float(best_iou), int(best_idx)

                if best_iou >= threshold and best_idx not in used:
                    tp_all.append(1)
                    used.add(best_idx)
                    if abs(threshold - 0.5) < 1e-6:
                        matched_ious.append(best_iou)
                else:
                    tp_all.append(0)

        if total_gt == 0:
            aps[round(float(threshold), 2)] = float("nan")
            continue

        if not scores_all:
            aps[round(float(threshold), 2)] = 0.0
            continue

        order = np.argsort(-np.asarray(scores_all))
        tp = np.asarray(tp_all, dtype=float)[order]
        fp = 1.0 - tp

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        aps[round(float(threshold), 2)] = det.average_precision(recall, precision)

    valid_aps = [x for x in aps.values() if not np.isnan(x)]

    return {
        "mAP50": aps.get(0.5, float("nan")),
        "mAP50_95": float(np.mean(valid_aps)) if valid_aps else float("nan"),
        "mIoU": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }

@torch.no_grad()
def validate(model, cases, device, stride=config.M3D_VAL_STRIDE):
    model.eval()

    det_preds, det_gt = [], []
    cls_logits_all, cls_targets_all = [], []
    seg_results, mask_dice = [], []

    ds = VolumeDataset(cases, hires=model.mask_head)

    for i in range(len(ds)):
        item = ds[i]
        x, m = to_gpu(item[0][None], item[1][None], device)

        with det.autocast(device):
            feats = model(x)

        # Segmentación
        seg_results.append(segmentation_metrics(feats["sem_logits"], m))     # pelvis y macro de los tres huesos

        # Clasificación (dentro de autocast: los mapas vienen en float16 y las cabezas tienen pesos float32)
        if not model.mask_head:
            with det.autocast(device):
                cls_out = model.classify_gt_instances(feats, m)
            if cls_out["logits"].shape[0] > 0:
                cls_logits_all.append(cls_out["logits"].float().cpu())
                cls_targets_all.append(cls_out["targets"].cpu())

        # Detección
        slices = torch.arange(0, N, stride, device=device)

        for view in config.M3D_DET_VIEWS:
            mv = VIEW_LUT_T[view].to(device)[take(m[None], view, slices)[:, 0]]
            gt, valid = det.region_boxes(mv, len(CLASSES[view]))
            with det.autocast(device):
                out = model.detect(feats, view, slices)
            preds = postprocess_objectness(out)

            for j in range(len(preds)):
                det_preds.append(preds[j])
                det_gt.append(gt[j][valid[j]].cpu())

            if model.mask_head and view == "axial":
                img_hr, msk_hr = item[2], item[3]
                # Rama de clase sobre las cajas reales (F1 / AUC de la clasificación anatómica por caja)
                j_, c_ = valid.nonzero(as_tuple=True)
                if len(j_):
                    im = img_hr[slices.cpu()].to(device).float().div_(255)[:, None]
                    rois = torch.cat([j_.float()[:, None], gt[j_, c_].float()], 1)
                    with det.autocast(device):
                        _, cl = model.roi_forward(feats, slices, rois, im)
                    cls_logits_all.append(cl.float().cpu()); cls_targets_all.append(c_.cpu())
                    del im
                # Máscara con las cajas predichas, frente al hueso real a doble resolución (Dice por hueso en los cortes)
                lab, _ = model.roi_predict(feats, preds, slices, img_hr)
                _, bone = hires_slices(img_hr, msk_hr, slices, "cpu")
                bone = bone.numpy()
                d = [2 * ((lab == c) & (bone == c)).sum() / ((lab == c).sum() + (bone == c).sum())
                     for c in (1, 2, 3) if (bone == c).any()]
                mask_dice.append(float(np.mean(d)) if d else float("nan"))

        del feats

    # Métricas de detección
    det_metrics = object_detection_metrics(det_preds, det_gt)

    # Métricas de clasificación
    if cls_logits_all:
        cls_logits = torch.cat(cls_logits_all, dim=0)
        cls_targets = torch.cat(cls_targets_all, dim=0)
        cls_metrics = classification_metrics(cls_logits, cls_targets)
    else:
        cls_metrics = {"macro_f1": float("nan"), "macro_auc": float("nan")}

    # Métricas de segmentación
    seg_dice = float(np.mean([x["dice"] for x in seg_results]))
    seg_iou = float(np.mean([x["iou"] for x in seg_results]))
    sem_dice = float(np.mean([x["dice_sem"] for x in seg_results]))
    bone_dice = {n: float(np.nanmean([x[f"dice_{n}"] for x in seg_results])) for n in ("sacro", "izq", "der")}
    sem_iou = float(np.mean([x["iou_sem"] for x in seg_results]))

    row = {
        "val_mAP50": det_metrics["mAP50"],
        "val_mAP50_95": det_metrics["mAP50_95"],
        "val_mIoU_det": det_metrics["mIoU"],
        "val_dice_seg": seg_dice,
        "val_iou_seg": seg_iou,
        "val_dice_sem": sem_dice,
        "val_iou_sem": sem_iou,
        "val_macro_f1": cls_metrics["macro_f1"],
        "val_macro_auc": cls_metrics["macro_auc"],
        "val_dice_mask": float(np.nanmean(mask_dice)) if mask_dice else float("nan"),
        **{f"val_dice_{n}": v for n, v in bone_dice.items()},
    }
    # Puntaje para elegir best.pth y para la parada temprana: promedio de una métrica por cabeza.
    row["val_score"] = float(np.nanmean([row[k] for k in config.M3D_SELECT_METRICS]))
    return row


# ======================================================================================
# Entrenamiento
# ======================================================================================

def run_config(attention: str, epochs: int) -> dict:
    keys = [k for k in dir(config) if k.startswith("M3D_")]
    return {"arch": "fundidora3d_det2d", "attention": attention, "cbam": attention == "cbam",
            "head_attention": attention != "none" and config.HEAD_ATTENTION,
            **{k.lower(): (str(getattr(config, k)) if isinstance(getattr(config, k), Path) else getattr(config, k))
               for k in keys}, "epochs": epochs,
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


@torch.no_grad()
def eval_loss(model, ids, device) -> dict:
    """Pérdida (total y por cabeza) en modo evaluación, sin aumento, con los cortes de detección sorteados con una
    semilla fija: comparable época a época con la pérdida de entrenamiento."""
    model.eval()
    gen = torch.Generator().manual_seed(config.SEED)
    rows = []
    for i in range(len(ids)):
        item = VolumeDataset([ids[i]], hires=model.mask_head)[0]
        x, m = to_gpu(item[0][None], item[1][None], device)
        hr = (item[2], item[3], None) if model.mask_head else None
        loss, t = step_loss(model, x, m, gen, hr=hr)
        rows.append({"loss": float(loss), **{k: t[k] for k in ("obj", "box", "seg", "anat", "mask") if k in t}})
    return pd.DataFrame(rows).mean().to_dict()


def train(args):
    det.seed_everything()
    device = det.get_device(args.device)
    torch.backends.cudnn.benchmark = False   # las formas cambian (aumento): re-afinar costaba ~10 s
    attention = _attention(args)
    name = args.name or RUNS[attention]
    out_dir = config.DETECTION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_config(attention, args.epochs)

    tr_ids, va_ids = case_ids("train"), case_ids("val")
    tr_eval_ids = tr_ids[::max(len(tr_ids) // config.M3D_TRAIN_EVAL_CASES, 1)][:config.M3D_TRAIN_EVAL_CASES]
    loader = torch.utils.data.DataLoader(VolumeDataset(tr_ids, hires=config.M3D_MASK_HEAD), batch_size=1, shuffle=True, num_workers=args.workers,
                                         persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
                                         generator=torch.Generator().manual_seed(config.SEED))
    model = Fundidora3DDetector(attention).to(device)
    opt = _optimizer(model)
    steps = len(loader)
    # Coseno sobre `args.epochs`, sin calentamiento: el lr parte de M3D_LR y baja hasta ~0 en la última época.
    total = args.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    gen = torch.Generator().manual_seed(config.SEED)

    start, best, history = 0, -1.0, []
    last_path = out_dir / "last.pth"
    if args.resume and last_path.exists():
        ck = load_checkpoint(last_path)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
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
        for batch in loader:
            aug = {}
            x, m = to_gpu(batch[0], batch[1], device, augment=True, params=aug)
            hr = (batch[2][0], batch[3][0], aug) if model.mask_head else None
            loss, terms = step_loss(model, x, m, gen, hr=hr)
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
        # Misma pérdida en validación, cada época (curvas de pérdida train frente a val)
        t1 = time.time()
        row |= {f"val_{k}": v for k, v in eval_loss(model, va_ids, device).items()}
        row["val_loss_time_s"] = time.time() - t1
        if device.type == "cuda":
            row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        is_val = (epoch + 1) % config.M3D_VAL_EVERY == 0 or epoch + 1 == args.epochs
        msg = f"[{name}] época {epoch + 1}/{args.epochs} loss train {row['train_loss']:.3f} val {row['val_loss']:.3f} (obj {row['train_obj']:.3f} box {row['train_box']:.3f} seg {row['train_seg']:.3f} anat {row['train_anat']:.3f}) {row['train_time_s']:.0f} s"
        if is_val:
            t1 = time.time()
            row |= validate(model, va_ids, device)
            # Mismas métricas en un subconjunto fijo de train, sin aumento (curvas de métricas train frente a val)
            row |= {k.replace("val_", "train_", 1): v for k, v in validate(model, tr_eval_ids, device).items()}
            row["val_time_s"] = time.time() - t1

            msg += (
                f" | det mAP50 {row['val_mAP50']:.3f} "
                f"mAP50-95 {row['val_mAP50_95']:.3f} "
                f"mIoU {row['val_mIoU_det']:.3f} "
                f"| seg HuesoDice {row['val_dice_seg']:.3f} "
                f"Dice huesos (macro) {row['val_dice_sem']:.3f} "
                f"IoU huesos {row['val_iou_sem']:.3f} "
                f"(sacro {row['val_dice_sacro']:.3f} izq {row['val_dice_izq']:.3f} der {row['val_dice_der']:.3f}) "
                f"| máscara Dice {row['val_dice_mask']:.3f} "
                f"| cls F1 {row['val_macro_f1']:.3f} "
                f"AUC {row['val_macro_auc']:.3f} "
                f"| puntaje {row['val_score']:.3f}"
            )
        row["time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if is_val:
            if row["val_score"] > best:
                best = row["val_score"]
                torch.save({**state, "val_score": best, **{k: row[k] for k in config.M3D_SELECT_METRICS}},
                           out_dir / "best.pth")
                msg += "  * mejor"
            low = [n for n in ("sacro", "izq", "der") if row[f"val_dice_{n}"] < config.M3D_MIN_BONE_DICE]
            if epoch + 1 >= 10 and low:                       # aviso de colapso de una clase
                msg += f"\n  ALERTA: Dice de {', '.join(low)} < {config.M3D_MIN_BONE_DICE} en validación (posible colapso)"
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "history": history, "rng": torch.get_rng_state()}, last_path)
        gam = "  ".join(f"γ {k} {v:+.3f}" for k, v in model.gammas().items())
        print(msg + f"  {gam}" + (f"  pasos saltados {skipped}" if skipped else ""), flush=True)
    return out_dir


def overfit(args):
    """Prueba de correctitud: sobreajustar `config.M3D_OVERFIT_CASES` CT sin aumento. La detección de esos CT
    debe quedar casi perfecta."""
    det.seed_everything()
    device = det.get_device(args.device)
    attention = _attention(args)
    ids = case_ids("train")[:config.M3D_OVERFIT_CASES]
    ds = VolumeDataset(ids, hires=config.M3D_MASK_HEAD)
    data = [ds[i] for i in range(len(ds))]
    model = Fundidora3DDetector(attention).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.OVERFIT_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")
    gen = torch.Generator().manual_seed(config.SEED)
    t0 = time.time()
    for it in range(args.iters):
        model.train()
        item = data[it % len(data)]
        x, m = to_gpu(item[0][None], item[1][None], device)
        loss, terms = step_loss(model, x, m, gen, hr=(item[2], item[3], None) if model.mask_head else None)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if it % 25 == 0 or it == args.iters - 1:
            print(
                f"iter {it:4d} loss {float(loss):.4f} obj {terms['obj']:.4f} "
                f"box {terms['box']:.4f} seg {terms['seg']:.4f} "
                f"anat {terms['anat']:.4f} ({time.time() - t0:.0f} s)",
                flush=True,)
    row = validate(model, ids, device, stride=2)
    out_dir = config.DETECTION_DIR / f"overfit_{RUNS[attention]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    print(
        "Overfit: "
        f"det mAP50 {row['val_mAP50']:.3f} "
        f"mAP50-95 {row['val_mAP50_95']:.3f} "
        f"mIoU {row['val_mIoU_det']:.3f} "
        f"| seg HuesoDice {row['val_dice_seg']:.3f} "
        f"Dice huesos (macro) {row['val_dice_sem']:.3f} "
        f"IoU huesos {row['val_iou_sem']:.3f} "
                f"(sacro {row['val_dice_sacro']:.3f} izq {row['val_dice_izq']:.3f} der {row['val_dice_der']:.3f}) "
                f"| máscara Dice {row['val_dice_mask']:.3f} "
        f"| cls F1 {row['val_macro_f1']:.3f} "
        f"AUC {row['val_macro_auc']:.3f}"
    )
    # Criterio para seguir al entrenamiento completo
    checks = {f"Dice {n}": row[f"val_dice_{n}"] for n in ("sacro", "izq", "der")}
    if model.mask_head:
        checks["Dice máscara"] = row["val_dice_mask"]
    bad = {k: v for k, v in checks.items() if not v >= config.M3D_MIN_BONE_DICE}
    print("OVERFIT OK: se puede entrenar" if not bad else
          "OVERFIT NO PASA (" + ", ".join(f"{k} {v:.3f}" for k, v in bad.items()) + f" < {config.M3D_MIN_BONE_DICE}): no entrenar")


@torch.no_grad()
def calibrate(args):
    """Pérdida inicial de cada tarea (modelo sin entrenar, semilla fija, sin aumento) para fijar los λ:
    λ_t = prioridad_t / pérdida inicial_t (`config.M3D_LOSS_PRIORITY`). Los valores medidos se copian a
    `config.M3D_LOSS_INIT`."""
    det.seed_everything()
    device = det.get_device(args.device)
    model = Fundidora3DDetector(_attention(args)).to(device).train()
    gen = torch.Generator().manual_seed(config.SEED)
    ids = case_ids("train")[:config.M3D_CALIBRATION_CASES]
    ds = VolumeDataset(ids, hires=model.mask_head)
    tasks = ("anat", "seg", "mask") if model.mask_head else ("anat", "seg")
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        x, m = to_gpu(item[0][None], item[1][None], device)
        _, t = step_loss(model, x, m, gen, hr=(item[2], item[3], None) if model.mask_head else None)
        rows.append({"case_id": ids[i], **{k: t[k] for k in ("obj", "box", *tasks)}})
        print(f"{ids[i]}  " + "  ".join(f"{k} {rows[-1][k]:.4f}" for k in ("obj", "box", *tasks)), flush=True)
    det_init = {k: float(np.mean([r[k] for r in rows])) for k in ("obj", "box")}
    box_w = det_init["obj"] / det_init["box"]          # la caja en la escala del objectness
    init = {"det": det_init["obj"] + box_w * det_init["box"],
            **{t: float(np.mean([r[t] for r in rows])) for t in tasks}}
    weights = {t: config.M3D_LOSS_PRIORITY[t] / init[t] for t in init}
    out = {"cases": rows, "det_init": det_init, "box_weight": box_w, "loss_init": init, "priority": config.M3D_LOSS_PRIORITY, "weights": weights,
           "share_at_start": {t: weights[t] * init[t] / sum(weights[k] * init[k] for k in init) for t in init}}
    config.DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    (config.DETECTION_DIR / "lambda_calibration.json").write_text(json.dumps(out, indent=2))
    print("M3D_DET_INIT = {" + ", ".join(f'"{k}": {v:.4f}' for k, v in det_init.items()) + f"}}   (λ_box {box_w:.2f})")
    print("M3D_LOSS_INIT = {" + ", ".join(f'"{t}": {v:.4f}' for t, v in init.items()) + "}")
    print("λ: " + "  ".join(f"{t} {w:.3f}" for t, w in weights.items()))


# ======================================================================================
# Evaluación final (todas las cabezas, pipeline completo)
# ======================================================================================

SEM_CLASSES = {"sacro": 1, "coxal_izq": 2, "coxal_der": 3}             # máscaras de la cabeza de segmentación


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def separation_eval(cid: str, inst_cube: np.ndarray, img_path, split: str, hires: bool = False) -> tuple[list, dict]:
    """Distancia de separación en el espacio ORIGINAL del CT, con el espaciado del header.

    - Real: `src.metrics.distancias.filas_caso` (módulo de P4): cada fragmento etiquetado del dataset frente al
      principal de su hueso (el de más vóxeles), EDT 3D. Los fragmentos que se tocan quedan a un paso de rejilla.
    - Predicha: la máscara de cada hueso vuelve del cubo (vecino más cercano); sus componentes conexos, el más grande
      es el principal, y la misma EDT con el espaciado del header (`separation_distances`). Cada fragmento real se
      empareja con el componente predicho que más se solapa con él: si es el principal, la red lo dejó unido a su
      hueso (distancia predicha 0); si no se solapa con nada, NaN."""
    from src.dataset import preprocessing as pre
    from src.metrics.distancias import filas_caso
    img = pre.reorient(pre.load_volume(img_path))
    codes = pre.sitk.GetArrayFromImage(pre.reorient(pre.load_volume(config.LABEL_DIR / f"{cid}.mha")))
    geom = pre.cube_geometry(img)
    if hires:                                                            # salida de la cabeza de máscara
        geom = pre.hires_geometry(geom, config.M3D_HIRES_FACTOR)
    pred = pre.cube_to_original(inst_cube, geom, img)
    sp_xyz = tuple(float(v) for v in img.GetSpacing())
    sp = sp_xyz[::-1]                                                    # (z, y, x) en mm
    min_vox = int(np.ceil(config.M3D_MIN_FRAGMENT_MM3 / np.prod(sp)))
    gt_rows = [r for r in filas_caso(codes, sp_xyz) if not r["principal"]]
    rows, counts, comp = [], {}, {}
    for b, region in enumerate(config.REGION_NAMES, start=1):
        lp, mp, rp = separation_distances(pred == b, sp, min_vox)
        comp[region] = (lp, mp, {r["comp"]: r["distance_mm"] for r in rp})
        counts[f"frag_pred_{region}"] = len(rp)
    for r in gt_rows:
        lp, mp, dist_p = comp[r["region"]]
        ov = lp[codes == r["fragmento"]]
        ov = ov[ov > 0]
        k = int(np.bincount(ov).argmax()) if len(ov) else 0
        rows.append({"case_id": cid, "split": split, "region": r["region"], "fragmento": r["fragmento"],
                     "n_voxels": r["n_voxels"], "dist_gt_mm": r["distancia_mm"],
                     "dist_pred_mm": 0.0 if (k and k == mp) else dist_p.get(k, np.nan)})
    counts["frag_gt"] = len(gt_rows)
    return rows, counts


def refine_boxes(boxes: torch.Tensor, bone: np.ndarray, margin: int) -> torch.Tensor:
    """Ajusta cada caja al hueso segmentado (máscara 2D del corte) dentro de la caja ampliada `margin` px.
    Si dentro no hay hueso, la caja queda igual."""
    out = boxes.clone()
    h, w = bone.shape
    for i, (x0, y0, x1, y1) in enumerate(boxes.tolist()):
        a0, b0 = max(int(np.floor(x0)) - margin, 0), max(int(np.floor(y0)) - margin, 0)
        a1, b1 = min(int(np.ceil(x1)) + margin, w), min(int(np.ceil(y1)) + margin, h)
        sub = bone[b0:b1, a0:a1]
        if sub.size == 0 or not sub.any():
            continue
        ys, xs = np.nonzero(sub.any(1))[0], np.nonzero(sub.any(0))[0]
        out[i] = torch.tensor([a0 + xs[0], b0 + ys[0], a0 + xs[-1] + 1, b0 + ys[-1] + 1], dtype=out.dtype)
    return out


def detection_eval(preds: list, gts: list, thresholds=np.arange(0.50, 0.96, 0.05)) -> dict:
    """mAP COCO (101 puntos) por umbral de IoU, IoU media de las cajas reales detectadas a IoU 0,5, curvas PR a
    0,5 y 0,75 y la mejor IoU de cada caja real. Cada caja predicha se empareja con a lo sumo una caja real del
    mismo corte."""
    total = sum(len(g) for g in gts)
    per_gt = []
    for g, p in zip(gts, preds):
        if len(g) == 0:
            continue
        iou = det.box_iou(g.float(), p["boxes"]) if len(p["boxes"]) else torch.zeros(len(g), 1)
        per_gt += iou.max(1).values.tolist()
    aps, curves, matched = {}, {}, []
    for t in thresholds:
        scores, tps = [], []
        for g, p in zip(gts, preds):
            order = p["scores"].argsort(descending=True)
            used = set()
            for b, s in zip(p["boxes"][order], p["scores"][order]):
                scores.append(float(s))
                if len(g) == 0:
                    tps.append(0); continue
                iou = det.box_iou(b[None], g.float())[0]
                v, j = float(iou.max()), int(iou.argmax())
                if v >= t and j not in used:
                    tps.append(1); used.add(j)
                    if abs(t - 0.5) < 1e-6:
                        matched.append(v)
                else:
                    tps.append(0)
        order = np.argsort(-np.asarray(scores))
        tp = np.asarray(tps, float)[order]
        rec = np.cumsum(tp) / max(total, 1)
        prec = np.cumsum(tp) / np.maximum(np.arange(1, len(tp) + 1), 1)
        aps[round(float(t), 2)] = det.average_precision(rec, prec) if len(tp) else 0.0
        if round(float(t), 2) in (0.5, 0.75):
            step = max(len(rec) // 400, 1)
            curves[str(round(float(t), 2))] = {"recall": rec[::step].tolist(), "precision": prec[::step].tolist()}
    return {"mAP50": aps[0.5], "mAP75": aps[0.75], "mAP50_95": float(np.mean(list(aps.values()))),
            "mIoU": float(np.mean(matched)) if matched else 0.0, "ap_by_iou": {str(k): v for k, v in aps.items()},
            "curves": curves, "gt_best_iou": per_gt, "n_gt": total, "n_pred": sum(len(p["boxes"]) for p in preds)}


def _macro_f1_auc(y: np.ndarray, prob: np.ndarray) -> dict:
    from src.models.classification import classification_metrics
    if len(y) == 0:
        return {"macro_f1": float("nan"), "macro_auc": float("nan")}
    return classification_metrics(torch.log(torch.from_numpy(prob).clamp_min(1e-8)), torch.from_numpy(y))


@torch.no_grad()
def _dist_summary(dist: pd.DataFrame) -> dict:
    """Resumen de la distancia de separación (un fragmento conminuto real por fila)."""
    if not len(dist):
        return {"fragments_gt": 0}
    ok = dist.dropna(subset=["dist_pred_mm"])
    far = ok[ok.dist_gt_mm > 2]
    return {"fragments_gt": int(len(dist)), "matched": float(len(ok) / len(dist)),
            "merged_with_main": float((ok.dist_pred_mm == 0).mean()) if len(ok) else float("nan"),
            "mae_mm": float((ok.dist_pred_mm - ok.dist_gt_mm).abs().mean()) if len(ok) else float("nan"),
            "mean_gt_mm": float(dist.dist_gt_mm.mean()), "gt_separated_gt_2mm": int((dist.dist_gt_mm > 2).sum()),
            "mae_mm_gt_2mm": float((far.dist_pred_mm - far.dist_gt_mm).abs().mean()) if len(far) else float("nan")}


@torch.no_grad()
def mask_head_eval(model, feats, m, preds_ax, cid, split, img_path, device, roi_rows, roi_gt, dist_rows) -> dict:
    """Evaluación de la cabeza de máscara y de la rama de clase en un CT.

    - Máscara: cajas axiales predichas -> máscaras pegadas a doble resolución -> Dice, IoU, HD95 y ASSD por hueso
      frente al hueso real a doble resolución; Dice de la pelvis; distancia de separación con esta salida.
    - Clase: sobre las cajas reales (F1 / AUC por caja) y de extremo a extremo (cada caja predicha contra la caja
      real del mismo corte con IoU ≥ 0,5; sin pareja, clase real −1).
    Agrega filas a `roi_rows`, `roi_gt` y `dist_rows`; devuelve las columnas del CT para `cases_<split>.csv`, las
    etiquetas de la cabeza de máscara y el hueso real, los dos a doble resolución (para la fusión)."""
    img_hr, msk_hr = load_hires(cid)
    sl = torch.arange(0, N, device=device)
    lab, rois = model.roi_predict(feats, preds_ax, sl, img_hr)
    _, bone = hires_slices(img_hr, msk_hr, sl, "cpu")
    bone = bone.numpy().astype(np.uint8)
    v = config.VOXEL_MM
    hm = head_metrics(lab, bone, SEM_CLASSES, spacing=(v, v / config.M3D_HIRES_FACTOR, v / config.M3D_HIRES_FACTOR))
    pel, _ = _binary_scores(torch.from_numpy(lab > 0), torch.from_numpy(bone > 0))
    d_rows, _ = separation_eval(cid, lab, img_path, split, hires=True)
    dist_rows += d_rows
    # Rama de clase sobre las cajas reales de todos los cortes
    gt, valid = det.region_boxes(REGION_LUT.to(device)[m], 3)
    for c0 in range(0, N, config.M3D_MASK_CHUNK):
        js = torch.arange(c0, min(c0 + config.M3D_MASK_CHUNK, N), device=device)
        j_, c_ = valid[js].nonzero(as_tuple=True)
        if not len(j_):
            continue
        im = img_hr[js.cpu()].to(device).float().div_(255)[:, None]
        rr = torch.cat([j_.float()[:, None], gt[js][j_, c_].float()], 1)
        with det.autocast(device):
            _, cl = model.roi_forward(feats, js, rr, im)
        roi_gt["y"] += c_.cpu().tolist()
        roi_gt["prob"] += torch.softmax(cl.float(), 1).cpu().numpy().tolist()
    # Extremo a extremo: cada caja predicha contra la real del mismo corte
    for r in rois:
        g, ok = gt[r["slice"]], valid[r["slice"]]
        true = -1
        if ok.any():
            iou = det.box_iou(torch.tensor([r["box"]], device=device), g.float())[0]
            iou[~ok] = 0
            if float(iou.max()) >= 0.5:
                true = int(iou.argmax())
        roi_rows.append({"case_id": cid, "split": split, "slice": r["slice"], "score": r["score"], "class_true": true,
                         "class_pred": r["class"], **{f"prob_{k}": float(r["prob"][k]) for k in range(3)}})
    return {"mascara_pelvis_dice": pel, **{f"mascara_{c}_{k}": val for c, cm in hm.items() for k, val in cm.items()}}, lab, bone


@torch.no_grad()   # sin esto el forward guarda el grafo y se queda sin memoria en una GPU de 6 GB
def evaluate(args):
    """Evaluación final de `best.pth` en val y test con el pipeline completo:
    detección axial (cajas de la cabeza y ajustadas a la segmentación), segmentación de las tres máscaras (Dice, IoU,
    HD95 y ASSD por hueso y su macro, y Dice/IoU de la pelvis), clasificación de cada instancia (hueso) sobre los
    fragmentos reales y de extremo a extremo, distancia de separación en el espacio original del CT (predicha frente
    a la real) y latencia en GPU y CPU."""
    from src.models.classification import classification_prediction
    det.seed_everything()
    device = det.get_device(args.device)
    run_dir = config.DETECTION_DIR / args.run
    out_dir = run_dir / "eval"
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(run_dir / "best.pth")
    extra = [k for k in ck["model"] if k.startswith("inst_head.")]
    if extra:
        print(f"Se ignora la cabeza de instancia del checkpoint ({len(extra)} tensores)", flush=True)
    model = load_model(ck, device)
    margins = list(config.M3D_REFINE_MARGINS)
    summary = {"run": args.run, "epoch": int(ck["epoch"]) + 1, "device": str(device), "splits": {}}
    if model.mask_head:
        summary["fusion_mode"] = config.M3D_FUSION_MODE
    prev = out_dir / "metrics.json"
    prev_latency = json.loads(prev.read_text()).get("latency") if args.keep_latency and prev.exists() else None
    show = set(case_ids("test")[:config.M3D_EVAL_EXAMPLES])

    for split in config.M3D_EVAL_SPLITS:
        ids = case_ids(split)
        dets = {v: {"gt": [], "raw": [], **{f"m{k}": [] for k in margins}} for v in config.M3D_DET_VIEWS}
        bone_rows, case_rows, dist_rows, cls_gt = [], [], [], {"y": [], "prob": []}
        dist_mask_rows, roi_rows, roi_gt = [], [], {"y": [], "prob": []}          # cabeza de máscara
        dist_fus_rows = []                                                        # fusión de las dos salidas
        img_paths = {q.stem: q for d in config.IMAGE_DIRS for q in d.glob("*.mha")}
        for cid in ids:
            img, msk, _ = VolumeDataset([cid])[0]
            x, m = to_gpu(img[None], msk[None], device)
            _sync(device); t0 = time.time()
            with det.autocast(device):
                feats = model(x)
            bone_t = segmentation_prediction(feats["sem_logits"].float())[0] > 0          # hueso (cabeza semántica)
            # Detección por vista
            for v in config.M3D_DET_VIEWS:
                sl = torch.arange(0, N, config.M3D_EVAL_STRIDE, device=device)
                with det.autocast(device):
                    out = model.detect(feats, v, sl)
                preds = postprocess_objectness(out)
                mv = VIEW_LUT_T[v].to(device)[take(m[None], v, sl)[:, 0]]
                gt, valid = det.region_boxes(mv, len(CLASSES[v]))
                bone_v = take(bone_t[None], v, sl)[:, 0].cpu().numpy()
                for j, p in enumerate(preds):
                    dets[v]["gt"].append(gt[j][valid[j]].cpu())
                    dets[v]["raw"].append(p)
                    for k in margins:
                        dets[v][f"m{k}"].append({"boxes": refine_boxes(p["boxes"], bone_v[j], k), "scores": p["scores"]})
            # Pipeline de instancias (detección axial -> regiones 3D -> hueso dentro de cada región -> fragmentos)
            with det.autocast(device):
                d_ax = model.detect(feats, "axial", torch.arange(0, N, device=device))
            preds_ax = postprocess_objectness(d_ax)
            regions = group_boxes_across_slices(preds_ax, torch.arange(0, N), iou_threshold=config.M3D_TRACK_IOU)
            with det.autocast(device):
                res = model.classify_pred_instances(feats, regions, min_voxels=config.M3D_MIN_INSTANCE_VOXELS)
                cg = model.classify_gt_instances(feats, m) if not model.mask_head else {"logits": torch.zeros(0)}
            mask_case = {}
            if model.mask_head:
                mask_case, mask_lab, bone_hr = mask_head_eval(model, feats, m, preds_ax, cid, split, img_paths[cid], device,
                                                              roi_rows, roi_gt, dist_mask_rows)
            _sync(device); t_case = time.time() - t0
            inst = res["instances"][0].cpu().numpy() if res["instances"].ndim == 4 else res["instances"].cpu().numpy()
            pred_cls, pred_prob = classification_prediction(res["logits"].float())
            ids_meta = [md["instance_id"] for md in res["metadata"]]
            cls_of = {i: int(c) for i, c in zip(ids_meta, pred_cls.cpu().tolist())}
            prob_of = {i: p for i, p in zip(ids_meta, pred_prob.cpu().numpy())}
            # Clasificación sobre los fragmentos reales (lo que mide el entrenamiento)
            if cg["logits"].shape[0]:
                cls_gt["y"] += cg["targets"].cpu().tolist()
                cls_gt["prob"] += torch.softmax(cg["logits"].float(), 1).cpu().numpy().tolist()
            # Segmentación: las tres máscaras de la salida final (dentro de las regiones detectadas)
            tgt = bone_targets(m.cpu())[0].numpy()
            hm = head_metrics(inst, tgt, SEM_CLASSES)
            dice_pelvis, iou_pelvis = _binary_scores(torch.from_numpy(inst > 0), torch.from_numpy(tgt > 0))
            # Clasificación de extremo a extremo: cada instancia (hueso) contra el hueso real mayoritario en ella
            for i in ids_meta:
                vals = tgt[inst == i]
                vals = vals[vals > 0]
                if len(vals):
                    bone_rows.append({"case_id": cid, "split": split, "instance": int(i),
                                      "class_true": int(np.bincount(vals).argmax()) - 1, "class_pred": cls_of[i],
                                      **{f"prob_{k}": float(prob_of[i][k]) for k in range(3)}})
            # Distancia de separación en el espacio original
            d_rows, d_counts = separation_eval(cid, inst, img_paths[cid], split)
            dist_rows += d_rows
            # Fusión de la semántica y la cabeza de máscara (postproceso), a doble resolución
            fus_case = {}
            if model.mask_head:
                fus = fuse_heads(inst.astype(np.uint8), mask_lab)
                v_ = config.VOXEL_MM
                fm = head_metrics(fus, bone_hr, SEM_CLASSES, spacing=(v_, v_ / config.M3D_HIRES_FACTOR, v_ / config.M3D_HIRES_FACTOR))
                fpel, _ = _binary_scores(torch.from_numpy(fus > 0), torch.from_numpy(bone_hr > 0))
                dist_fus_rows += separation_eval(cid, fus, img_paths[cid], split, hires=True)[0]
                fus_case = {"fusion_pelvis_dice": fpel, **{f"fusion_{c}_{k}": val for c, cm in fm.items() for k, val in cm.items()}}
                del fus, mask_lab, bone_hr
            case_rows.append({"case_id": cid, "split": split, "dice_pelvis": dice_pelvis, "iou_pelvis": iou_pelvis,
                              **{f"huesos_{c}_{k}": v for c, cm in hm.items() for k, v in cm.items()}, **mask_case, **fus_case,
                              **d_counts, "n_regions": len(regions), "time_s": t_case})
            mac = hm["promedio"]
            print(f"[{split}] {cid}: Dice pelvis {dice_pelvis:.3f} | huesos (macro) Dice {mac['dice']:.3f} IoU "
                  f"{mac['iou']:.3f} HD95 {mac['hd95']:.1f} mm ASSD {mac['assd']:.1f} mm | "
                  + " ".join(f"{c} {hm[c]['dice']:.3f}" for c in SEM_CLASSES if c in hm)
                  + (f" | máscara Dice {mask_case['mascara_promedio_dice']:.3f}" if mask_case else "")
                  + (f" | fusión Dice {fus_case['fusion_promedio_dice']:.3f}" if fus_case else "")
                  + f" | {len(d_rows)} fragmentos conminutos (real) | {t_case:.1f} s", flush=True)
            if split == "test" and cid in show:
                np.savez_compressed(out_dir / "examples" / f"{cid}.npz", inst=inst.astype(np.uint8),
                                    bone=bone_t.cpu().numpy(), sem=res["semantic_mask"].numpy().astype(np.uint8),
                                    inst_class=np.array([[i, cls_of.get(i, -1)] for i in range(1, int(inst.max()) + 1)]),
                                    **{f"boxes_{v}": np.array([[j, *b] for j, p in enumerate(dets[v]["raw"][-N:])
                                                               for b in p["boxes"].tolist()]) for v in config.M3D_DET_VIEWS},
                                    **{f"scores_{v}": np.array([s for p in dets[v]["raw"][-N:] for s in p["scores"].tolist()])
                                       for v in config.M3D_DET_VIEWS})
            del feats, res
        # Detección: cabeza y cajas ajustadas, por vista y en conjunto
        det_res = {}
        for name in ["raw"] + [f"m{k}" for k in margins]:
            det_res[name] = {v: detection_eval(dets[v][name], dets[v]["gt"]) for v in config.M3D_DET_VIEWS}
            det_res[name]["all"] = detection_eval(sum((dets[v][name] for v in config.M3D_DET_VIEWS), []),
                                                  sum((dets[v]["gt"] for v in config.M3D_DET_VIEWS), []))
        bones = pd.DataFrame(bone_rows); cases = pd.DataFrame(case_rows); dist = pd.DataFrame(dist_rows)
        bones.to_csv(out_dir / f"instances_{split}.csv", index=False)
        cases.to_csv(out_dir / f"cases_{split}.csv", index=False)
        dist.to_csv(out_dir / f"distances_{split}.csv", index=False)
        ok = dist.dropna(subset=["dist_pred_mm"]) if len(dist) else dist
        mask_summary = {}
        if model.mask_head:
            dm, rr = pd.DataFrame(dist_mask_rows), pd.DataFrame(roi_rows)
            dm.to_csv(out_dir / f"distances_mask_{split}.csv", index=False)
            df_ = pd.DataFrame(dist_fus_rows)
            df_.to_csv(out_dir / f"distances_fusion_{split}.csv", index=False)
            rr.to_csv(out_dir / f"rois_{split}.csv", index=False)
            mt = rr[rr.class_true >= 0] if len(rr) else rr
            mask_summary = {
                "segmentation_bones_mask": {c: {k: float(np.nanmean(cases[f"mascara_{c}_{k}"])) for k in ("dice", "iou", "hd95", "assd")}
                                            for c in [*SEM_CLASSES, "promedio"] if f"mascara_{c}_dice" in cases},
                "segmentation_pelvis_mask": {"dice": float(cases.mascara_pelvis_dice.mean())},
                "separation_distance_mask": _dist_summary(dm),
                "segmentation_bones_fusion": {c: {k: float(np.nanmean(cases[f"fusion_{c}_{k}"])) for k in ("dice", "iou", "hd95", "assd")}
                                              for c in [*SEM_CLASSES, "promedio"] if f"fusion_{c}_dice" in cases},
                "segmentation_pelvis_fusion": {"dice": float(cases.fusion_pelvis_dice.mean())},
                "separation_distance_fusion": _dist_summary(df_),
                "classification_roi_gt": _macro_f1_auc(np.array(roi_gt["y"]), np.array(roi_gt["prob"])),
                "classification_roi_end_to_end": _macro_f1_auc(mt.class_true.to_numpy(),
                                                               mt[["prob_0", "prob_1", "prob_2"]].to_numpy()) if len(mt) else {},
                "confusion_roi_end_to_end": pd.crosstab(mt.class_true, mt.class_pred).reindex(index=range(3), columns=range(3),
                                                                                          fill_value=0).values.tolist()
                if len(mt) else [],
                "rois_unmatched_frac": float((rr.class_true < 0).mean()) if len(rr) else float("nan"),
            }
        summary["splits"][split] = {
            "n_cases": len(ids),
            "detection": {name: {v: {k: r[k] for k in ("mAP50", "mAP75", "mAP50_95", "mIoU", "n_gt", "n_pred")}
                                 for v, r in res_.items()} for name, res_ in det_res.items()},
            "detection_curves": {v: det_res["raw"][v]["curves"] for v in [*config.M3D_DET_VIEWS, "all"]},
            "detection_ap_by_iou": {name: det_res[name]["all"]["ap_by_iou"] for name in det_res},
            "detection_gt_best_iou": {v: det_res["raw"][v]["gt_best_iou"] for v in config.M3D_DET_VIEWS},
            # Tres máscaras: por hueso y macro (promedio sobre los CT donde el hueso está presente)
            "segmentation_bones": {c: {k: float(np.nanmean(cases[f"huesos_{c}_{k}"])) for k in ("dice", "iou", "hd95", "assd")}
                                   for c in [*SEM_CLASSES, "promedio"] if f"huesos_{c}_dice" in cases},
            "segmentation_pelvis": {"dice": float(cases.dice_pelvis.mean()), "iou": float(cases.iou_pelvis.mean())},
            "separation_distance": {                     # un fragmento conminuto real por fila (módulo de P4)
                "fragments_gt": int(len(dist)),
                "matched": float(len(ok) / max(len(dist), 1)),
                "merged_with_main": float((ok.dist_pred_mm == 0).mean()) if len(ok) else float("nan"),
                "mae_mm": float((ok.dist_pred_mm - ok.dist_gt_mm).abs().mean()) if len(ok) else float("nan"),
                "mean_gt_mm": float(dist.dist_gt_mm.mean()) if len(dist) else float("nan"),
                "gt_separated_gt_2mm": int((dist.dist_gt_mm > 2).sum()) if len(dist) else 0,
                "mae_mm_gt_2mm": float((ok[ok.dist_gt_mm > 2].dist_pred_mm - ok[ok.dist_gt_mm > 2].dist_gt_mm).abs().mean())
                if len(ok[ok.dist_gt_mm > 2]) else float("nan"),
            },
            **mask_summary,
            "classification_gt_instances": _macro_f1_auc(np.array(cls_gt["y"]), np.array(cls_gt["prob"])),
            "classification_end_to_end": _macro_f1_auc(bones.class_true.to_numpy(),
                                                       bones[["prob_0", "prob_1", "prob_2"]].to_numpy())
            if len(bones) else {},
        }
    # Margen del ajuste: el de mayor mAP50-95 en val, aplicado tal cual en test
    if "val" in summary["splits"]:
        best_m = max(margins, key=lambda k: summary["splits"]["val"]["detection"][f"m{k}"]["all"]["mAP50_95"])
        summary["refine_margin_from_val"] = best_m
    # Latencia del pipeline completo (red + detección axial + NMS + regiones + instancias + clasificación)
    img, msk, _ = VolumeDataset(case_ids("test")[:1])[0]
    img_hr = load_hires(case_ids("test")[0])[0] if model.mask_head else None
    lat = {}
    for dev_name, runs in (() if prev_latency else ((str(device), config.M3D_LATENCY_RUNS_GPU), ("cpu", config.M3D_LATENCY_RUNS_CPU))):
        d = torch.device(dev_name)
        mdl = model.to(d)
        x, _ = to_gpu(img[None], msk[None], d)
        times = []
        for r in range(runs + 1):
            _sync(d); t0 = time.time()
            infer_instances_axial(mdl, x, img_hr=img_hr)
            _sync(d)
            if r:
                times.append(time.time() - t0)
        lat[dev_name] = {"s_per_ct": float(np.median(times)), "ms_per_slice": float(np.median(times) / N * 1000),
                         "runs": runs}
        print(f"latencia {dev_name}: {lat[dev_name]['s_per_ct']:.2f} s por CT "
              f"({lat[dev_name]['ms_per_slice']:.1f} ms por corte axial)", flush=True)
    model.to(device)
    summary["latency"] = prev_latency or lat
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Listo: {out_dir / 'metrics.json'}")



# ======================================================================================
# Inferencia sobre un CT cualquiera
# ======================================================================================
def read_ct(path):
    """Lee un CT 3D: archivo `.mha` / `.nii` / `.nii.gz` / `.nrrd` o carpeta con una serie DICOM."""
    from src.dataset import preprocessing as pre
    path = Path(path)
    if path.is_dir():
        reader = pre.sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(path))
        if not files:
            raise ValueError(f"{path}: la carpeta no tiene una serie DICOM")
        reader.SetFileNames(files)
        return reader.Execute()
    return pre.load_volume(path)


FUSION_MODES = ("huesos", "union_sacro", "relleno")


def fuse_heads(sem: np.ndarray, mask: np.ndarray, mode: str = config.M3D_FUSION_MODE) -> np.ndarray:
    """Postproceso que une la cabeza semántica y la cabeza de máscara. `sem`: etiquetas del cubo (Z, Y, X);
    `mask`: etiquetas de la cabeza de máscara (Z, f·Y, f·X). Devuelve etiquetas en la rejilla de `mask`
    (0 fondo, 1 sacro, 2 coxal izq., 3 coxal der.).

    - "huesos":      coxales de la cabeza de máscara; sacro de la semántica (donde no hay coxal).
    - "union_sacro": lo anterior y, además, el sacro que marca la cabeza de máscara donde aún es fondo.
    - "relleno":     lo anterior y, en los cortes donde la cabeza de máscara no marca nada de un coxal (no hubo caja),
                     ese coxal sale de la semántica.
    """
    f = mask.shape[-1] // sem.shape[-1]
    up = sem.repeat(f, 1).repeat(f, 2) if f > 1 else sem
    out = np.where(mask >= 2, mask, 0).astype(np.uint8)
    out[(up == 1) & (out == 0)] = 1
    if mode in ("union_sacro", "relleno"):
        out[(mask == 1) & (out == 0)] = 1
    if mode == "relleno":
        for b in (2, 3):
            vacio = ~(mask == b).any((1, 2))
            out[vacio] = np.where((up[vacio] == b) & (out[vacio] == 0), b, out[vacio])
    return out


@torch.no_grad()
def infer_ct(model, ct, device) -> dict:
    """Pipeline completo sobre un CT (`sitk.Image`) con cualquier número de cortes, espaciado y orientación.

    Preprocesa igual que el caché (LPS, cubo de 400 mm, ventana HU; y el cubo a doble resolución para la cabeza de
    máscara), corre `infer_instances_axial` y devuelve todo en la rejilla ORIGINAL del CT:
      semantic, mask   uint8 con la forma del CT (0 fondo, 1 sacro, 2 coxal izq., 3 coxal der.): cabeza semántica y
                       cabeza de máscara (None si el modelo no la tiene)
      fusion           las dos unidas con `fuse_heads` (None sin cabeza de máscara)
      boxes            una fila por caja y corte original: corte, x0, y0, x1, y1 (píxeles del corte), puntaje, hueso
      slices_outside   cortes originales que quedan fuera del cubo (sin predicción)
      time_s           tiempo del preprocesamiento y de la red
    """
    from src.dataset import preprocessing as pre
    t0 = time.time()
    img = pre.reorient(ct)
    geom = pre.cube_geometry(img)
    cube = np.round(pre.window_hu(pre.sitk.GetArrayFromImage(pre.to_cube(img, geom))) * 255).astype(np.uint8)
    x = torch.from_numpy(cube).to(device).float().div_(255)[None, None]
    img_hr, geom_hr = None, None
    if model.mask_head:
        geom_hr = pre.hires_geometry(geom, config.M3D_HIRES_FACTOR)
        img_hr = torch.from_numpy(np.round(pre.window_hu(pre.sitk.GetArrayFromImage(pre.to_cube(img, geom_hr))) * 255)
                                  .astype(np.uint8))
    t_pre = time.time() - t0
    _sync(device); t0 = time.time()
    res = infer_instances_axial(model, x, img_hr=img_hr)
    _sync(device); t_net = time.time() - t0
    inst = res["instances"][0].cpu().numpy().astype(np.uint8)
    out = {"semantic": pre.cube_to_original(inst, geom, ct),
           "mask": pre.cube_to_original(res["mask_labels"], geom_hr, ct) if model.mask_head else None,
           "fusion": pre.cube_to_original(fuse_heads(inst, res["mask_labels"]), geom_hr, ct) if model.mask_head else None}

    # Cajas: del cubo a cada corte original. El corte original k toma las cajas del corte del cubo más cercano.
    origin, sp = np.array(geom["origin"]), np.array(geom["spacing"])
    d_cube = np.array(geom["direction"]).reshape(3, 3)
    d_ct = np.array(ct.GetDirection()).reshape(3, 3)
    if abs(float(d_ct[:, 2] @ d_cube[:, 2])) < 0.99:
        print("AVISO: los cortes del archivo no son axiales; las cajas se dan sobre el eje más cercano al axial.", flush=True)
    if model.mask_head:
        cajas = [(r["slice"], r["box"], r["score"], r["class"]) for r in res["rois"]]
    else:                                            # sin rama de clase: el hueso mayoritario de la semántica en la caja
        cajas = []
        for j, p in enumerate(res["detections"]):
            for b, sc in zip(p["boxes"].tolist(), p["scores"].tolist()):
                if sc >= config.M3D_MASK_SCORE_THR:
                    x0, y0, x1, y1 = [int(round(t)) for t in b]
                    vals = inst[j, max(y0, 0):y1, max(x0, 0):x1]
                    vals = vals[vals > 0]
                    cajas.append((j, b, sc, int(np.bincount(vals).argmax()) - 1 if len(vals) else -1))
    por_corte = {}
    for c in cajas:
        por_corte.setdefault(int(c[0]), []).append(c)
    n = ct.GetSize()[2]
    rows, outside = [], 0
    for k in range(n):
        pk = np.array(ct.TransformIndexToPhysicalPoint((0, 0, k)))
        zc = int(round(float(((pk - origin) @ d_cube)[2] / sp[2])))
        if not 0 <= zc < N:
            outside += 1
            continue
        for _, (x0, y0, x1, y1), sc, cls in por_corte.get(zc, []):
            # esquinas de la caja (bordes de píxel del cubo -> índice continuo -> mm -> índice del CT original)
            pts = [origin + d_cube @ (np.array([cx - 0.5, cy - 0.5, zc]) * sp) for cx in (x0, x1) for cy in (y0, y1)]
            ij = np.array([ct.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in q))[:2] for q in pts]) + 0.5
            (a0, b0), (a1, b1) = ij.min(0), ij.max(0)
            rows.append({"corte": k, "x0": a0, "y0": b0, "x1": a1, "y1": b1, "puntaje": sc,
                         "hueso": config.REGION_NAMES[cls] if cls >= 0 else "sin_clase"})
    out.update(boxes=pd.DataFrame(rows, columns=["corte", "x0", "y0", "x1", "y1", "puntaje", "hueso"]),
               slices_outside=outside, time_s={"preprocesamiento": t_pre, "red": t_net})
    return out


def infer(args):
    """`model3d infer <ct>`: cajas, máscara de cada hueso (fragmento = hueso) y pertenencia para cada corte de un CT
    con cualquier número de cortes. Escribe en `--out` (por defecto `results/inferencia/<nombre>/`):
      mascara_semantica.nii.gz, mascara_cabeza.nii.gz   etiquetas en la rejilla del CT (se abren encima del CT)
      mascara_fusion.nii.gz                             las dos unidas (`fuse_heads`): la salida recomendada
      cajas.csv                                         una fila por caja y corte
      cortes.csv                                        por corte: cajas y píxeles de cada hueso en cada máscara
      resumen.json                                      volumen por hueso, fragmentos separados y su distancia, tiempos
    """
    from src.dataset import preprocessing as pre
    device = det.get_device(args.device)
    ck = load_checkpoint(config.DETECTION_DIR / args.run / "best.pth")
    model = load_model(ck, device)
    src = Path(args.ct)
    name = src.name.removesuffix(".gz").rsplit(".", 1)[0] if src.is_file() else src.name
    out_dir = Path(args.out) if args.out else config.RESULTS_DIR / "inferencia" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ct = read_ct(src)
    nx, ny, nz = ct.GetSize()
    print(f"CT {src.name}: {nz} cortes de {nx} × {ny}, espaciado {tuple(round(v, 3) for v in ct.GetSpacing())} mm | "
          f"modelo {args.run} (época {int(ck['epoch']) + 1}) en {device.type}", flush=True)
    res = infer_ct(model, ct, device)

    salidas = {"semantica": res["semantic"]} | ({"cabeza": res["mask"], "fusion": res["fusion"]} if res["mask"] is not None else {})
    sp = tuple(float(v) for v in ct.GetSpacing())[::-1]                    # (k, j, i) del array, en mm
    min_vox = int(np.ceil(config.M3D_MIN_FRAGMENT_MM3 / np.prod(sp)))
    cortes = pd.DataFrame({"corte": np.arange(nz)})
    cortes["cajas"] = res["boxes"].groupby("corte").size().reindex(cortes.corte, fill_value=0).values
    resumen = {"ct": str(src), "run": args.run, "epoca": int(ck["epoch"]) + 1, "dispositivo": device.type,
               "cortes": nz, "tamano_xy": [nx, ny], "espaciado_mm": list(ct.GetSpacing()),
               "cortes_fuera_del_cubo": res["slices_outside"], "cajas": int(len(res["boxes"])),
               "tiempo_s": res["time_s"], "salidas": {}}
    for nombre, lab in salidas.items():
        m = pre.sitk.GetImageFromArray(lab)
        m.CopyInformation(ct)
        pre.sitk.WriteImage(m, str(out_dir / f"mascara_{nombre}.nii.gz"), True)
        resumen["salidas"][nombre] = {}
        for b, region in enumerate(config.REGION_NAMES, start=1):
            cortes[f"{nombre}_{region}_px"] = (lab == b).reshape(nz, -1).sum(1)
            _, _, frs = separation_distances(lab == b, sp, min_vox)       # espacio original
            resumen["salidas"][nombre][region] = {
                "volumen_cm3": float((lab == b).sum() * np.prod(sp) / 1000),
                "cortes_con_hueso": int((cortes[f"{nombre}_{region}_px"] > 0).sum()),
                "fragmentos_separados": len(frs),
                "distancia_al_principal_mm": [round(float(r["distance_mm"]), 2) for r in frs]}
    res["boxes"].round(2).to_csv(out_dir / "cajas.csv", index=False)
    cortes.to_csv(out_dir / "cortes.csv", index=False)
    (out_dir / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    for nombre, d in resumen["salidas"].items():
        print(f"[{nombre}] " + " | ".join(
            f"{config.REGION_LABELS[r]}: {v['volumen_cm3']:.0f} cm³ en {v['cortes_con_hueso']} cortes, "
            f"{v['fragmentos_separados']} fragmentos separados" for r, v in d.items()), flush=True)
    print(f"{len(res['boxes'])} cajas en {int((cortes.cajas > 0).sum())} de {nz} cortes | fuera del cubo de "
          f"{config.CUBE_MM:.0f} mm: {res['slices_outside']} cortes | preprocesamiento {res['time_s']['preprocesamiento']:.1f} s, "
          f"red {res['time_s']['red']:.1f} s\nListo: {out_dir}", flush=True)


def bench(args):
    """Memoria máxima y tiempo de un paso de entrenamiento (ida y vuelta) con un volumen aleatorio."""
    device = det.get_device(args.device)
    torch.backends.cudnn.benchmark = False
    model = Fundidora3DDetector(_attention(args)).to(device)
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
        aug = {}
        x, m = to_gpu(img, msk, device, augment=True, params=aug)
        hr = None
        if model.mask_head:
            F_ = config.M3D_HIRES_FACTOR
            hr = (img[0].repeat_interleave(F_, 1).repeat_interleave(F_, 2), msk[0].repeat_interleave(F_, 1)
                  .repeat_interleave(F_, 2), aug)
        loss, _ = step_loss(model, x, m, gen, hr=hr)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        print(f"paso {i}: {time.time() - t0:.2f} s  memoria máx {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB "
              f"(reservada {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB)", flush=True)


RUNS = {"cbam": config.M3D_RUN_CBAM, "se": config.M3D_RUN_SE, "none": config.M3D_RUN_NO_CBAM}


def _attention(args) -> str:
    return "none" if args.no_cbam else args.attention


def main() -> None:
    p = argparse.ArgumentParser(description="Fundidora 3D con detección 2D por corte.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default="auto")
        sp.add_argument("--attention", default=config.M3D_ATTENTION, choices=("cbam", "se", "none"),
                        help="cbam: CBAM + γ; se: solo atención de canal (squeeze-excitation) + γ; none: sin atención")
        sp.add_argument("--no-cbam", action="store_true", help="igual que --attention none")

    t = sub.add_parser("train"); common(t)
    t.add_argument("--epochs", type=int, default=config.M3D_EPOCHS)
    t.add_argument("--workers", type=int, default=config.M3D_WORKERS)
    t.add_argument("--name", default=None)
    t.add_argument("--resume", action="store_true")
    o = sub.add_parser("overfit"); common(o)
    o.add_argument("--iters", type=int, default=config.M3D_OVERFIT_ITERS)
    c = sub.add_parser("calibrate"); common(c)
    e = sub.add_parser("evaluate"); common(e)
    e.add_argument("--run", default=config.M3D_RUN_CBAM)
    e.add_argument("--keep-latency", action="store_true",
                   help="no mide la latencia: conserva la del metrics.json anterior (p. ej. la medida en otra máquina)")
    b = sub.add_parser("bench"); common(b)
    b.add_argument("--iters", type=int, default=5)
    i = sub.add_parser("infer", help="cajas, máscaras y pertenencia de un CT con cualquier número de cortes")
    i.add_argument("ct", help="archivo .mha / .nii / .nii.gz o carpeta con una serie DICOM")
    i.add_argument("--run", default=config.M3D_RUN_CBAM)
    i.add_argument("--out", default=None, help="carpeta de salida (por defecto results/inferencia/<nombre>/)")
    i.add_argument("--device", default="auto")
    h = sub.add_parser("cache-hires", help="caché del CT a doble resolución en el plano (cabeza de máscara)")
    h.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.cmd == "cache-hires":
        return build_hires_cache(args.overwrite)
    {"train": train, "overfit": overfit, "calibrate": calibrate, "evaluate": evaluate, "bench": bench, "infer": infer}[args.cmd](args)


if __name__ == "__main__":
    main()