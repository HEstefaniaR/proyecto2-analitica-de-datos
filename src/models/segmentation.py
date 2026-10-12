"""Segmentación 3D por hueso (D-81, asesoría del 2026-10-08).

Una sola cabeza sobre P0: 0 = fondo, 1 = sacro, 2 = coxal izquierdo, 3 = coxal derecho.
Cada hueso es una instancia con TODOS sus fragmentos (los pequeños y el grande): el dataset tiene tres labels por
hueso y el profesor pidió tres máscaras, sin separar los fragmentos de su hueso.

Inferencia en dos etapas: etapa 1, regiones óseas detectadas (cajas axiales agrupadas entre cortes); etapa 2, dentro
de esas regiones, la máscara de cada hueso. La distancia de separación de cada fragmento se mide después, sobre la
máscara de su hueso (componentes conexos, `separation_distances`).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label

from src import config


class SegmentationHead3D(nn.Module):
    """Conv 1×1×1 sobre P0 -> logits de fondo, sacro, coxal izquierdo y coxal derecho."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        return self.classifier(p0)


def bone_targets(fragment_mask: torch.Tensor) -> torch.Tensor:
    """Códigos de fragmento (1-10 sacro, 11-20 coxal izq., 21-30 coxal der.) -> hueso de cada vóxel (0-3)."""
    m = fragment_mask.long()
    if m.ndim == 3:
        m = m.unsqueeze(0)
    return torch.where(m > 0, (m - 1) // 10 + 1, torch.zeros_like(m))


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice suave multiclase."""

    probs = torch.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target.long(), logits.shape[1])
    onehot = onehot.permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probs * onehot).sum(dims)
    denominator = probs.sum(dims) + onehot.sum(dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    fragment_mask: torch.Tensor,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    """`ce_weight` · CE + `dice_weight` · Dice multiclase (fondo, sacro, coxal izq., coxal der.)."""
    target = bone_targets(fragment_mask).to(logits.device)
    ce = F.cross_entropy(logits.float(), target)
    dice = multiclass_dice_loss(logits, target)
    return ce_weight * ce + dice_weight * dice, {"seg_ce": float(ce.detach()), "seg_dice_loss": float(dice.detach())}


@torch.no_grad()
def segmentation_prediction(logits: torch.Tensor) -> torch.Tensor:
    """Clase de cada vóxel (argmax)."""
    return logits.float().argmax(dim=1)


@torch.no_grad()
def connected_components_3d(
    binary_mask: torch.Tensor,
    min_voxels: int = 1,
) -> torch.Tensor:
    """Genera IDs de instancia con componentes conexos."""

    if binary_mask.ndim == 3:
        binary_mask = binary_mask.unsqueeze(0)

    device = binary_mask.device
    result = []
    # D-71: vecindad configurable (6 = caras, 18 = + aristas, 26 = + esquinas)
    structure = generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[config.M3D_CONNECTIVITY])

    for mask in binary_mask:
        mask_np = mask.detach().cpu().numpy().astype(bool)
        labels_np, num_instances = label(mask_np, structure=structure)

        if min_voxels > 1:
            # Quita las islas pequeñas y renumera 1..K en el mismo orden (conteo vectorizado: un solo recorrido)
            sizes = np.bincount(labels_np.ravel())
            keep = sizes >= min_voxels
            keep[0] = False
            lut = np.zeros(len(sizes), dtype=np.int32)
            lut[keep] = np.arange(1, keep.sum() + 1)
            labels_np = lut[labels_np]

        result.append(torch.from_numpy(labels_np.astype(np.int64)))

    return torch.stack(result).to(device)


@torch.no_grad()
def boxes_to_region_mask_3d(
    boxes_by_slice: dict,
    volume_shape,
    device,
):
    """Convierte cajas axiales en una región 3D."""

    depth, height, width = volume_shape
    region_mask = torch.zeros(depth, height, width, dtype=torch.bool, device=device)

    for slice_id, boxes in boxes_by_slice.items():
        if boxes is None or len(boxes) == 0:
            continue

        z = int(slice_id)
        if z < 0 or z >= depth:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.round().long()

            x1 = int(torch.clamp(x1, 0, width - 1))
            x2 = int(torch.clamp(x2, 1, width))
            y1 = int(torch.clamp(y1, 0, height - 1))
            y2 = int(torch.clamp(y2, 1, height))

            if x2 <= x1 or y2 <= y1:
                continue

            region_mask[z, y1:y2, x1:x2] = True

    return region_mask


def bone_instances(sem: np.ndarray, region: np.ndarray | None = None):
    """Etapa 2: una instancia por hueso (id = hueso, 1-3) con todos sus fragmentos, dentro de las regiones
    detectadas (`region`, etapa 1) si se dan. Devuelve (instancias uint8, [{"instance_id", "bone" 0-2}])."""
    out = sem.astype(np.uint8)
    if region is not None:
        out = np.where(region, out, 0).astype(np.uint8)
    return out, [{"instance_id": b, "bone": b - 1} for b in (1, 2, 3) if (out == b).any()]


def separation_distances(bone: np.ndarray, spacing, min_voxels: int = config.M3D_MIN_INSTANCE_VOXELS):
    """Distancia de separación (rúbrica 3.3) dentro de la máscara de UN hueso: componentes conexos
    (`config.M3D_CONNECTIVITY`); el más grande es el fragmento principal; para cada uno de los demás, la distancia
    mínima borde a borde al principal en mm (`distance_transform_edt` con `spacing` = (z, y, x) en mm).
    Fragmentos que se tocan quedan en el mismo componente (distancia 0).
    Devuelve (etiquetas de componentes, id del principal, [{"comp", "voxels", "distance_mm"}])."""
    st = generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[config.M3D_CONNECTIVITY])
    lab = np.zeros(bone.shape, np.int32)
    if not bone.any():
        return lab, 0, []
    zz, yy, xx = [np.nonzero(bone.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    box = np.s_[zz[0]:zz[-1] + 1, yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1]
    sub, n = label(bone[box], structure=st)
    sizes = np.bincount(sub.ravel())
    sizes[0] = 0
    keep = np.nonzero(sizes >= min_voxels)[0]
    lut = np.zeros(len(sizes), np.int32)
    lut[keep] = keep
    sub = lut[sub]
    lab[box] = sub
    main = int(sizes.argmax())
    rows = []
    if len(keep) > 1:
        dist = distance_transform_edt(sub != main, sampling=spacing)
        for k in keep:
            if k != main:
                rows.append({"comp": int(k), "voxels": int(sizes[k]), "distance_mm": float(dist[sub == k].min())})
    return lab, main, rows


def _binary_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
):
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, fragment_mask: torch.Tensor):
    """Dice e IoU de la pelvis (hueso contra fondo) y Dice e IoU macro de los tres huesos (D-81)."""
    target = bone_targets(fragment_mask).to(logits.device)
    pred = segmentation_prediction(logits)
    dice_bone, iou_bone = _binary_scores(pred > 0, target > 0)
    per = {b: _binary_scores(pred == b, target == b) for b in (1, 2, 3) if (target == b).any()}
    return {"dice": dice_bone, "iou": iou_bone, "dice_sem": float(np.mean([d for d, _ in per.values()])),
            "iou_sem": float(np.mean([i for _, i in per.values()])),
            # Dice de cada hueso (NaN si el hueso no está en el CT): detecta el colapso de una clase (D-90)
            **{f"dice_{n}": per[b][0] if b in per else float("nan") for b, n in ((1, "sacro"), (2, "izq"), (3, "der"))}}


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing=config.VOXEL_MM) -> dict:
    """HD95 y ASSD (mm) entre dos máscaras bool, como en el challenge PENGWIN (Metrics Reloaded).

    Superficie = vóxeles de la máscara con algún vecino (6) fuera. HD95 = máximo de los percentiles 95 de las
    distancias dirigidas pred → real y real → pred; ASSD = promedio de todas las distancias de superficie en los dos
    sentidos. Si la predicción está vacía, como en el challenge: HD95 = diámetro y ASSD = radio de la esfera que
    envuelve la caja del objeto real. Si el real está vacío, NaN (no se promedia).
    `spacing`: un número (vóxel isótropo) o (z, y, x) en mm.
    """
    sp3 = np.broadcast_to(np.asarray(spacing, float), (3,))
    if not gt.any():
        return {"hd95": float("nan"), "assd": float("nan")}
    if not pred.any():
        zz, yy, xx = [np.nonzero(gt.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
        diam = float(np.linalg.norm(np.array([zz[-1] - zz[0] + 1, yy[-1] - yy[0] + 1, xx[-1] - xx[0] + 1]) * sp3))
        return {"hd95": diam, "assd": diam / 2}
    both = pred | gt
    zz, yy, xx = [np.nonzero(both.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    box = np.s_[max(zz[0] - 1, 0):zz[-1] + 2, max(yy[0] - 1, 0):yy[-1] + 2, max(xx[0] - 1, 0):xx[-1] + 2]
    p, g = pred[box], gt[box]
    sp = p & ~binary_erosion(p, border_value=0)
    sg = g & ~binary_erosion(g, border_value=0)
    d_pg = distance_transform_edt(~sg, sampling=sp3)[sp]       # de la superficie predicha a la real
    d_gp = distance_transform_edt(~sp, sampling=sp3)[sg]
    return {"hd95": float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))),
            "assd": float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))}


def head_metrics(pred: np.ndarray, target: np.ndarray, classes: dict, spacing=config.VOXEL_MM) -> dict:
    """Dice, IoU, HD95 y ASSD de una cabeza de segmentación, por clase (`classes` = {nombre: valor}) y su promedio
    sobre las clases presentes en el CT."""
    out = {}
    for name, c in classes.items():
        pc, gc = pred == c, target == c
        if not gc.any():
            continue
        dice, iou = _binary_scores(torch.from_numpy(pc), torch.from_numpy(gc))
        out[name] = {"dice": dice, "iou": iou, **surface_metrics(pc, gc, spacing)}
    out["promedio"] = {k: float(np.nanmean([v[k] for v in out.values()])) for k in ("dice", "iou", "hd95", "assd")}
    return out
