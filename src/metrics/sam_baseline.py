"""Baseline zero-shot de SAM (Segment Anything, Meta) guiado por las cajas del detector.

Cada corte del caché (uint8 0-255) se pasa a RGB replicando canales y las cajas PREDICHAS por el
detector (post-NMS, una como máximo por región) se usan como prompt de caja: SAM nunca recibe la
GT. La máscara que devuelve para cada caja se compara con la GT de esa región en el corte (el mapa
de clases binarizado) con Dice e IoU. SAM es solo un baseline de comparación: no se integra al
pipeline de detección ni se usa para generar etiquetas.

Funciones puras; la orquestación (detector -> cajas -> SAM -> tabla por corte y región) está en
scripts/run_sam_baseline.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def load_sam(checkpoint: str, device: str = "auto") -> "SamPredictor":
    """Carga SAM ViT-B desde `checkpoint` y lo deja en modo evaluación.

    `device` "auto" -> cuda si está disponible, si no cpu. Devuelve el `SamPredictor` listo para
    recibir imágenes con `set_image`.
    """
    from segment_anything import SamPredictor, sam_model_registry

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device=device).eval()
    return SamPredictor(sam)


def slice_to_rgb(img2d: np.ndarray) -> np.ndarray:
    """Corte uint8 del caché (H, W) en 0-255 -> RGB (H, W, 3) uint8 replicando el canal de intensidad."""
    rgb = np.repeat(np.asarray(img2d, dtype=np.uint8)[..., None], 3, axis=2)
    return np.ascontiguousarray(rgb)


def masks_for_boxes(predictor: "SamPredictor", img_rgb: np.ndarray, boxes_xyxy: np.ndarray,
                    tiempos: dict[str, float] | None = None) -> np.ndarray:
    """Máscaras binarias (N, H, W) de SAM para las cajas (N, 4) en píxeles (x0, y0, x1, y1).

    `set_image` corre UNA vez por corte (codificación de la imagen, la parte costosa) y las N cajas
    se procesan en un solo lote con `predict_torch` tras `apply_boxes_torch`, con
    `multimask_output=False` (una máscara por caja). Si `tiempos` no es None, deja en él los
    segundos de cada etapa ("set_image" y "predict") de esta llamada.
    """
    h, w = img_rgb.shape[:2]
    if len(boxes_xyxy) == 0:
        return np.zeros((0, h, w), dtype=bool)
    t0 = time.perf_counter()
    predictor.set_image(img_rgb)
    t1 = time.perf_counter()
    boxes = predictor.transform.apply_boxes_torch(
        torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32), device=predictor.device), (h, w))
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=boxes,
                                          multimask_output=False)
    if tiempos is not None:
        tiempos["set_image"] = t1 - t0
        tiempos["predict"] = time.perf_counter() - t1
    return masks[:, 0].cpu().numpy().astype(bool)


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    """Dice (2|P∩G| / (|P|+|G|)) e IoU (|P∩G| / |P∪G|) entre máscaras booleanas del mismo tamaño.

    Convención: GT vacía y predicción vacía -> (None, None); esa fila no aporta información sobre
    la región y no se cuenta en los resúmenes. En cualquier otro caso una intersección vacía da
    0.0, incluido el falso positivo del detector (caja en un corte sin GT de esa región).
    """
    n_p, n_g = int(pred.sum()), int(gt.sum())
    if n_p == 0 and n_g == 0:
        return None, None
    inter = int(np.logical_and(pred, gt).sum())
    return float(2.0 * inter / (n_p + n_g)), float(inter / (n_p + n_g - inter))
