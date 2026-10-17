"""Inferencia del modelo 3D para el dashboard: instancias, distancias y cajas en el CT original.

Une el pipeline de un CT cualquiera (`models.model3d.infer_ct`: preprocesamiento, detección axial,
segmentación semántica, cabeza de máscara, clase por caja y fusión) y la medición de separación
(`models.segmentation.separation_distances`) en una llamada sin interfaz; los visualizadores de
dashboard/ consumen `predict_case`. Todo se devuelve en la rejilla del CT reorientado, con el
espaciado del header, así que las distancias en mm son comparables con el análisis de ground truth
(scripts/run_distancias_gt.py).
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # una operación del CBAM 3D no existe en la GPU del Mac

import numpy as np
import torch

from src import config
from src.models import detection as det
from src.models import model3d as M
from scipy.ndimage import distance_transform_edt

from src.models.segmentation import separation_distances


def device_disponible() -> str:
    """Dispositivo de inferencia: CUDA si hay, después MPS y por defecto CPU."""
    return det.get_device("auto").type


def load_model(run: str = config.DASH_MODEL_RUN, device: str | None = None) -> torch.nn.Module:
    """Carga `best.pth` del run en modo evaluación, con la arquitectura guardada en el checkpoint
    (tipo de atención, canal trabecular, cabeza de máscara…), igual que `model3d evaluate`."""
    ck = M.load_checkpoint(config.DETECTION_DIR / run / "best.pth")
    return M.load_model(ck, torch.device(device or device_disponible()))


def puntos_de_medicion(etiquetas: np.ndarray, principal: int, comps: list[int], spacing_zyx) -> dict[int, dict]:
    """Para cada componente de `comps`, el par de vóxeles más cercanos entre él y el principal (borde a borde).

    Una sola EDT del principal (con `return_indices`) en la caja que envuelve el hueso: la distancia de cada vóxel al
    principal y cuál es su vóxel más cercano. Devuelve comp → {"distancia_mm", "punto" (en el fragmento) y
    "punto_principal"}, los dos en índices (z, y, x) del CT. Es la misma medida de `separation_distances`."""
    if not comps:
        return {}
    idx = np.argwhere(etiquetas > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    sub = etiquetas[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    dist, ind = distance_transform_edt(sub != principal, sampling=spacing_zyx, return_indices=True)
    out = {}
    for comp in comps:
        d = np.where(sub == comp, dist, np.inf)
        p = np.unravel_index(int(np.argmin(d)), d.shape)
        q = ind[:, p[0], p[1], p[2]]
        out[comp] = {"distancia_mm": float(d[p]), "punto": tuple(int(v) for v in np.array(p) + lo),
                     "punto_principal": tuple(int(v) for v in q + lo)}
    return out


def predict_case(img, model: torch.nn.Module, device: str | None = None) -> dict:
    """Pipeline completo sobre un CT ya reorientado (sitk.Image), en la rejilla original.

    Etapas: `model3d.infer_ct` (cubo isótropo y cubo a doble resolución, detección → regiones →
    segmentación semántica → cabeza de máscara y clase por caja → fusión) y distancia de separación
    de cada fragmento conminuto (componentes conexos de cada hueso; el principal es el de más
    vóxeles; EDT borde a borde en mm con el espaciado del header, fragmentos de al menos
    M3D_MIN_FRAGMENT_MM3).

    La máscara que se muestra es `config.DASH_OUTPUT` ("fusion", "mask" o "semantic"); si el modelo
    no tiene cabeza de máscara, la semántica.

    Devuelve un diccionario serializable (numpy + listas/dicts) para cachear en el dashboard:

      instancias     uint8 (Z, Y, X) del CT original, un id por hueso (1, 2, 3)
      fragmentos     una fila por fragmento (componente conexo) de cada hueso: hueso, región, prob,
                     comp, principal, voxels, distancia_mm y centroide (z, y, x) en índices de vóxel;
                     en los conminutos, además `punto` y `punto_principal`: los dos vóxeles más
                     cercanos entre el fragmento y el principal (los extremos de la medición)
      fragmentos_id  uint8 (Z, Y, X) del CT original, un id consecutivo (1, 2, …) por fragmento en
                     el mismo orden de `fragmentos`; 0 fuera de todo hueso
      cajas          detecciones por corte en índices del CT original (puntaje ≥ M3D_MASK_SCORE_THR),
                     con el hueso que dice la rama de clase
      salida         nombre de la salida de segmentación mostrada
      spacing        (sx, sy, sz) del CT original en mm
      tiempos        segundos por etapa
      device         dispositivo usado
    """
    dev = torch.device(device) if device else next(model.parameters()).device
    res = M.infer_ct(model, img, dev)
    salida = config.DASH_OUTPUT if res.get(config.DASH_OUTPUT) is not None else "semantic"
    instancias = res[salida]
    tiempos = {"preproceso": res["time_s"]["preprocesamiento"], "pipeline": res["time_s"]["red"]}

    t0 = time.perf_counter()
    spacing = tuple(float(s) for s in img.GetSpacing())                # (sx, sy, sz)
    fragmentos_id = np.zeros_like(instancias)
    fragmentos: list[dict] = []
    # El umbral M3D_MIN_FRAGMENT_MM3 (≈ 191 mm³) es en mm³: en la rejilla original
    # (vóxel anisótropo) corresponden más vóxeles que los 50 del cubo isótropo.
    min_voxels_orig = max(1, round(config.M3D_MIN_FRAGMENT_MM3 / float(np.prod(spacing))))
    # Confianza del hueso: puntaje medio de las cajas detectadas de ese hueso
    prob = res["boxes"].groupby("hueso").puntaje.mean().to_dict() if len(res["boxes"]) else {}

    for id_h, region in enumerate(config.REGION_NAMES, start=1):
        hueso = instancias == id_h
        if not hueso.any():
            continue
        etiquetas, principal, filas = separation_distances(hueso, spacing[::-1], min_voxels=min_voxels_orig)
        if not principal or not (etiquetas == principal).any():       # todo el hueso bajo el mínimo
            continue
        puntos = puntos_de_medicion(etiquetas, principal, [f["comp"] for f in filas], spacing[::-1])
        for fila in filas + [{"comp": principal, "distance_mm": 0.0}]:
            comp = fila["comp"]
            mascara = etiquetas == comp
            fragmentos_id[mascara] = len(fragmentos) + 1
            vz, vy, vx = np.nonzero(mascara)
            fragmentos.append({
                "hueso": int(id_h), "region": region, "prob": round(float(prob.get(region, 1.0)), 4), "comp": int(comp),
                "principal": bool(comp == principal),
                "voxels": int(mascara.sum()),
                "distancia_mm": round(float(fila["distance_mm"]), 3),
                "centroide": tuple(float(v) for v in (vz.mean(), vy.mean(), vx.mean())),
                "punto": puntos.get(comp, {}).get("punto"),
                "punto_principal": puntos.get(comp, {}).get("punto_principal"),
            })
    tiempos["distancias"] = time.perf_counter() - t0

    cajas = [{"x0": float(r.x0), "x1": float(r.x1), "y0": float(r.y0), "y1": float(r.y1), "z0": int(r.corte),
              "z1": int(r.corte), "score": float(r.puntaje), "hueso": r.hueso} for r in res["boxes"].itertuples()]
    return {"instancias": instancias, "fragmentos": fragmentos, "fragmentos_id": fragmentos_id, "cajas": cajas,
            "salida": salida, "spacing": spacing, "tiempos": {k: round(v, 2) for k, v in tiempos.items()},
            "device": dev.type}
