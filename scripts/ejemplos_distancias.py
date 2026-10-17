"""Ejemplos visuales de la distancia de separación: real frente a predicha sobre las máscaras.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/ejemplos_distancias.py

Para cada fragmento de `EJEMPLOS` (caso, hueso, código de fragmento):
- Real: EDT 3D del principal del hueso (el fragmento con más vóxeles) en el espacio original del CT, con el espaciado
  del header (igual que `src.metrics.distancias`). Se guardan los dos puntos más cercanos (fragmento y principal).
- Predicha: la máscara del hueso que da el modelo vuelve al espacio original; componentes conexos (el más grande es
  el principal) y la misma EDT (igual que `model3d evaluate`). El fragmento real se empareja con el componente que más
  se solapa: si es el principal, quedó unido (0 mm); si no se solapa con nada, no se detectó.

Guarda en `results/distancias/ejemplos/<caso>_<fragmento>.npz` el corte axial del punto más cercano (recortado), las
dos máscaras y los puntos; el notebook `p4_distancias_sam.ipynb` los dibuja.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import preprocessing as pre
from src.models import detection as det
from src.models import model3d as M
from src.models.segmentation import separation_distances

# (caso, hueso, fragmento): un fragmento que toca su hueso, uno separado que el modelo separa y dos separados que no
EJEMPLOS = [("012", "left_hipbone", 12), ("039", "right_hipbone", 23), ("052", "right_hipbone", 23),
            ("084", "left_hipbone", 14)]
HALF_MM = 60                                   # medio lado del recorte del corte, en mm


def puntos_cercanos(lab: np.ndarray, main, frag, sp):
    """Distancia mínima (mm) entre `lab == frag` y `lab == main`, y los dos vóxeles más cercanos (z, y, x)."""
    idx = np.argwhere((lab == main) | (lab == frag))
    lo, hi = idx.min(0), idx.max(0) + 1
    sub = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    d, ind = distance_transform_edt(sub != main, sampling=sp, return_indices=True)
    d = np.where(sub == frag, d, np.inf)
    p = np.unravel_index(np.argmin(d), d.shape)
    q = tuple(ind[:, p[0], p[1], p[2]])
    return float(d[p]), np.array(p) + lo, np.array(q) + lo


def main() -> None:
    device = det.get_device("auto")
    ck = M.load_checkpoint(config.DETECTION_DIR / config.M3D_RUN_CBAM / "best.pth")
    model = M.load_model(ck, device)   # arquitectura del checkpoint; ignora inst_head
    img_paths = {q.stem: q for d in config.IMAGE_DIRS for q in d.glob("*.mha")}
    out = config.RESULTS_DIR / "distancias" / "ejemplos"
    out.mkdir(parents=True, exist_ok=True)

    for cid, region, frag in EJEMPLOS:
        # Predicción del modelo (mismo pipeline que `evaluate`)
        img_u8, msk, _ = M.VolumeDataset([cid])[0]
        x, m = M.to_gpu(img_u8[None], msk[None], device)
        sl = torch.arange(0, M.N, device=device)
        with torch.no_grad(), det.autocast(device):
            feats = model(x)
            preds = M.postprocess_objectness(model.detect(feats, "axial", sl))
            regions = M.group_boxes_across_slices(preds, sl.cpu(), iou_threshold=config.M3D_TRACK_IOU)
            inst = model.classify_pred_instances(feats, regions)["instances"][0].cpu().numpy().astype(np.uint8)

        # Espacio original
        img = pre.reorient(pre.load_volume(img_paths[cid]))
        ct = pre.sitk.GetArrayFromImage(img)
        codes = pre.sitk.GetArrayFromImage(pre.reorient(pre.load_volume(config.LABEL_DIR / f"{cid}.mha")))
        pred = pre.cube_to_original(inst, pre.cube_geometry(img), img)
        sp = tuple(float(v) for v in img.GetSpacing())[::-1]                 # (z, y, x) en mm
        b = config.REGION_NAMES.index(region) + 1
        rc = [c for c in config.REGION_CODES[region] if (codes == c).any()]
        main_c = max(rc, key=lambda c: int((codes == c).sum()))

        d_gt, p_gt, q_gt = puntos_cercanos(codes, main_c, frag, sp)

        min_vox = int(np.ceil(config.M3D_MIN_FRAGMENT_MM3 / np.prod(sp)))
        lp, mp, _ = separation_distances(pred == b, sp, min_vox)
        ov = lp[codes == frag]
        ov = ov[ov > 0]
        k = int(np.bincount(ov).argmax()) if len(ov) else 0
        if k == 0:
            estado, d_pr, p_pr, q_pr = "no detectado", np.nan, None, None
        elif k == mp:
            estado, d_pr, p_pr, q_pr = "unido al principal", 0.0, None, None
        else:
            estado = "separado"
            d_pr, p_pr, q_pr = puntos_cercanos(lp, mp, k, sp)

        # Corte axial del punto real más cercano, recortado alrededor de los dos puntos
        z = int(p_gt[0])
        cy, cx = (p_gt[1:] + q_gt[1:]) / 2
        hy, hx = int(HALF_MM / sp[1]), int(HALF_MM / sp[2])
        hy = max(hy, int(abs(p_gt[1] - q_gt[1]) / 2) + 20)
        hx = max(hx, int(abs(p_gt[2] - q_gt[2]) / 2) + 20)
        y0, x0 = max(int(cy) - hy, 0), max(int(cx) - hx, 0)
        y1, x1 = min(int(cy) + hy, ct.shape[1]), min(int(cx) + hx, ct.shape[2])
        crop = np.s_[y0:y1, x0:x1]
        gt_l = np.zeros(codes.shape[1:], np.uint8)
        gt_l[codes[z] == main_c] = 1
        gt_l[codes[z] == frag] = 2
        pr_l = np.zeros(codes.shape[1:], np.uint8)
        pr_l[(pred[z] == b)] = 3                                              # resto del hueso predicho
        pr_l[lp[z] == mp] = 1
        if k and k != mp:
            pr_l[lp[z] == k] = 2
        rel = lambda p: None if p is None else np.array([p[0] - z, p[1] - y0, p[2] - x0])
        np.savez_compressed(
            out / f"{cid}_{frag}.npz", ct=ct[z][crop].astype(np.int16), gt=gt_l[crop], pred=pr_l[crop],
            frag_gt=(codes[z] == frag)[crop], z=z, spacing=np.array(sp), region=region, case_id=cid, frag=frag,
            d_gt=d_gt, d_pred=d_pr, estado=estado,
            p_gt=rel(p_gt), q_gt=rel(q_gt),
            p_pr=rel(p_pr) if p_pr is not None else np.full(3, np.nan),
            q_pr=rel(q_pr) if q_pr is not None else np.full(3, np.nan))
        print(f"{cid} {region} frag {frag}: real {d_gt:.2f} mm | predicha {d_pr:.2f} mm ({estado}) | corte {z}",
              flush=True)


if __name__ == "__main__":
    main()
