"""Datos para verificar la división train / val / test (notebook 05): reproducción de `splits.csv`, parecido entre CT
de splits distintos y pérdida del modelo final sin aumento de datos en los 100 CT.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/verificar_division.py [--run unet3d_cbam_trab]

Salidas en `results/splits/`:
- `reproduccion.csv`: split y tipo de fractura de `splits/splits.csv` frente a los que da hoy `src.dataset.splits`
  (lee las 100 máscaras otra vez).
- `parecido.npz`: correlación entre cada par de CT (cubo del caché reducido a 32³) y el md5 de cada cubo.
- `perdida_sin_aumento.csv`: por CT, la misma pérdida que la curva de validación (modo evaluación, sin aumento, cortes
  de detección con semilla fija) y el Dice de la cabeza semántica. En train es lo que el modelo logra sobre los CT con
  los que se entrenó, sin el aumento.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import splits as sp
from src.models import detection as det
from src.models import model3d as M
from src.models.segmentation import segmentation_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Datos para verificar la división de los casos.")
    ap.add_argument("--run", default=config.M3D_RUN_CBAM)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = config.RESULTS_DIR / "splits"
    out.mkdir(parents=True, exist_ok=True)
    actual = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})

    # 1. Reproducción de la división con el código y la semilla de hoy
    nuevo = sp.make_splits(sp.list_cases())
    rep = actual.merge(nuevo, on="case_id", how="outer", suffixes=("_csv", "_recalculado"))
    rep.to_csv(out / "reproduccion.csv", index=False)
    print(f"Reproducción: {int((rep.split_csv == rep.split_recalculado).sum())} de {len(rep)} casos con el mismo split; "
          f"{int((rep.fracture_type_csv == rep.fracture_type_recalculado).sum())} con el mismo tipo de fractura", flush=True)

    # 2. Parecido entre CT: cubo del caché reducido a 32³ (promedio de bloques de 8³), correlación de Pearson
    ids, small, md5 = actual.case_id.tolist(), [], []
    for cid in ids:
        img = np.load(config.CACHE_DIR / f"{cid}_img.npy")
        md5.append(hashlib.md5(img.tobytes()).hexdigest())
        n = img.shape[0] // 8
        small.append(img.reshape(n, 8, n, 8, n, 8).mean((1, 3, 5), dtype=np.float32).ravel())
    np.savez_compressed(out / "parecido.npz", case_id=np.array(ids), corr=np.corrcoef(np.stack(small)), md5=np.array(md5))
    print(f"Parecido: {len(set(md5))} cubos distintos de {len(ids)}", flush=True)

    # 3. Pérdida sin aumento en los 100 CT con el modelo final
    device = det.get_device(args.device)
    model = M.load_model(M.load_checkpoint(config.DETECTION_DIR / args.run / "best.pth"), device)
    rows = []
    for cid, split in zip(actual.case_id, actual.split):
        item = M.VolumeDataset([cid], hires=model.mask_head)[0]
        x, m = M.to_gpu(item[0][None], item[1][None], device)
        hr = (item[2], item[3], None) if model.mask_head else None
        with torch.no_grad():
            loss, t = M.step_loss(model, x, m, torch.Generator().manual_seed(config.SEED), hr=hr)
            with det.autocast(device):
                feats = model(x)
            sm = segmentation_metrics(feats["sem_logits"].float(), m[None])
        rows.append({"case_id": cid, "split": split, "loss": float(loss),
                     **{k: float(t[k]) for k in ("obj", "box", "seg", "anat", "mask") if k in t},
                     **{k: float(v) for k, v in sm.items()}})
        print(f"[{split}] {cid}: pérdida {float(loss):.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in sm.items()), flush=True)
        pd.DataFrame(rows).to_csv(out / "perdida_sin_aumento.csv", index=False)
    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
