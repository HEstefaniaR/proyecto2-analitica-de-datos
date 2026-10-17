"""Compara las dos salidas de segmentación del modelo 3D (cabeza semántica y cabeza de máscara) en la MISMA
rejilla, y guarda ejemplos con las salidas de todas las cabezas para el notebook 03.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/comparar_cabezas.py [--runs unet3d_cbam_trab unet3d_se_trab]

`model3d evaluate` mide la cabeza semántica en el cubo (256³, 1,56 mm) y la cabeza de máscara a doble resolución en
el plano (256 × 512 × 512, 0,78 mm), cada una contra la máscara real de su rejilla. Aquí las dos se miden contra la
misma máscara real, la de doble resolución (la semántica se lleva a 512 × 512 por vecino más cercano):

- `semantica`: salida final de la cabeza semántica (dentro de las regiones detectadas).
- `mascara`: salida de la cabeza de máscara (una máscara por caja axial con puntaje ≥ `M3D_MASK_SCORE_THR`).
- `fusion`: la cabeza de máscara y, donde no marca nada, la semántica.
- `fusion_huesos`, `fusion_union_sacro`, `fusion_relleno`: los modos de `model3d.fuse_heads` (coxales de la cabeza de
  máscara y sacro de la semántica, con dos variantes). El modo se elige en validación (`--split val`).

Salidas en `results/detection/<run>/eval/`:
- `heads_<split>.csv`: CT × salida × hueso con Dice, IoU, HD95 y ASSD.
- `heads_cases_<split>.csv`: por CT, cortes con hueso real, cortes donde cada salida queda vacía y acuerdo entre las dos.
- `heads_fragments_<split>.csv`: por fragmento conminuto real, fracción de sus vóxeles que cada salida marca con su hueso.
- `examples_heads/<caso>.npz`: cuatro CT (el mejor, el peor, donde más gana la cabeza de máscara y donde más gana la
  semántica; se eligen con el primer run y se guardan los mismos en los demás) más los de `--extra`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models import detection as det
from src.models import model3d as M
from src.models.segmentation import head_metrics

F_HR = config.M3D_HIRES_FACTOR
SPACING = (config.VOXEL_MM, config.VOXEL_MM / F_HR, config.VOXEL_MM / F_HR)


def slice_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Dice de los tres huesos juntos en cada corte axial (NaN si el corte no tiene hueso real ni predicho)."""
    inter = ((pred == gt) & (gt > 0)).sum((1, 2))
    den = (pred > 0).sum((1, 2)) + (gt > 0).sum((1, 2))
    return np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)


@torch.no_grad()
def predict(model, cid: str, device):
    """Salidas de todas las cabezas en un CT y la máscara real a doble resolución."""
    img, msk, _ = M.VolumeDataset([cid])[0]
    x, _ = M.to_gpu(img[None], msk[None], device)
    img_hr, msk_hr = M.load_hires(cid)
    res = M.infer_instances_axial(model, x, img_hr=img_hr)
    sem = res["instances"][0].cpu().numpy().astype(np.uint8)                       # (Z, Y, X)
    mask = res["mask_labels"]                                                        # (Z, 2Y, 2X)
    codes = msk_hr.numpy()
    gt = np.where(codes > 0, (codes.astype(np.int16) - 1) // 10 + 1, 0).astype(np.uint8)
    return {"sem": sem, "mask": mask, "gt": gt, "codes": codes, "rois": res["rois"], "n_regions": len(res["regions"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cabeza semántica frente a cabeza de máscara en la misma rejilla.")
    ap.add_argument("--runs", nargs="+", default=[config.M3D_RUN_CBAM, config.M3D_RUN_SE])
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--extra", nargs="*", default=[], help="CT que se guardan además de los cuatro elegidos")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    det.seed_everything()
    device = det.get_device(args.device)
    ids = M.case_ids(args.split)
    chosen = None

    for run in args.runs:
        out_dir = config.DETECTION_DIR / run / "eval"
        model = M.load_model(M.load_checkpoint(config.DETECTION_DIR / run / "best.pth"), device)
        rows, case_rows, frag_rows, keep = [], [], [], {}
        for cid in ids:
            p = predict(model, cid, device)
            gt = p["gt"]
            sem_up = p["sem"].repeat(F_HR, 1).repeat(F_HR, 2)
            outs = {"semantica": sem_up, "mascara": p["mask"], "fusion": np.where(p["mask"] > 0, p["mask"], sem_up),
                    **{f"fusion_{m}": M.fuse_heads(p["sem"], p["mask"], m) for m in M.FUSION_MODES}}
            for name, lab in outs.items():
                for hueso, mt in head_metrics(lab, gt, M.SEM_CLASSES, spacing=SPACING).items():
                    rows.append({"case_id": cid, "split": args.split, "salida": name, "hueso": hueso, **mt})
            sd = {k: slice_dice(outs[k], gt) for k in ("semantica", "mascara")}
            has = (gt > 0).any((1, 2))
            agree = 2 * ((sem_up == p["mask"]) & (sem_up > 0)).sum() / max((sem_up > 0).sum() + (p["mask"] > 0).sum(), 1)
            case_rows.append({"case_id": cid, "split": args.split, "cortes_con_hueso": int(has.sum()),
                              "cortes_vacios_semantica": int((has & ~(sem_up > 0).any((1, 2))).sum()),
                              "cortes_vacios_mascara": int((has & ~(p["mask"] > 0).any((1, 2))).sum()),
                              "dice_entre_cabezas": float(agree), "cajas": len(p["rois"]), "regiones": p["n_regions"]})
            # Fragmentos conminutos reales: cuánto de cada uno marca cada salida con su hueso
            for b, region in enumerate(config.REGION_NAMES, start=1):
                codes_b = [c for c in config.REGION_CODES[region] if (p["codes"] == c).any()]
                main_c = max(codes_b, key=lambda c: int((p["codes"] == c).sum())) if codes_b else None
                for c in codes_b:
                    f = p["codes"] == c
                    frag_rows.append({"case_id": cid, "split": args.split, "region": region, "fragmento": c,
                                      "principal": c == main_c, "volumen_mm3": float(f.sum() * np.prod(SPACING)),
                                      **{f"recall_{k}": float((outs[k][f] == b).mean()) for k in outs}})
            keep[cid] = {"sem": p["sem"], "mask": p["mask"],
                         "boxes": np.array([[r["slice"], *r["box"]] for r in p["rois"]]).reshape(-1, 5),
                         "scores": np.array([r["score"] for r in p["rois"]]),
                         "cls": np.array([r["class"] for r in p["rois"]], dtype=np.int64),
                         "slice_dice_sem": sd["semantica"], "slice_dice_mask": sd["mascara"]}
            d = {k: next(r["dice"] for r in rows[::-1] if r["case_id"] == cid and r["salida"] == k and r["hueso"] == "promedio")
                 for k in outs}
            print(f"[{run}] {cid}: Dice macro semántica {d['semantica']:.4f} | máscara {d['mascara']:.4f} | "
                  f"fusión {d['fusion']:.4f}", flush=True)
        heads = pd.DataFrame(rows)
        heads.to_csv(out_dir / f"heads_{args.split}.csv", index=False)
        pd.DataFrame(case_rows).to_csv(out_dir / f"heads_cases_{args.split}.csv", index=False)
        pd.DataFrame(frag_rows).to_csv(out_dir / f"heads_fragments_{args.split}.csv", index=False)
        if chosen is None:                                    # se eligen con el primer run
            t = heads[heads.hueso == "promedio"].pivot(index="case_id", columns="salida", values="dice")
            media, dif = t[["semantica", "mascara"]].mean(1), t.mascara - t.semantica
            chosen = {"mejor": media.idxmax(), "peor": media.idxmin(), "gana_mascara": dif.idxmax(),
                      "gana_semantica": dif.idxmin()}
            print("Casos elegidos:", chosen, flush=True)
        if args.split != "test":                              # los ejemplos del notebook 03 son de test
            del keep, model
            continue
        ex = out_dir / "examples_heads"
        ex.mkdir(exist_ok=True)
        for q in ex.glob("*.npz"):
            q.unlink()
        for cid in dict.fromkeys([*chosen.values(), *args.extra]):
            papel = ",".join(k for k, v in chosen.items() if v == cid)
            np.savez_compressed(ex / f"{cid}.npz", papel=papel, **keep[cid])
        del keep, model


if __name__ == "__main__":
    main()
