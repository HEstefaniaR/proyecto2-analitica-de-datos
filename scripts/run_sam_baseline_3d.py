"""Baseline SAM zero-shot con el modelo 3D final: SAM segmenta las cajas PREDICHAS por la cabeza de
detección axial del modelo 3D y se compara con las tres máscaras del propio modelo, frente a la máscara real.

    uv run python scripts/run_sam_baseline_3d.py [--run unet3d_cbam] [--split test] [--limit 3] [--stride 1]

Por cada CT del split:
1. El modelo 3D da las cajas axiales (post-NMS, puntaje ≥ `config.M3D_SAM_SCORE_THR`) y sus tres máscaras
   (sacro, coxal izquierdo, coxal derecho), igual que en `model3d evaluate`.
2. Cada caja es el prompt de SAM ViT-B en su corte axial (el corte del caché en RGB). La región de la caja es el hueso
   que el modelo predice mayoritariamente dentro de ella (la detección no tiene clase). SAM nunca recibe la máscara
   real.
3. Las máscaras de SAM se apilan en un volumen por hueso y se comparan con la máscara real, igual que las del modelo:
   Dice e IoU por hueso en 3D (en los cortes procesados) y por corte en 2D.

Reutiliza las funciones de P4 (`src/metrics/sam_baseline.py`: `load_sam`, `slice_to_rgb`, `masks_for_boxes`,
`dice_iou`). SAM es solo un baseline de comparación: no entra al pipeline ni genera etiquetas.

Salidas en `results/sam/`: `sam3d_<split>_volumen.csv` (CT × hueso), `sam3d_<split>_cortes.csv` (corte × hueso) y
`resumen_3d_<split>.md`. Con `--limit N` (prueba corta) las salidas llevan el sufijo `_limitN`.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.metrics import sam_baseline as samb
from src.models import detection as det
from src.models import model3d as M
from src.models.segmentation import bone_targets

HUESOS = {1: "sacrum", 2: "left_hipbone", 3: "right_hipbone"}


def cargar_modelo(run: str, device: torch.device):
    ck = M.load_checkpoint(config.DETECTION_DIR / run / "best.pth")
    model = M.load_model(ck, device)   # arquitectura del checkpoint; ignora inst_head
    return model, int(ck["epoch"]) + 1


@torch.no_grad()
def prediccion_modelo(model, cid: str, device: torch.device):
    """Imagen uint8 (Z, Y, X), máscara real por hueso, máscaras del modelo y cajas axiales post-NMS."""
    img, msk, _ = M.VolumeDataset([cid])[0]
    x, m = M.to_gpu(img[None], msk[None], device)
    sl = torch.arange(0, M.N, device=device)
    with det.autocast(device):
        feats = model(x)
        out = model.detect(feats, "axial", sl)
    preds = M.postprocess_objectness(out)
    regions = M.group_boxes_across_slices(preds, sl.cpu(), iou_threshold=config.M3D_TRACK_IOU)
    with det.autocast(device):
        res = model.classify_pred_instances(feats, regions)
    inst = res["instances"][0].cpu().numpy().astype(np.uint8)
    gt = bone_targets(m.cpu())[0].numpy().astype(np.uint8)
    return img.numpy(), gt, inst, preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline SAM zero-shot con las cajas axiales del modelo 3D.")
    ap.add_argument("--run", default=config.M3D_RUN_CBAM)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=0, help="solo los N primeros CT (prueba corta)")
    ap.add_argument("--stride", type=int, default=1, help="procesar 1 de cada N cortes axiales")
    ap.add_argument("--score-thr", type=float, default=config.M3D_SAM_SCORE_THR)
    ap.add_argument("--sam-checkpoint", default=str(config.SAM_CHECKPOINT))
    ap.add_argument("--device", default="auto", help="dispositivo del modelo 3D")
    ap.add_argument("--sam-device", default="auto", help="dispositivo de SAM: auto (cuda, mps o cpu) | cuda | mps | cpu")
    args = ap.parse_args()

    det.seed_everything()
    device = det.get_device(args.device)
    sam_dev = args.sam_device
    if sam_dev == "auto":
        sam_dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, epoch = cargar_modelo(args.run, device)
    predictor = samb.load_sam(args.sam_checkpoint, device=sam_dev)
    ids = M.case_ids(args.split)[: args.limit or None]
    print(f"Modelo {args.run} (época {epoch}) en {device.type} | SAM ViT-B en {sam_dev} | {args.split}: {len(ids)} CT, "
          f"1 de cada {args.stride} cortes, cajas con puntaje ≥ {args.score_thr}", flush=True)

    vol_rows, slice_rows, lat = [], [], []
    t0 = time.time()
    for n, cid in enumerate(ids, start=1):
        img, gt, inst, preds = prediccion_modelo(model, cid, device)
        sam = np.zeros_like(gt)
        cortes = list(range(0, M.N, args.stride))
        n_cajas = 0
        for k in cortes:
            p = preds[k]
            keep = p["scores"] >= args.score_thr
            boxes = p["boxes"][keep].numpy().astype(np.float32)
            if len(boxes):
                # Región de cada caja: hueso que predice el modelo mayoritariamente dentro de ella
                labels = []
                for x0, y0, x1, y1 in boxes.round().astype(int):
                    vals = inst[k, max(y0, 0):y1, max(x0, 0):x1]
                    vals = vals[vals > 0]
                    labels.append(int(np.bincount(vals).argmax()) if len(vals) else 0)
                labels = np.array(labels)
                if (labels > 0).any():
                    tiempos = {}
                    masks = samb.masks_for_boxes(predictor, samb.slice_to_rgb(img[k]), boxes[labels > 0], tiempos)
                    lat.append(tiempos["set_image"] + tiempos["predict"])
                    for mk, lb in zip(masks, labels[labels > 0]):
                        sam[k][mk & (sam[k] == 0)] = lb
                    n_cajas += int((labels > 0).sum())
            for b, region in HUESOS.items():
                d_s, i_s = samb.dice_iou(sam[k] == b, gt[k] == b)
                d_m, i_m = samb.dice_iou(inst[k] == b, gt[k] == b)
                if d_s is not None or d_m is not None:
                    slice_rows.append({"case_id": cid, "split": args.split, "slice": k, "region": region,
                                       "dice_sam": d_s, "iou_sam": i_s, "dice_modelo": d_m, "iou_modelo": i_m,
                                       "n_gt_px": int((gt[k] == b).sum())})
        for b, region in HUESOS.items():
            d_s, i_s = samb.dice_iou(sam[cortes] == b, gt[cortes] == b)
            d_m, i_m = samb.dice_iou(inst[cortes] == b, gt[cortes] == b)
            vol_rows.append({"case_id": cid, "split": args.split, "region": region, "dice_sam": d_s, "iou_sam": i_s,
                             "dice_modelo": d_m, "iou_modelo": i_m})
        v = pd.DataFrame(vol_rows[-3:])
        print(f"[{n}/{len(ids)}] {cid}: {n_cajas} cajas | Dice 3D SAM {v.dice_sam.mean():.3f} · modelo "
              f"{v.dice_modelo.mean():.3f} | {time.time() - t0:.0f} s", flush=True)

    out = config.RESULTS_DIR / "sam"
    out.mkdir(parents=True, exist_ok=True)
    suf = f"_limit{args.limit}" if args.limit else ""
    vol, cor = pd.DataFrame(vol_rows), pd.DataFrame(slice_rows)
    vol.to_csv(out / f"sam3d_{args.split}{suf}_volumen.csv", index=False)
    cor.to_csv(out / f"sam3d_{args.split}{suf}_cortes.csv", index=False)

    def tabla(df):
        t = df.groupby("region")[["dice_sam", "iou_sam", "dice_modelo", "iou_modelo"]].mean()
        t.loc["macro (3 huesos)"] = t.mean()
        return t.round(3)

    lat = np.asarray(lat) if lat else np.zeros(1)
    md = f"""# Baseline SAM zero-shot con el modelo 3D (rúbrica 4.2)

SAM ViT-B (`{Path(args.sam_checkpoint).name}`) zero-shot. Prompt: cada caja axial PREDICHA por el modelo 3D
(`{args.run}`, época {epoch}; post-NMS, puntaje ≥ {args.score_thr}); la región de la caja es el hueso que el modelo
predice dentro de ella. La máscara real nunca entra como prompt. Split {args.split}: {len(ids)} CT, 1 de cada
{args.stride} cortes axiales.

## Dice e IoU 3D por hueso (promedio de los CT)

```
{tabla(vol).to_string()}
```

## Dice e IoU 2D por corte (promedio de los cortes con hueso real o predicho)

```
{tabla(cor).to_string()}
```

## Latencia de SAM por corte con cajas ({sam_dev})

Media {lat.mean():.2f} s, mediana {np.median(lat):.2f} s (set_image + predict).
"""
    (out / f"resumen_3d_{args.split}{suf}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
