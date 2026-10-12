"""Baseline SAM zero-shot: segmenta las cajas PREDICHAS por el detector y mide Dice/IoU contra la GT.

    cd ENTREGABLE && uv run python scripts/run_sam_baseline.py --run detection/k3_cbam [--limit 3]

Por cada corte del split (una vista, por defecto coronal) con al menos una caja final tras NMS:
el detector entrega las cajas con `SliceDataset` + `make_loader` + `postprocess` (el mismo camino
de inferencia del proyecto), cada caja se usa como prompt de caja de SAM (una por región; la GT
nunca entra como prompt) y la máscara resultante se compara con la GT de esa región (mapa de
clases del corte binarizado) con Dice e IoU (`src/metrics/sam_baseline.py`).

Salidas en `results/sam/`: `sam_baseline_<split>.csv` (una fila por corte y región detectada, con
columnas case_id, split, slice, region, score, dice, iou, n_gt_px, n_pred_px) y `resumen.md`
(métricas globales y por región, latencia de SAM y configuración de la corrida). Con `--limit N`
se procesan solo los N primeros casos (prueba de humo) y las salidas llevan el sufijo `_limitN`.

Si `load_model` no puede cargar `<run>/best.pth` (checkpoint de otra versión del código), se
recorren los demás runs con `best.pth` de `detection/` y `results/detection/` y se documenta cuál
se usó.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import preprocessing as pre
from src.metrics import sam_baseline as samb
from src.models import detection as det

PROYECTO = Path(__file__).resolve().parents[1]
SAM_CHECKPOINT = PROYECTO / "models" / "sam" / "sam_vit_b_01ec64.pth"


# ======================================================================================
# Detector: carga con fallback y cajas por corte
# ======================================================================================

def checkpoints_disponibles() -> list[Path]:
    """Todos los `best.pth` de runs del detector (carpetas `detection/` y `results/detection/`)."""
    patrones = ("detection/*/best.pth", "results/detection/*/best.pth")
    return sorted({p for patron in patrones for p in PROYECTO.glob(patron)})


def cargar_detector(run: str, device: torch.device) -> tuple[torch.nn.Module, dict, Path]:
    """`load_model` sobre `<run>/best.pth`; si ese checkpoint no es compatible, prueba el resto.

    Devuelve (modelo, config, ruta usada). Los intentos fallidos se listan en pantalla.
    """
    pedido = (PROYECTO / run / "best.pth").resolve()
    candados = [pedido] + [p for p in checkpoints_disponibles() if p != pedido]
    errores = []
    for ruta in candados:
        try:
            modelo, cfg = det.load_model(ruta, device)
        except Exception as e:                                   # checkpoint de otra arquitectura o corrupto
            errores.append(f"  {ruta}: {type(e).__name__}: {e}")
            continue
        if ruta != pedido:
            print(f"ATENCION: no se pudo cargar el run pedido ({run}); se usa {ruta.relative_to(PROYECTO)}")
        return modelo, cfg, ruta
    raise RuntimeError("Ningún checkpoint del detector se pudo cargar:\n" + "\n".join(errores))


@torch.no_grad()
def cajas_del_detector(modelo, cortes: pd.DataFrame, device: torch.device, view: str, k: int,
                       score_thr: float, iou_thr: float, batch: int = 32) -> list[dict]:
    """Cajas post-NMS del detector para cada corte de `cortes` (una sola vista), en el orden de las filas.

    Mismo camino de inferencia que `run_inference` (SliceDataset + make_loader + to_device +
    autocast), pero conserva la salida cruda por lote para aplicar `postprocess` con los umbrales
    de la línea de comandos.
    """
    loader = det.make_loader(det.SliceDataset(cortes, k_slices=k), batch)
    salida: list[dict | None] = [None] * len(cortes)
    for x_u8, m_u8, idx, _ in loader:
        x, _ = det.to_device(x_u8, m_u8, device)
        with det.autocast(device):
            out = modelo(x, view)
        preds = det.postprocess(out, score_thr=score_thr, iou_thr=iou_thr)
        for j, i in enumerate(idx.tolist()):
            salida[i] = preds[j]
    return salida


# ======================================================================================
# GT y verificación de cajas
# ======================================================================================

def caja_de_mascara(m: np.ndarray) -> np.ndarray | None:
    """Caja (x0, y0, x1, y1) en píxeles de una máscara binaria (H, W); None si está vacía."""
    idx = np.argwhere(m)
    if not len(idx):
        return None
    (r0, c0), (r1, c1) = idx.min(0), idx.max(0) + 1
    return np.array([c0, r0, c1, r1], dtype=float)


def verificar_cajas(fila: pd.Series, pred: dict, y: np.ndarray, view: str) -> str:
    """Líneas de verificación: caja predicha de cada región y su IoU de caja contra la GT del corte."""
    lineas = []
    for bx, sc, lb in zip(pred["boxes"].numpy(), pred["scores"].numpy(), pred["labels"].numpy()):
        region = det.CLASSES[view][int(lb)]
        c = det.CLASSES[view].index(region)
        gt = caja_de_mascara(y == c + 1)
        nombre = config.REGION_LABELS.get(region, region)
        if gt is None:
            lineas.append(f"    {nombre:10s} score {sc:.2f}  caja ({bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f})"
                          f"  GT vacía en el corte (falso positivo del detector)")
            continue
        iou = float(det.box_iou(torch.as_tensor(bx[None], dtype=torch.float32),
                                torch.as_tensor(gt[None], dtype=torch.float32))[0, 0])
        lineas.append(f"    {nombre:10s} score {sc:.2f}  caja ({bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f})"
                      f"  GT ({gt[0]:.0f},{gt[1]:.0f},{gt[2]:.0f},{gt[3]:.0f})  IoU caja {iou:.3f}")
    return "\n".join(lineas)


# ======================================================================================
# Corrida
# ======================================================================================

def tabla_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Media y mediana de Dice e IoU por región y global, contando solo filas con métrica definida."""
    d = df.dropna(subset=["dice", "iou"])
    por_region = d.groupby("region").agg(n=("iou", "size"), dice_media=("dice", "mean"),
                                         dice_mediana=("dice", "median"), iou_media=("iou", "mean"),
                                         iou_mediana=("iou", "median"))
    glob = pd.DataFrame({"n": [len(d)], "dice_media": [d.dice.mean()], "dice_mediana": [d.dice.median()],
                         "iou_media": [d.iou.mean()], "iou_mediana": [d.iou.median()]}, index=["global"])
    tabla = pd.concat([por_region, glob])
    tabla.index = [f"{config.REGION_LABELS[i]} ({i})" if i in config.REGION_LABELS else i for i in tabla.index]
    return tabla.round(3)


def escribir_resumen(df: pd.DataFrame, lat: dict[str, list[float]], args, cfg_det: dict, checkpoint: Path,
                     device: torch.device, n_cortes: int, n_con_cajas: int, ruta_md: Path) -> None:
    """`resumen.md`: métricas globales y por región, latencia de SAM por etapa y configuración."""
    lat_set, lat_pred = np.asarray(lat["set_image"]), np.asarray(lat["predict"])
    excluidas = int((df.dice.isna() & df.iou.isna()).sum())
    tabla = tabla_resumen(df).to_string(float_format=lambda v: f"{v:.3f}")
    pedidos = ", ".join(f"{k} = {v}" for k, v in [("SCORE_THR", args.score_thr), ("NMS_IOU", args.iou_thr)])
    md = f"""# Baseline SAM zero-shot con las cajas del detector como prompt

SAM ViT-B (`{checkpoint.relative_to(PROYECTO)}`) en modo inferencia zero-shot: para cada corte con
detecciones, la caja PREDICHA por el detector (post-NMS) es el prompt de caja de cada región. La
GT nunca se usa como prompt. Dice e IoU contra la máscara GT del corte por región.

| Parámetro | Valor |
| --- | --- |
| Checkpoint del detector usado | `{checkpoint.relative_to(PROYECTO)}` |
| Run solicitado | `{args.run}` |
| Split / vista | {args.split} / {args.view} |
| Umbrales del detector | {pedidos} |
| K_SLICES del run | {cfg_det.get('k_slices', config.K_SLICES)} |
| Dispositivo | {device.type} |
| Casos / cortes del split | {df.case_id.nunique()} / {n_cortes} |
| Cortes con cajas (entran a SAM) | {n_con_cajas} |
| Filas (corte × región) | {len(df)} ({excluidas} sin métrica: GT y predicción vacías) |

## Dice e IoU por región (media y mediana)

```
{tabla}
```

Filas contadas: las que tienen métrica definida. Una caja en un corte sin GT de esa región cuenta
como fila con Dice/IoU 0.0 (falso positivo del detector). Las filas con GT y predicción vacías no
entran en los resúmenes.

## Latencia de SAM por corte (solo inferencia de SAM, {len(lat_set)} cortes)

| Etapa | Media (s) | Mediana (s) |
| --- | --- | --- |
| set_image (codificación de la imagen) | {lat_set.mean():.3f} | {np.median(lat_set):.3f} |
| predict (cajas del corte en un lote) | {lat_pred.mean():.3f} | {np.median(lat_pred):.3f} |
| Total por corte | {(lat_set + lat_pred).mean():.3f} | {np.median(lat_set + lat_pred):.3f} |
"""
    ruta_md.write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline SAM zero-shot con cajas del detector como prompt.")
    ap.add_argument("--run", default="detection/k3_cbam", help="carpeta del run del detector (checkpoint <run>/best.pth)")
    ap.add_argument("--sam-checkpoint", default=str(SAM_CHECKPOINT), help="ruta del checkpoint de SAM ViT-B")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--view", default="coronal", choices=det.VIEWS)
    ap.add_argument("--limit", type=int, default=0, help="solo los N primeros casos (prueba de humo)")
    ap.add_argument("--score-thr", type=float, default=config.SCORE_THR)
    ap.add_argument("--iou-thr", type=float, default=config.NMS_IOU)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--batch", type=int, default=32, help="cortes por lote en la inferencia del detector")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "sam"), help="carpeta de salida")
    args = ap.parse_args()

    det.seed_everything()
    device = det.get_device(args.device)
    modelo, cfg_det, checkpoint = cargar_detector(args.run, device)
    k = int(cfg_det.get("k_slices", config.K_SLICES))
    predictor = samb.load_sam(args.sam_checkpoint, device=args.device)
    print(f"SAM ViT-B en {device.type} | detector: {checkpoint.relative_to(PROYECTO)} (k_slices = {k})")

    index = det.load_index()
    cortes = det.split_views(index, args.split, 1)[args.view].reset_index(drop=True)
    if args.limit:
        casos = sorted(cortes.case_id.unique())[:args.limit]
        cortes = cortes[cortes.case_id.isin(casos)].reset_index(drop=True)
    print(f"Split {args.split} · vista {args.view}: {len(cortes)} cortes de {cortes.case_id.nunique()} casos")

    preds = cajas_del_detector(modelo, cortes, device, args.view, k, args.score_thr, args.iou_thr, args.batch)

    salida = Path(args.out)
    salida.mkdir(parents=True, exist_ok=True)
    sufijo = f"_limit{args.limit}" if args.limit else ""
    ruta_csv = salida / f"sam_baseline_{args.split}{sufijo}.csv"

    vols: dict[str, tuple] = {}

    def cubos(cid: str):
        if cid not in vols:                                   # memmap: carga diferida, una vez por caso
            vols[cid] = (np.load(config.CACHE_DIR / f"{cid}_img.npy", mmap_mode="r"),
                         np.load(config.CACHE_DIR / f"{cid}_msk.npy", mmap_mode="r"))
        return vols[cid]

    filas, lat = [], {"set_image": [], "predict": []}
    n_con_cajas, n_verificados = 0, 0
    t0 = time.time()
    for n in range(len(cortes)):
        p = preds[n]
        if len(p["boxes"]) == 0:
            continue
        fila = cortes.iloc[n]
        cid, s = fila["case_id"], int(fila["slice"])
        n_con_cajas += 1
        img_m, msk_m = cubos(cid)
        y = det.VIEW_LUT[args.view][pre.view_slices(msk_m, args.view, s)]

        if n_verificados < 3:                                 # verificación: ¿las cajas son coherentes con la GT?
            n_verificados += 1
            print(f"Verificación de cajas · caso {cid} · corte {s}:\n{verificar_cajas(fila, p, y, args.view)}")

        rgb = samb.slice_to_rgb(pre.view_slices(img_m, args.view, s))
        boxes = p["boxes"].numpy().astype(np.float32)
        scores, labels = p["scores"].numpy(), p["labels"].numpy()
        orden = np.argsort(-scores)                           # una caja por región; orden estable por puntaje
        tiempos: dict[str, float] = {}
        masks = samb.masks_for_boxes(predictor, rgb, boxes[orden], tiempos=tiempos)
        lat["set_image"].append(tiempos["set_image"])
        lat["predict"].append(tiempos["predict"])

        for j, jj in enumerate(orden):
            region = det.CLASSES[args.view][int(labels[jj])]
            gt = y == det.CLASSES[args.view].index(region) + 1
            dice, iou = samb.dice_iou(masks[j], gt)
            filas.append({"case_id": cid, "split": args.split, "slice": s, "region": region,
                          "score": float(scores[jj]), "dice": dice, "iou": iou,
                          "n_gt_px": int(gt.sum()), "n_pred_px": int(masks[j].sum())})
        if n_con_cajas % 50 == 0:
            print(f"  {n_con_cajas} cortes con cajas ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(filas, columns=["case_id", "split", "slice", "region", "score", "dice", "iou",
                                      "n_gt_px", "n_pred_px"])
    df.to_csv(ruta_csv, index=False)
    ruta_md = salida / f"resumen{sufijo}.md"
    escribir_resumen(df, lat, args, cfg_det, checkpoint, device, len(cortes), n_con_cajas, ruta_md)
    print(f"Guardado: {ruta_csv} ({len(df)} filas, {n_con_cajas} cortes con cajas, {time.time() - t0:.0f} s)")
    print(f"Guardado: {ruta_md}")


if __name__ == "__main__":
    main()
