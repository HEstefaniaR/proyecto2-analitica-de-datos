"""Calcula la distancia de cada fragmento al principal de su hueso sobre las máscaras GT.

    cd ENTREGABLE && uv run python scripts/run_distancias_gt.py --split all [--modo 3d]

Selecciona los casos desde splits/splits.csv (un caso = un paciente; --split all recorre todo el
dataset), lee cada máscara con SimpleITK y toma el espaciado físico del header. Escribe un CSV
con una fila por fragmento: caso, split, tipo de fractura, región, código, si es principal,
volumen en vóxeles y distancia al principal en mm.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.metrics.distancias import codes_inesperados, filas_caso, load_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Distancia fragmento-principal (mm) sobre las máscaras GT.")
    ap.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    ap.add_argument("--modo", default="3d", choices=("3d", "2d"),
                    help="EDT volumétrica (3d, por defecto) o por corte axial (2d)")
    ap.add_argument("--out", default=None, help="CSV de salida")
    args = ap.parse_args()

    # case_id como texto: sin dtype, pandas convierte "012" en 12 y rompe la ruta del .mha
    splits = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})
    casos = (splits.sort_values("case_id") if args.split == "all"
             else splits[splits.split == args.split].sort_values("case_id"))
    assert not casos.empty, f"El split {args.split!r} no tiene casos en splits.csv"
    nombre = (f"distancias_gt_{args.modo}.csv" if args.split == "all"
              else f"distancias_gt_{args.split}_{args.modo}.csv")
    salida = Path(args.out) if args.out else config.RESULTS_DIR / "distancias" / nombre
    salida.parent.mkdir(parents=True, exist_ok=True)

    filas, t0 = [], time.time()
    for i, caso in enumerate(casos.itertuples(index=False), 1):
        mask, spacing = load_mask(config.LABEL_DIR / f"{caso.case_id}.mha")
        raros = codes_inesperados(mask)
        if raros:
            print(f"  ATENCION {caso.case_id}: códigos fuera de la taxonomía 1-30: {raros}", flush=True)
        nuevas = filas_caso(mask, spacing, modo=args.modo)
        for fila in nuevas:
            fila.update(case_id=caso.case_id, split=args.split if args.split != "all" else caso.split,
                        fracture_type=caso.fracture_type, modo=args.modo)
        filas.extend(nuevas)
        regiones = ",".join(sorted({f["region"] for f in nuevas})) or "sin fragmentos"
        print(f"[{i:2d}/{len(casos)}] {caso.case_id} ({caso.fracture_type}): {regiones} — {time.time() - t0:.0f}s",
              flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(salida, index=False)
    con = df[~df.principal]
    print(f"\nGuardado: {salida}  ({len(df)} filas, {len(casos)} casos, {time.time() - t0:.0f}s)")
    if len(con):
        print("Distancias de fragmentos conminutos por región (mm):")
        print(con.groupby("region")["distancia_mm"].describe()[["count", "min", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
