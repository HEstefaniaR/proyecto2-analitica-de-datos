"""Inferencia de los seis CT de la galería del informe (cinco peores y el mejor de test, por el Dice de la máscara
fusionada) con `model3d infer`.
Ejecutar desde la raíz: uv run python scripts/inferir_casos_test.py"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd

# Cinco CT de test con menor Dice de pelvis de la máscara fusionada, y el de mayor Dice
_df = pd.read_csv(ROOT / "results" / "detection" / "unet3d_cbam_trab" / "eval" / "cases_test.csv",
                  dtype={"case_id": str}).sort_values("fusion_pelvis_dice")
CASOS = _df.head(5).case_id.tolist() + [_df.iloc[-1].case_id]
print("Casos:", CASOS)

for cid in CASOS:
    rutas = [
        ROOT / "data" / carpeta / f"{cid}.mha"
        for carpeta in [
            "PENGWIN_CT_train_images_part1",
            "PENGWIN_CT_train_images_part2",
        ]
    ]
    ct = next((p for p in rutas if p.exists()), None)
    if ct is None:
        print(f"Falta el CT {cid}")
        continue
    destino = ROOT / "results" / "inferencia_informe" / cid
    if (destino / "mascara_fusion.nii.gz").exists():
        print(f"Caso {cid}: inferencia existente")
        continue
    print(f"Procesando CT {cid}", flush=True)
    subprocess.run([
        sys.executable, "-m", "src.models.model3d",
        "infer", str(ct),
        "--run", "unet3d_cbam_trab",
        "--out", str(destino),
        "--device", "auto"
    ], cwd=ROOT, check=True)

print("Inferencias finalizadas")
