"""Constantes compartidas por el EDA, el entrenamiento y la inferencia."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_DIRS = (
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
)
LABEL_DIR = DATA_ROOT / "PENGWIN_CT_train_labels"
SPLITS_DIR = PROJECT_ROOT / "splits"
EDA_DIR = PROJECT_ROOT / "results" / "eda"

# Semilla fija
SEED = 99

# Proporciones train / val / test, por caso.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Estratos de los splits
SPLIT_STRATA = {
    "sacrum_fractured": ("SF", "SHF"),
    "hip_unilateral_or_none": ("UHF", "PRD"),
    "hip_bilateral": ("BHF",),
}

# --- Preprocesamiento (mismos valores en el EDA, el entrenamiento y la inferencia) ---

# Orientación a la que se lleva todo volumen (código DICOM de SimpleITK). "LPS" es la dirección
TARGET_ORIENTATION = "LPS"

# Ventana HU [low, high]: se recorta a ese rango y se reescala linealmente a [0, 1].
HU_WINDOW = (200, 1300.0)

# Recorte en el plano axial: tamaño físico (x, y) en mm, centrado en el volumen. Si el volumen es
# más pequeño se rellena (imagen con PAD_HU, máscara con 0). El eje z no se recorta.
CROP_MM = (350.0, 350.0)
PAD_HU = -1024
