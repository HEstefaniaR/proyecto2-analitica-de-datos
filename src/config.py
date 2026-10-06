"""Constantes compartidas por el EDA, el entrenamiento, la inferencia y el dashboard.

Es la única fuente de parámetros del proyecto: los notebooks, los .py y el dashboard los leen de aquí.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_DIRS = (
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
)
LABEL_DIR = DATA_ROOT / "PENGWIN_CT_train_labels"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
DETECTION_DIR = RESULTS_DIR / "detection"

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

# =====================================================================================
# Preprocesamiento (mismos valores en el EDA, el entrenamiento y la inferencia)
# =====================================================================================

# Orientación a la que se lleva todo volumen (código DICOM de SimpleITK). "LPS": el índice x crece
# hacia la izquierda del paciente, y hacia atrás y z hacia la cabeza.
TARGET_ORIENTATION = "LPS"

# Ventana HU [low, high]: se recorta a ese rango y se reescala linealmente a [0, 1].
HU_WINDOW = (50, 1300.0)

# Cubo de entrada: CUBE_MM mm por lado, vóxel isótropo, INPUT_SIZE vóxeles por lado
# (400 / 256 = 1,5625 mm). Centrado en el centro del FOV en x, y y en los cortes con hueso en z.
# Lo que queda fuera del CT se rellena con PAD_HU (aire); la máscara, con 0.
CUBE_MM = 400.0
INPUT_SIZE = 256
VOXEL_MM = CUBE_MM / INPUT_SIZE
PAD_HU = -1024

# Corte "con hueso" (sin usar la máscara, así que sirve en inferencia): al menos MIN_BONE_FRACTION
# de sus píxeles con HU >= BONE_HU. Se usa para centrar el cubo en z.
BONE_HU = 150
MIN_BONE_FRACTION = 0.002

# =====================================================================================
# Regiones (clases)
# =====================================================================================

# Regiones anatómicas en el orden de las salidas de la red. Códigos de fragmento de cada una:
# 1-10 sacro, 11-20 coxal izquierdo, 21-30 coxal derecho.
REGION_NAMES = ("sacrum", "left_hipbone", "right_hipbone")
REGION_CODES = {"sacrum": range(1, 11), "left_hipbone": range(11, 21), "right_hipbone": range(21, 31)}
REGION_COLORS = {"sacrum": "#e15759", "left_hipbone": "#4e79a7", "right_hipbone": "#59a14f",
                 "hipbone": "#9c6ade"}
REGION_LABELS = {"sacrum": "Sacro", "left_hipbone": "Coxal izq.", "right_hipbone": "Coxal der.", "hipbone": "Coxal"}

# Una región cuenta como presente en un corte (y genera caja) si ocupa al menos estos píxeles.
MIN_REGION_PX = 16

# =====================================================================================
# Vistas (estilo QuickNAT: un backbone compartido que recibe cortes de las tres vistas)
# =====================================================================================

VIEWS = ("axial", "coronal", "sagittal")
# Clases de cada vista. En un corte sagital el coxal izquierdo y el derecho se ven iguales: la vista
# sagital usa una sola clase "hipbone" y el lado se asigna por la posición del corte (como QuickNAT
# con los hemisferios).
VIEW_CLASSES = {
    "axial": REGION_NAMES,
    "coronal": REGION_NAMES,
    "sagittal": ("sacrum", "hipbone"),
}
# Peso de cada vista al fusionar en 3D (QuickNAT: 0,4 axial, 0,4 coronal, 0,2 sagital)
VIEW_WEIGHTS = {"axial": 0.4, "coronal": 0.4, "sagittal": 0.2}
# Entrada 2.5D: el corte y sus vecinos como canales
K_SLICES = 3

# =====================================================================================
# Modelo: backbone FundidoraFPN + CBAM con γ + cabeza con anclas
# =====================================================================================

WIDTHS = (32, 64, 128, 256)                 # canales de los 4 bloques del encoder (FundidoraPC)
BOTTLENECK_DILATIONS = (2, 4)
# CBAM solo en los bloques 3 y 4 del encoder (penúltima y última capa), según el profesor
CBAM_BLOCKS = (3, 4)
CBAM_REDUCTION = 8
CBAM_KERNEL = 7
# γ aprendible por capa con atención: y = x + γ · CBAM(x). Empieza en 0 (la atención no aporta) y
# la red aprende cuánto usarla; se registra en cada época.
GAMMA_INIT = 0.0
HEAD_ATTENTION = True                       # CBAM + γ también en la cabeza con anclas (RPN)

# Cabeza con anclas (como la RPN del taller 3) sobre p3
STRIDE = 8
HEAD_CHANNELS = 128
# Escalas = p25 / p50 / p75 de la raíz del área de las cajas de train en cada vista; proporciones
# ancho/alto ≈ p20 / p50 / p80. Calculadas con `python -m src.models.detection anchors`.
ANCHORS = {
    "axial": {"scales": (34, 46, 58), "ratios": (0.63, 0.86, 1.58)},
    "coronal": {"scales": (39, 60, 85), "ratios": (0.36, 0.68, 1.3)},
    "sagittal": {"scales": (35, 52, 71), "ratios": (0.52, 0.74, 1.03)},
}
POS_IOU, NEG_IOU = 0.5, 0.4                 # IoU >= 0,5 positiva; < 0,4 negativa; en medio se ignora
ANCHOR_BATCH, POS_FRACTION = 256, 0.5       # muestreo por imagen: hasta 256 anclas, máximo 1:1
LAMBDAS = {"cls": 1.0, "box": 1.0}          # L = λ_cls · BCE + λ_box · SmoothL1

# Postproceso
SCORE_THR = 0.05
NMS_IOU = 0.5
PRE_NMS = 100
MAX_DET = 1                                 # a lo sumo una caja por región y corte
PLOT_SCORE_THR = 0.3
FUSION_SCORE_THR = 0.5                      # cortes que cuentan para la caja 3D de cada vista

# =====================================================================================
# Entrenamiento
# =====================================================================================

RUN_CBAM = "tres_vistas_cbam"
RUN_NO_CBAM = "tres_vistas_sin_cbam"
CACHE_DIR = DATA_ROOT / f"cache_cubo_{int(CUBE_MM)}mm_{INPUT_SIZE}"

EPOCHS = 35
SAMPLES_PER_VIEW = 6000                     # cortes por vista y época (18 000 por época)
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-2                         # AdamW (decay desacoplado); no se aplica a γ
WARMUP_EPOCHS = 1                           # el lr sube linealmente de ~0 a LR durante estas épocas
EARLY_STOP_PATIENCE = 10                    # para si val mAP50-95 no mejora en estas épocas
EARLY_STOP_START = 18                       # la parada temprana solo puede actuar desde esta época (lr ya más bajo)
NEG_FRAC = 0.15                             # fracción de cortes sin regiones por época
GRAD_CLIP = 10.0
AMP = True                                  # CUDA: float16 + GradScaler (torch.cuda.amp); MPS: bfloat16 (no se desborda)
WORKERS = 4                                 # procesos de carga del entrenamiento (persistentes)
EVAL_WORKERS = 0                            # evaluación en el proceso principal (lee del memmap)
VAL_STRIDE = 3                              # durante el entrenamiento se valida con 1 de cada 3 cortes
TRAIN_EVAL_SLICES = 600                     # cortes fijos de train por vista para medir métricas de train

# Aumento de datos (solo en entrenamiento). Sin volteo horizontal: cambiaría el lado del coxal.
AUG_ROTATION_DEG = 15
AUG_SCALE = 0.15
AUG_SHIFT = 0.08
AUG_CONTRAST = 0.20
AUG_BRIGHTNESS = 0.05
AUG_GAMMA = 1.4                             # corrección gamma de intensidad entre 1/1,4 (≈ 0,7) y 1,4
# Degradación (simula CT de peor calidad); cada una se aplica a un corte con probabilidad AUG_DEGRADE_PROB
AUG_DEGRADE_PROB = 0.3
AUG_NOISE_STD = 0.03                        # ruido gaussiano: desviación máxima (imagen en [0, 1])
AUG_BLUR_SIGMA = (0.5, 1.5)                 # desenfoque gaussiano: sigma en píxeles
AUG_LOWRES_FACTOR = (1.5, 3.0)              # baja resolución: se reduce por este factor y se vuelve a 256

# Prueba de overfit
OVERFIT_SLICES = 8                          # por vista
OVERFIT_ITERS = 500
OVERFIT_LR = 1e-3

# =====================================================================================
# Evaluación
# =====================================================================================

CAM_SAMPLES = 256                           # cortes por vista para medir Grad-CAM
CAM_LAYERS = ("p3", "p4")
LATENCY_RUNS, LATENCY_WARMUP = 50, 10
# Prueba de reconstrucción del extractor (conv 9×9, stride 6, ≤ 15 canales)
PROBE_LAYER = "p2"
PROBE_BOTTLENECK = 15
PROBE_KERNEL, PROBE_STRIDE = 9, 6
PROBE_ITERS = 1500
PROBE_LR = 3e-3
PROBE_BONE_THR = 0.1                        # píxel de hueso: intensidad > 0,1

# Objetivos de la rúbrica (detección)
TARGETS = {"mAP50": 0.80, "mAP50_95": 0.60, "mIoU": 0.75, "macro_f1": 0.90}

# =====================================================================================
# Modelo 3D: backbone Fundidora 3D (encoder + decoder U-Net hasta P2) + detección 2D por corte
# (C.cambios/propuesta_red_3d.md). Por ahora solo detección; la segmentación aún no se hace.
# Un paso = un CT completo. Pérdida y anclas: las del 2.5D (LAMBDAS, ANCHORS).
# =====================================================================================

M3D_RUN_CBAM = "unet3d_cbam"
M3D_RUN_NO_CBAM = "unet3d_sin_cbam"
M3D_INPUT = 256                             # lado del cubo que entra a la red (se reduce del caché de 256³ si es menor)
M3D_WIDTHS = (16, 32, 64, 128)              # canales de los 4 bloques del encoder FundidoraPC 3D
M3D_NECK = 256                              # canales del cuello dilatado (BOTTLENECK_DILATIONS)
M3D_DECODER = (64, 32)                      # canales de P3 (1/8) y P2 (1/4)
M3D_CBAM_BLOCKS = (3, 4)                    # CBAM 3D + γ en los bloques 3 y 4 (antes de la bifurcación)
M3D_CHECKPOINT = True                       # gradient checkpointing en los niveles de alta resolución (GPU de 6 GB)
# Detección 2D por corte: cortes de P3 y P2 interpolados en la posición del corte, llevados a la grilla
# de 32 × 32 (STRIDE 8) y concatenados; cabeza con anclas por vista (AnchorHead del 2.5D).
M3D_DET_SLICES = 32                         # cortes por vista y paso (con la mezcla NEG_FRAC)
M3D_EPOCHS = 70                             # 70 CT por época; se puede seguir con --resume --epochs N
M3D_LR = 1e-3
M3D_WEIGHT_DECAY = 1e-2
M3D_WARMUP_EPOCHS = 3
M3D_VAL_EVERY = 5                           # validar cada N épocas
M3D_VAL_STRIDE = 3                          # en la validación, 1 de cada 3 cortes por vista (como el 2.5D)
M3D_EARLY_STOP_PATIENCE = 4                 # validaciones sin mejorar val mAP50-95 (= 20 épocas)
M3D_EARLY_STOP_START = 40                   # la parada temprana solo actúa desde esta época
M3D_WORKERS = 2
M3D_AUG_ROTATION_DEG = 10                   # rotación 3D (cada eje); el resto del aumento usa AUG_*
M3D_OVERFIT_CASES = 2
M3D_OVERFIT_ITERS = 300

# =====================================================================================
# EDA
# =====================================================================================

EDA_RECOMPUTE = True                        # True: recorre los 100 volúmenes; False: lee las tablas de EDA_DIR
EDA_CASE_ID = "004"                         # caso de las secciones de un solo caso
EDA_EXAMPLE_CASES = ("001", "002", "095", "023")
EDA_HU_HIST_RANGE = (-3100, 4200)           # rango del histograma HU; fuera se acumula en los extremos
ANISOTROPY_THRESHOLD = 1.5                  # razón máx/mín del espaciado a partir de la cual se marca
