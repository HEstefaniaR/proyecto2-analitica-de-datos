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
# Modelo 3D multitarea:
# backbone Fundidora 3D + detección class-agnostic por corte +
# segmentación fondo/hueso/borde + clasificación anatómica por instancia.
# El decoder U-Net recupera resolución hasta P0.
# =====================================================================================

M3D_RUN_CBAM = "unet3d_cbam_trab"          # D-87: CBAM en enc4, P3 y detección + canal trabecular
M3D_RUN_SE = "unet3d_se_trab"               # P-19: igual, con solo atención de canal (squeeze-excitation)
M3D_RUN_NO_CBAM = "unet3d_sin_atencion"
M3D_RUN_PREV = "unet3d_cbam"                # modelo entregado en el avance de la semana 10 (CBAM en enc3 y enc4, 1 canal)
M3D_ATTENTION = "cbam"                      # "cbam" | "se" | "none" (`model3d train --attention`)
M3D_TRAB_WINDOW = (100.0, 700.0)            # D-83: segundo canal de entrada, ventana trabecular (HU); None = 1 canal
M3D_DECODER_ATTENTION = ("p3",)             # D-84: atención en la primera capa del decoder (P3)
# D-90 (colapso del 2026-10-09: P0 con ReLU muertas en sacro y coxal derecho). LeakyReLU en el decoder (no hay
# neuronas muertas) y supervisión profunda: una segunda salida semántica sobre P1 (128³) con su propia pérdida.
M3D_DECODER_ACT = "leaky"                   # "leaky" | "relu" (modelos anteriores)
M3D_LEAKY_SLOPE = 0.01
M3D_SEG_AUX = True                          # salida semántica auxiliar sobre P1 (solo entrenamiento)
M3D_SEG_AUX_WEIGHT = 0.5                    # L_seg = L(P0) + 0,5 · L(P1)
M3D_MIN_BONE_DICE = 0.5                     # control: el overfit y la validación (desde la época 10) avisan si algún
                                            # hueso queda por debajo (colapso de una clase)

# Cabeza de máscara tipo Mask R-CNN (D-82, D-88): para cada caja axial, ROI Align de P2 y P1 (cortes axiales) y del CT
# a doble resolución en el plano; rama de máscara (una por clase) y rama de clase (la cabeza de clasificación).
M3D_MASK_HEAD = True
M3D_HIRES_FACTOR = 2                        # CT y máscara a 400/512 = 0,78 mm por píxel en el plano (mismos 256 cortes)
M3D_HIRES_DIR = DATA_ROOT / f"cache_cubo_{int(CUBE_MM)}mm_{INPUT_SIZE}_axial{INPUT_SIZE * M3D_HIRES_FACTOR}"
M3D_ROI_FEAT = 48                           # lado del ROI Align de las características (P2 y P1)
M3D_MASK_SIZE = 96                          # lado de la máscara de cada caja (ROI Align del CT a 0,78 mm)
M3D_MASK_CHANNELS = 64                      # canales de la rama de máscara
M3D_MASK_DETACH = True                      # D-88: la pérdida de la cabeza no llega al backbone (el detector no cambia)
M3D_MASK_JITTER = 0.10                      # entrenamiento: cajas reales movidas/escaladas hasta ±10 % de su lado
M3D_MASK_SCORE_THR = 0.5                    # inferencia: cajas con puntaje ≥ este valor pasan a la cabeza de máscara
M3D_MASK_THR = 0.5                          # probabilidad mínima de la máscara para marcar el píxel
M3D_MASK_CHUNK = 32                         # cortes por lote al aplicar la cabeza en inferencia
M3D_FUSION_MODE = "union_sacro"             # D-93: postproceso que une las dos salidas (`model3d.fuse_heads`); elegido en validación
M3D_INPUT = 256                             # lado del cubo que entra a la red (se reduce del caché de 256³ si es menor)
M3D_WIDTHS = (16, 32, 64, 128)              # canales de los 4 bloques del encoder FundidoraPC 3D
M3D_NECK = 128                              # D-70: cuello de 2 conv dilatadas (BOTTLENECK_DILATIONS) sin subir canales
                                            # (estructura de la clase 9.2 + dilatación; antes 256 = 70 % de los parámetros)
M3D_DECODER = (64, 32, 16, 8)               # canales de P3 (1/8), P2 (1/4), P1 (1/2) y P0 (1/1); P0 con 16
                                            # no cabe en 6 GB (D-65)

# D-81 (asesoría del 2026-10-08): una sola cabeza de segmentación con tres máscaras, sacro, coxal izquierdo y coxal
# derecho (más el fondo). Cada hueso es una instancia con TODOS sus fragmentos: el profesor pidió no separar los
# fragmentos pequeños de su hueso ("si quedaban separados, tocaba unirlos"). El Dice se mide por hueso y su macro.
# Inferencia: etapa 1 = regiones detectadas (cajas axiales); etapa 2 = máscara de cada hueso dentro de ellas.
M3D_SEM_CLASSES = 4
M3D_SEG_CE_WEIGHT = 0.5                     # pérdida de segmentación = 0,5 · CE + 0,5 · Dice multiclase, sin peso por
M3D_SEG_DICE_WEIGHT = 0.5                   # clase ("Dice + entropía cruzada: perfecto", asesoría 2026-10-08)
M3D_RESTRICT_TO_DETECTION = True            # etapa 1: solo el hueso dentro de las regiones detectadas (cajas axiales)

# Cabeza de clasificación: sacro, coxal izquierdo y coxal derecho.
M3D_CLS_CLASSES = 3
M3D_CLS_HIDDEN = 64

# Detección 3D por corte: una salida de objectness, sin clase anatómica.
M3D_DET_CLASSES = 1
M3D_DET_VIEWS = ("axial",)                 # D-78: solo la cabeza axial (la única que usa el pipeline: las regiones 3D
                                            # salen de ella; la coronal era la peor). Antes: axial, coronal y sagital.
M3D_MAX_DET = 3
# Reconstrucción de regiones e instancias durante inferencia.
M3D_TRACK_IOU = 0.3
M3D_MIN_INSTANCE_VOXELS = 50
M3D_MIN_FRAGMENT_MM3 = 50 * VOXEL_MM ** 3        # fragmento mínimo para la distancia de separación (≈ 191 mm³ =
                                            # 50 vóxeles del cubo), medido en el espacio original del CT
M3D_CONNECTIVITY = 6                        # D-71: vecindad de los componentes conexos 3D (6, 18 o 26)

# Pesos λ de la pérdida multitarea: L = λ_det · (BCE obj + SmoothL1 caja) + λ_seg · L_seg + λ_anat · CE, con
# L_seg = 0,5 · CE + 0,5 · Dice de los tres huesos (D-81).
# λ_t = prioridad_t / pérdida inicial_t: al inicio cada tarea aporta a la pérdida total en la proporción de su
# prioridad, sin importar la escala natural de su pérdida (Dice + BCE ≈ 2, CE con 3 clases ≈ ln 3 ≈ 1,1).
# Prioridades: segmentación > detección = clasificación (la segmentación de instancia es la tarea más difícil y
# de ella dependen la clasificación de fragmentos y la distancia de separación).
M3D_LOSS_PRIORITY = {"det": 1.5, "anat": 1.5, "seg": 2.0, "mask": 2.0}   # D-67: detección igualada a clasificación
# Pérdida media de cada tarea con el modelo sin entrenar (semilla fija), medida con
# `uv run python -m src.models.model3d calibrate` (salida en results/detection/lambda_calibration.json).
M3D_CALIBRATION_CASES = 10                  # CT de train usados para medirla
# Dentro de la detección (D-72), como en la clase 6.2: la caja se pondera para quedar en la escala del objectness,
# λ_box = pérdida inicial de objectness / pérdida inicial de caja. L_det = BCE_obj + λ_box · SmoothL1.
M3D_DET_INIT = {"obj": 0.6924, "box": 0.0932}     # medido 2026-10-09 (D-90: decoder LeakyReLU + salida auxiliar en P1) → λ_box 7,43
M3D_BOX_WEIGHT = M3D_DET_INIT["obj"] / M3D_DET_INIT["box"]
M3D_LOSS_INIT = {"det": 1.3847, "anat": 1.1029, "seg": 1.6201, "mask": 1.3477}   # medido 2026-10-09 (seg incluye 0,5 · auxiliar); vale para CBAM y SE (γ = 0 al inicio)
M3D_LOSS_WEIGHTS = {t: M3D_LOSS_PRIORITY[t] / M3D_LOSS_INIT[t] for t in M3D_LOSS_PRIORITY}
M3D_CBAM_BLOCKS = (4,)                      # D-87: atención + γ solo en el bloque 4 (el 3 terminó con γ ≈ 0)
M3D_CHECKPOINT = True                       # gradient checkpointing en los niveles de alta resolución (GPU de 6 GB)
# Detección 2D por corte: cortes de P3 y P2 interpolados en la posición del corte, llevados a la grilla
# de 32 × 32 (STRIDE 8) y concatenados; cabeza con anclas por vista (AnchorHead del 2.5D).
M3D_DET_SLICES = 32                         # cortes por vista y paso (con la mezcla NEG_FRAC)
M3D_EPOCHS = 140                            # la usuaria pidió 140 (2026-10-06); --resume --epochs N para seguir
M3D_LR = 1e-3
M3D_WEIGHT_DECAY = 1e-2
M3D_VAL_EVERY = 5                           # validar cada N épocas
M3D_VAL_STRIDE = 3                          # en la validación, 1 de cada 3 cortes por vista (como el 2.5D)
M3D_TRAIN_EVAL_CASES = 15                   # CT fijos de train (sin aumento) para las métricas de entrenamiento
# best.pth es la época con mayor promedio de una métrica de validación por cabeza (D-66):
# detección mAP50-95, segmentación Dice y clasificación macro F1.
M3D_SELECT_METRICS = ("val_mAP50_95", "val_dice_sem", "val_dice_mask", "val_macro_f1")   # D-81: Dice macro de los tres huesos
M3D_WORKERS = 2
M3D_AUG_ROTATION_DEG = 10                   # rotación 3D (cada eje); el resto del aumento usa AUG_*
M3D_OVERFIT_CASES = 2
M3D_OVERFIT_ITERS = 300

# Baseline SAM zero-shot con el modelo 3D (rúbrica 4.2; `scripts/run_sam_baseline_3d.py`)
SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"   # ViT-B oficial de Meta (no va al repo)
M3D_SAM_SCORE_THR = 0.5                     # cajas axiales del modelo con puntaje ≥ este valor se usan como prompt

# Evaluación final del modelo 3D (`model3d evaluate`, salida en results/detection/<run>/eval/)
M3D_EVAL_SPLITS = ("val", "test")
M3D_EVAL_STRIDE = 1                         # todos los cortes de cada vista (la validación del entrenamiento usa 3)
M3D_REFINE_MARGINS = (0, 2, 4, 6, 8)        # px: ajuste de cada caja al hueso segmentado dentro de la caja ampliada;
                                            # el margen se elige en val y se aplica tal cual en test
M3D_EVAL_EXAMPLES = 3                       # CT de test que se guardan para las figuras cualitativas
M3D_LATENCY_RUNS_GPU, M3D_LATENCY_RUNS_CPU = 5, 2   # repeticiones (tras 1 de calentamiento) del pipeline completo

# =====================================================================================
# EDA
# =====================================================================================

EDA_RECOMPUTE = True                        # True: recorre los 100 volúmenes; False: lee las tablas de EDA_DIR
EDA_CASE_ID = "004"                         # caso de las secciones de un solo caso
EDA_EXAMPLE_CASES = ("001", "002", "095", "023")
EDA_HU_HIST_RANGE = (-3100, 4200)           # rango del histograma HU; fuera se acumula en los extremos
ANISOTROPY_THRESHOLD = 1.5                  # razón máx/mín del espaciado a partir de la cual se marca

# =====================================================================================
# Dashboard (visualizador crudo; independiente de la ventana del modelo)
# =====================================================================================

DASH_HU_RANGE = (100, 600)                  # rango de hueso del visualizador crudo (lo, hi) en HU
DASH_HU_CLIP_MAX = 1300
DASH_MESH_STEP = 3                          # step_size de marching_cubes; menor = malla más detallada
DASH_MIN_COMPONENT_MM3 = 1000               # descarta estructuras umbralizadas más pequeñas
DASH_BONE_COLOR = "#e3d5b8"
DASH_SLICE_WINDOW = (int(HU_WINDOW[0]), int(HU_WINDOW[1]))   # por defecto, la ventana del modelo
# Run cuyo `best.pth` usa el dashboard como modelo de producción (una línea para cambiarlo).
DASH_MODEL_RUN = M3D_RUN_CBAM
# Salida de segmentación que muestra el dashboard: "fusion" (D-93), "mask" (cabeza de máscara) o "semantic".
# Los modelos sin cabeza de máscara solo tienen "semantic".
DASH_OUTPUT = "fusion"
