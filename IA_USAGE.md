# IA_USAGE · Uso de IA generativa en el proyecto PENGWIN

**Proyecto:** detección, segmentación y clasificación de fracturas pélvicas en CT.

**Modelos de IA generativa usados:**

- **Claude Opus 5.5** (Anthropic), con Claude Code: asistente de programación y de análisis. Escribía código y scripts, corría pruebas y evaluaciones y, antes de cada cambio de diseño, presentaba opciones; la decisión la tomaba el equipo.
- **ChatGPT** (OpenAI): consultas puntuales sobre valores de partida de hiperparámetros y sobre cómo implementar componentes.

**Cómo leer este archivo:**

- Cada entrada tiene el prompt, lo que respondió o propuso la IA, el código final y la decisión del equipo con su análisis crítico.
- Las entradas están en **orden cronológico**, según la fecha en que la decisión quedó en el código (registro de decisiones del proyecto). Por eso Claude y ChatGPT aparecen intercalados.
- Los prompts de Claude están resumidos por tema, con las decisiones que tomó el equipo. Los de ChatGPT son los que registró el equipo.

## Resumen

| # | Fecha | Herramienta | Tema |
|---|---|---|---|
| 1 | 2026-09-24 a 09-25 | Claude | División de los datos y EDA |
| 2 | 2026-10-01 a 10-02 | ChatGPT | Detección — escalas y proporciones de las anclas |
| 3 | 2026-10-01 a 10-02 | ChatGPT | Detección — umbral de NMS y filtrado de cajas |
| 4 | 2026-10-01 a 10-02 | ChatGPT | Detección — umbrales de asignación positiva y negativa |
| 5 | 2026-10-03 a 10-07 | Claude | Modelo 3D multitarea |
| 6 | 2026-10-06 a 10-07 | ChatGPT | Multitarea — pesos λ de las pérdidas |
| 7 | 2026-10-06 a 10-07 | Claude | Pérdida multitarea |
| 8 | 2026-10-07 | ChatGPT | Clasificación — tamaño de la capa oculta y dropout |
| 9 | 2026-10-07 | Claude | Evaluación y notebooks |
| 10 | 2026-10-08 | ChatGPT | Detección 3D — qué vistas conservar |
| 11 | 2026-10-08 | ChatGPT | Segmentación — combinación de pérdidas |
| 12 | 2026-10-09 | ChatGPT | Incorporación de una cabeza de máscaras por ROI |
| 13 | 2026-10-09 | ChatGPT | Incorporación de una segunda ventana HU para mejorar el contraste óseo |
| 14 | 2026-10-09 | Claude | Cambio del modelo 3D inicial |
| 15 | 2026-10-09 | ChatGPT | Corrección del decoder para evitar el colapso de clases durante la segmentación |
| 16 | 2026-10-10 a 10-11 | Claude | Resultados finales, fusión, inferencia y verificación |

## 1. División de los datos y EDA

**Herramienta:** Claude · **Fecha de la decisión:** 2026-09-24 a 09-25

**Prompt a Claude:** «Necesito que la división train / validación / test no dependa del notebook del EDA. Hazla en un script aparte que se corra una sola vez, por caso, 70 / 15 / 15, estratificada por tipo de fractura y con semilla fija, y que guarde un archivo. El notebook solo debe leer ese archivo para mostrar cómo quedaron las clases. La ventana HU es de 50 a 1300, según la fuente de radiología que te paso.»

**Respuesta:** El script `src/dataset/splits.py` y la propuesta de agrupar los cinco tipos de fractura en tres estratos (sacro fracturado, coxal unilateral o sin fractura, coxal bilateral). También propuso una ventana HU de 200 a 1500.

**Código final:**

```python
# src/config.py
SEED = 99
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_STRATA = {
    "sacrum_fractured": ("SF", "SHF"),
    "hip_unilateral_or_none": ("UHF", "PRD"),
    "hip_bilateral": ("BHF",),
}
HU_WINDOW = (50, 1300.0)
```

**Decisión del equipo y comparación:** Aceptamos los tres estratos, porque SF (3 casos) y PRD (1 caso) no alcanzaban para repartirse solos en tres conjuntos. Rechazamos la ventana de 200 a 1500 HU y usamos la de una fuente de radiología (50 a 1300): el EDA mostró que el percentil 5 del hueso etiquetado es 21 HU, así que empezar en 200 dejaba fuera hueso poco denso. **Acierto:** la división quedó fija en `splits/splits.csv` y se reproduce igual con la semilla 99. **Error de la IA:** la ventana propuesta no correspondía a los datos.

## 2. Detección — escalas y proporciones de las anclas

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-01 a 10-02

**Pregunta a ChatGPT:** ¿Qué escalas y aspect ratios puedo usar para las anclas de la cabeza de detección en cortes de CT? ¿Sirve usar las mismas para las tres vistas?

**Respuesta:** Utilizar un juego genérico de tres escalas y tres proporciones por vista, como punto de partida:

```python
# Propuesta 
ANCHORS = {view: {"scales": (32, 64, 96), "ratios": (0.5, 1.0, 2.0)}
           for view in ("axial", "coronal", "sagittal")}
```

**Cambio reflejado en el código final** (`src/config.py`, líneas 104–113):

```python
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
```

**Decisión del equipo y comparación:** No nos convenía dejar las mismas anclas para todas las vistas porque las cajas de los huesos cambian de forma dependiendo del plano. En vez de asumir tamaños estándar, hicimos un cálculo a partir de cuantiles de las cajas de entrenamiento. Así las propuestas quedan más adaptadas al dataset. **Acierto:** las anclas representan mejor la geometría observada utilizando el cambio que hicimos. **Limitación:** no tenemos aquí una ablación con las anclas genéricas, así que no afirmamos cuánto mejoró el mAP por este cambio aislado.

## 3. Detección — umbral de NMS y filtrado de cajas

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-01 a 10-02

**Pregunta a ChatGPT:** ¿Qué IoU y confianza puedo usar para NMS sin repetir cajas del mismo hueso?

**Respuesta:** Empezar con NMS relativamente permisivo, por ejemplo `0.7`, y un umbral de confianza `0.5`.

```python
# Propuesta
SCORE_THR = 0.5
NMS_IOU = 0.7
PRE_NMS = 200
```

**Código final** (`src/config.py`, líneas 118–124):

```python
# Postproceso
SCORE_THR = 0.05
NMS_IOU = 0.5
PRE_NMS = 100
MAX_DET = 1                                 # a lo sumo una caja por región y corte
PLOT_SCORE_THR = 0.3
FUSION_SCORE_THR = 0.5                      # cortes que cuentan para la caja 3D de cada vista
```

**Decisión del equipo y comparación:** Bajamos el IoU de NMS a `0.5` y conservamos más candidatos al principio con `SCORE_THR = 0.05`, antes de aplicar los filtros posteriores. Nos interesaba no perder cajas potencialmente útiles solo por estar por debajo de una confianza alta demasiado pronto. Además, el detector 2D establece una caja como máximo por región y corte. **Acierto:** el flujo distingue selección inicial, eliminación de duplicados y umbrales para visualización/fusión. **Riesgo:** una confianza tan baja puede aumentar falsos positivos y carga de postproceso. No es correcto atribuir la métrica final exclusivamente a NMS.

## 4. Detección — umbrales de asignación positiva y negativa

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-01 a 10-02

**Pregunta a ChatGPT:** ¿Cómo marco anclas positivas y negativas cuando algunas se solapan parcialmente con la caja real?

**Respuesta:** Usar un criterio más estricto para positivos y negativos, como `0.7` y `0.3`.

```python
# Propuesta 
POS_IOU, NEG_IOU = 0.7, 0.3
ANCHOR_BATCH, POS_FRACTION = 256, 0.25
```

**Código final** (`src/config.py`, líneas 114–116):

```python
POS_IOU, NEG_IOU = 0.5, 0.4                 # IoU >= 0,5 positiva; < 0,4 negativa; en medio se ignora
ANCHOR_BATCH, POS_FRACTION = 256, 0.5       # muestreo por imagen: hasta 256 anclas, máximo 1:1
LAMBDAS = {"cls": 1.0, "box": 1.0}          # L = λ_cls · BCE + λ_box · SmoothL1
```

**Decisión del equipo y comparación:** Nos parecía más razonable permitir positivos desde `0.5`, porque exigir `0.7` en un problema con estructuras irregulares podía dejar pocas anclas útiles, especialmente en regiones difíciles. Dejamos un intervalo de IoU que se ignora en vez de forzar etiquetas dudosas. También se configuró un máximo de mitad de positivos por minibatch de anclas. **Limitación:** aceptar positivos más débiles puede introducir más ruido; debería justificarse con conteos de anclas y validación, no solo por intuición.

## 5. Modelo 3D multitarea

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-03 a 10-07

**Prompt a Claude:** «Voy a intentar un backbone 3D con detección 2D por corte, entrenando en una RTX 3060 de 6 GB. Usa el siguiente backbone: U-Net 3D con un encoder de cuatro bloques (16, 32, 64 y 128 canales, con InstanceNorm), un cuello con dos convoluciones dilatadas y un decoder que vuelve a la resolución completa, con sus cabezas de segmentación y de clasificación. La detección sigue siendo la nuestra, por corte, con anclas y NMS propio.  Entrena 140 épocas.»

**Respuesta:** La implementación en `src/models/model3d.py`, gradient checkpointing y precisión mixta para que el modelo cupiera en 6 GB, tres tamaños posibles para el cuello de botella, y parada temprana y calentamiento de la tasa de aprendizaje.

**Código final:**

```python
# src/config.py
M3D_WIDTHS = (16, 32, 64, 128)
M3D_NECK = 128
M3D_DECODER = (64, 32, 16, 8)
M3D_CHECKPOINT = True                       # gradient checkpointing en los niveles de alta resolución
M3D_EPOCHS = 140
```

**Decisión del equipo y comparación:** Usamos gradient checkpointing y precisión mixta: sin ellos el modelo no cabía en la tarjeta. De los tres tamaños del cuello elegimos el intermedio (128 canales), porque con 256 el cuello tenía el 70 % de los parámetros. Descartamos la parada temprana y el calentamiento: las curvas no mostraban sobreajuste que cortar y se guarda la mejor época en validación. **Acierto:** el modelo completo entrenó en 6 GB (pico de 3,9 GB). **Limitación:** el lote quedó en un CT por paso, por debajo del lote de 8 a 16 que recomienda la rúbrica.

## 6. Multitarea — pesos λ de las pérdidas

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-06 a 10-07

**Pregunta:** ¿Cómo puedo sumar las pérdidas de detección, segmentación, máscara y clasificación, qué pesos les pongo?

**Respuesta sugerida, ejemplo inicial:** Tomar una suma uniforme como baseline:

```python
# Propuesta 
loss_weights = {"det": 1.0, "anat": 1.0, "seg": 1.0, "mask": 1.0}
```

**Código final** (`src/config.py`, líneas 255–270):

```python
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
```

**Decisión del equipo y comparación:** No usamos los cuatro pesos iguales sin revisar las magnitudes de cada término. Preferimos dar prioridad a segmentación y máscara (`2.0`), y repartir la contribución inicial según las pérdidas medidas en diez casos de entrenamiento. Esto evita que una tarea domine la optimización solo porque su pérdida es numéricamente más grande. **Acierto:** hay calibración explícita y reproducible. **Limitación:** equilibrar contribuciones iniciales no garantiza que el balance siga siendo óptimo durante todas las épocas.

## 7. Pérdida multitarea

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-06 a 10-07

**Prompt a Claude:** «Los λ de la pérdida no pueden quedarse en 1. Lo ideal es darle más peso a la segmentación, y que la detección y la clasificación pesen igual.»

**Respuesta:** Calcular cada λ como la prioridad de la tarea dividida entre su pérdida inicial, medida con el modelo sin entrenar, y el comando `python -m src.models.model3d calibrate` para medirla.

**Código final:**

```python
# src/config.py
M3D_LOSS_PRIORITY = {"det": 1.5, "anat": 1.5, "seg": 2.0, "mask": 2.0}
M3D_LOSS_INIT = {"det": 1.3847, "anat": 1.1029, "seg": 1.6201, "mask": 1.3477}
M3D_LOSS_WEIGHTS = {t: M3D_LOSS_PRIORITY[t] / M3D_LOSS_INIT[t] for t in M3D_LOSS_PRIORITY}
```

**Decisión del equipo y comparación:** Aceptamos la regla y fijamos nosotras las prioridades: segmentación 2, detección 1,5 y clasificación 1,5 (después se agregó la máscara con 2). **Acierto:** cada tarea arranca aportando su prioridad sin depender de la escala de su pérdida. **Limitación:** las pérdidas de máscara y de clasificación no le mandan gradiente al backbone, así que para el backbone solo compiten la detección y la segmentación; y el balance inicial no garantiza que siga siendo el mejor durante todo el entrenamiento.

## 8. Clasificación — tamaño de la capa oculta y dropout

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-07

**Pregunta a ChatGPT:** ¿Cómo estructuro una cabeza pequeña que clasifique cada instancia ósea en tres clases, evitando sobreajuste?

**Respuesta:** Probar una capa oculta más grande y dropout relativamente alto:

```python
# Propuesta 
hidden_channels = 128
nn.Dropout(p=0.5)
num_classes = 3
```

**Código final** (`src/models/classification.py`, líneas 28–56, y `src/config.py`, líneas 239–241):

```python
class ClassificationHead3D(nn.Module):
    """Clasifica el vector de características de una instancia."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                hidden_channels,
                num_classes,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(features)

# Cabeza de clasificación: sacro, coxal izquierdo y coxal derecho.
M3D_CLS_CLASSES = 3
M3D_CLS_HIDDEN = 64
```

**Decisión del equipo y comparación:** Nos quedamos con una cabeza más sencilla: 64 unidades ocultas y dropout de `0.2`. La idea fue no hacer innecesariamente pesada esta parte, porque el backbone ya extrae la mayoría de características. Un dropout de `0.5` podía borrar demasiada información de vectores de instancia pequeños. **Acierto:** menos parámetros y regularización moderada. **Limitación:** no encontramos evidencia de una corrida controlada de `128/0.5` contra `64/0.2`; la conveniencia técnica no equivale a una mejora de F1 demostrada.

## 9. Evaluación y notebooks

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-07

**Prompt a Claude:** «Haz la evaluación final del mejor modelo en validación y test, con las métricas de la rúbrica. Cada curva debe comparar entrenamiento contra validación. Agrega HD95 y ASSD para cada cabeza de segmentación, igual que Dice e IoU.»

**Respuesta:** El comando `python -m src.models.model3d evaluate` y los notebooks 02 y 03. La primera versión del notebook comparaba métricas distintas y solo en validación.

**Código final:**

```bash
uv run python -m src.models.model3d evaluate --run unet3d_cbam_trab
```

**Decisión del equipo y comparación:** Rechazamos la primera versión del notebook: al comparar métricas distintas solo en validación no dejaba ver si había sobreajuste. Aceptamos la segunda, donde cada gráfica compara la misma cantidad en entrenamiento y validación. **Acierto:** HD95 y ASSD quedaron para todas las salidas de segmentación. **Error de la IA:** la primera versión no respondía a lo que pedimos.

## 10. Detección 3D — qué vistas conservar

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-08

**Pregunta a ChatGPT:** ¿Conviene detectar sobre las vistas axial, coronal y sagital del cubo CT para tener más información?

**Respuesta:** Mantener las tres vistas y combinar sus propuestas:

```python
# Propuesta
M3D_DET_VIEWS = ("axial", "coronal", "sagittal")
```

**Código final** (`src/config.py`, líneas 243–250):

```python
# Detección 3D por corte: una salida de objectness, sin clase anatómica.
M3D_DET_CLASSES = 1
M3D_DET_VIEWS = ("axial",)                 # D-78: solo la cabeza axial (la única que usa el pipeline: las regiones 3D
                                            # salen de ella; la coronal era la peor). Antes: axial, coronal y sagital.
M3D_MAX_DET = 3
# Reconstrucción de regiones e instancias durante inferencia.
M3D_TRACK_IOU = 0.3
M3D_MIN_INSTANCE_VOXELS = 50
```

**Decisión del equipo y comparación:** Al revisar el pipeline vimos que las regiones 3D se construían a partir de cajas axiales, y el propio registro de configuración señala que la vista coronal era la peor. Por eso conservamos solo la axial en el modelo 3D. **Acierto:** se reducen salidas que no se estaban aprovechando y el sistema resulta más coherente. **Limitación:** perder otras orientaciones puede perjudicar ciertos fragmentos o casos difíciles. Este cambio no significa que una sola vista sea mejor en cualquier dataset.

## 11. Segmentación — combinación de pérdidas

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-08

**Pregunta a ChatGPT:** ¿Para distinguir fondo, sacro, coxal izquierdo y coxal derecho es mejor usar solo Cross Entropy o combinarla con Dice?

**Respuesta:** Comenzar dando mayor prioridad a Cross Entropy:

```python
# Propuesta inicial ilustrativa
M3D_SEG_CE_WEIGHT = 0.8
M3D_SEG_DICE_WEIGHT = 0.2
```

**Código final** (`src/config.py`, líneas 230–237, y `src/models/segmentation.py`, líneas 60–70):

```python
M3D_SEM_CLASSES = 4
M3D_SEG_CE_WEIGHT = 0.5                     # pérdida de segmentación = 0,5 · CE + 0,5 · Dice multiclase, sin peso por
M3D_SEG_DICE_WEIGHT = 0.5                   # clase ("Dice + entropía cruzada: perfecto", asesoría 2026-10-08)
M3D_RESTRICT_TO_DETECTION = True            # etapa 1: solo el hueso dentro de las regiones detectadas (cajas axiales)

def segmentation_loss(
    logits: torch.Tensor,
    fragment_mask: torch.Tensor,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
):
    """`ce_weight` · CE + `dice_weight` · Dice multiclase (fondo, sacro, coxal izq., coxal der.)."""
    target = bone_targets(fragment_mask).to(logits.device)
    ce = F.cross_entropy(logits.float(), target)
    dice = multiclass_dice_loss(logits, target)
    return ce_weight * ce + dice_weight * dice, {"seg_ce": float(ce.detach()), "seg_dice_loss": float(dice.detach())}
```

**Decisión del equipo y comparación:** Preferimos dejar `0.5` para Cross Entropy y `0.5` para Dice. La primera ayuda a corregir errores por vóxel; Dice mira directamente cuánto se superponen la región predicha y la real. Como hay mucho fondo y algunas regiones óseas ocupan menos espacio, no queríamos depender solamente de acertar vóxeles individuales. **Acierto:** la función combina dos señales complementarias. **Limitación:** el código por sí solo no demuestra que 50/50 sea superior a 80/20: una comparación justa exigiría entrenar ambos bajo las mismas condiciones.

## 12. Incorporación de una cabeza de máscaras por ROI

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-09

#### Prompt enviado a ChatGPT

> En nuestro proyecto PENGWIN ya tenemos una cabeza de detección que predice bounding boxes y una cabeza de segmentación semántica que identifica sacro, coxal izquierdo y coxal derecho. Sin embargo, como equipo queremos añadir otra cabeza que genere una máscara específica para cada región detectada, porque necesitamos identificar mejor los fragmentos y sus bordes. ¿Cómo podemos implementar esta cabeza adicional en PyTorch utilizando ROI Align, sin reemplazar las cabezas que ya tenemos?

#### Respuesta de ChatGPT

Implementar una cabeza adicional inspirada en el funcionamiento de Mask R-CNN, pero construida con módulos propios de PyTorch. La propuesta consistía en extraer las características de cada región detectada, procesarlas mediante convoluciones y generar una máscara por clase anatómica.

Como configuración inicial, propongo utilizar un ROI de 32×32, una máscara de 64×64 y 32 canales internos, para evitar que la nueva rama consumiera demasiada memoria.

**Fragmento de código propuesto:**

```python
M3D_MASK_HEAD = True
M3D_ROI_FEAT = 32
M3D_MASK_SIZE = 64
M3D_MASK_CHANNELS = 32
M3D_MASK_DETACH = False
M3D_MASK_JITTER = 0.05
```

También planteó que se podía utilizar una combinación de Binary Cross Entropy y Dice Loss para entrenar las máscaras individuales.

#### Modificaciones realizadas por el equipo

Aunque la IA nos ayudó a entender cómo implementar la cabeza, la decisión de agregarla surgió del equipo. Consideramos que la segmentación semántica global no era suficiente para el nivel de detalle que queríamos obtener en los fragmentos.

Además, revisamos los parámetros propuestos y decidimos trabajar con ROIs y máscaras de mayor resolución, aprovechando también el CT a doble resolución en el plano.

**Código final implementado** (`src/config.py`, líneas 209–222):

```python
M3D_MASK_HEAD = True
M3D_HIRES_FACTOR = 2
M3D_HIRES_DIR = DATA_ROOT / f"cache_cubo_{int(CUBE_MM)}mm_{INPUT_SIZE}_axial{INPUT_SIZE * M3D_HIRES_FACTOR}"
M3D_ROI_FEAT = 48
M3D_MASK_SIZE = 96
M3D_MASK_CHANNELS = 64
M3D_MASK_DETACH = True
M3D_MASK_JITTER = 0.10
M3D_MASK_SCORE_THR = 0.5
M3D_MASK_THR = 0.5
M3D_MASK_CHUNK = 32
M3D_FUSION_MODE = "union_sacro"
```

La cabeza quedó implementada mediante la clase `RoiMaskHead` en `src/models/model3d.py`:

```python
class RoiMaskHead(nn.Module):
    def __init__(self, in_feat: int, in_img: int,
                 mid: int = config.M3D_MASK_CHANNELS,
                 classes: int = 3):
        super().__init__()
        relu = lambda: nn.ReLU(inplace=True)

        self.feat = nn.Sequential(
            nn.Conv2d(in_feat, mid, 3, padding=1), relu(),
            nn.Conv2d(mid, mid, 3, padding=1), relu()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(mid, mid, 2, stride=2), relu()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid + in_img, mid // 2, 3, padding=1), relu(),
            nn.Conv2d(mid // 2, mid // 2, 3, padding=1), relu()
        )

        self.mask = nn.Conv2d(mid // 2, classes, 1)

    def forward(self, f: torch.Tensor, img: torch.Tensor):
        h = self.feat(f)
        m = self.fuse(torch.cat([self.up(h), img], 1))
        return self.mask(m), h.mean((2, 3))
```

La pérdida utilizada para la cabeza fue BCE + Dice:

```python
def mask_loss(logits, cls, target):
    if logits.shape[0] == 0:
        return logits.sum() * 0.0

    lg = logits[
        torch.arange(len(cls), device=logits.device), cls
    ].float()

    bce = F.binary_cross_entropy_with_logits(lg, target)
    p = torch.sigmoid(lg)

    dice = 1 - (
        2 * (p * target).sum((1, 2)) + 1
    ) / (
        p.sum((1, 2)) + target.sum((1, 2)) + 1
    )

    return bce + dice.mean()
```

#### Comparación y análisis crítico

**Aciertos:** La IA nos dio una estructura útil para implementar la idea que ya teníamos. Nos permitió entender cómo extraer características de cada ROI, recuperar resolución con convoluciones transpuestas y generar máscaras condicionadas por la clase anatómica.

**Cambios del equipo:** No dejamos los parámetros iniciales. Aumentamos `M3D_ROI_FEAT` de 32 a 48, `M3D_MASK_SIZE` de 64 a 96 y los canales de 32 a 64. Consideramos que una máscara de mayor resolución era más apropiada para conservar información de los bordes, especialmente porque trabajamos con fragmentos óseos que pueden ser pequeños o estar muy cercanos entre sí.

También cambiamos `M3D_MASK_DETACH` a `True`, para evitar que el entrenamiento de esta nueva cabeza modificara directamente las características del backbone. Esto nos permitía incorporar el refinamiento sin alterar mediante esa pérdida el aprendizaje de la detección.

**Limitaciones:** La propuesta de la IA no consideraba completamente la resolución que necesitábamos para nuestros fragmentos. Por otro lado, aumentar la resolución y los canales también incrementa el consumo de memoria y tiempo de procesamiento. Además, esta cabeza depende de la calidad de las bounding boxes: si la región detectada es incorrecta, la máscara también puede verse afectada.

**Resultado alcanzado:** Se incorporó una cabeza adicional que genera una máscara por clase para cada ROI y devuelve características que pueden aprovecharse en clasificación. Esto amplió la arquitectura inicial, aunque para afirmar que mejoró el Dice es necesario comparar los resultados con la cabeza activada y desactivada bajo las mismas condiciones.

## 13. Incorporación de una segunda ventana HU para mejorar el contraste óseo

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-09

#### Prompt enviado a ChatGPT

> Estamos trabajando con tomografías de pelvis y actualmente hacemos el preprocesamiento utilizando una ventana HU general para resaltar el hueso. Sin embargo, como equipo queremos añadir una segunda ventana enfocada en el tejido trabecular, porque pensamos que una sola ventana puede perder detalles internos de los huesos. ¿Cómo podemos implementar las dos ventanas como canales de entrada en nuestro modelo PyTorch? ¿Qué debemos tener en cuenta para no perder la información original?

#### Respuesta de ChatGPT

Pueden generar dos representaciones diferentes del mismo volumen CT, aplicando ventanas HU independientes y normalizando cada una entre 0 y 1.

Una posibilidad era conservar una ventana amplia para las estructuras óseas generales y utilizar una segunda ventana más estrecha para resaltar diferencias de intensidad dentro del hueso.

Como ejemplo inicial, planteemos una segunda ventana de 200–1000 HU.

**Fragmento de código propuesto como punto de partida:**

```python
HU_WINDOW = (50.0, 1300.0)
TRAB_WINDOW = (200.0, 1000.0)

bone = window_hu(hu, HU_WINDOW)
trab = window_hu(hu, TRAB_WINDOW)

image = np.stack([bone, trab], axis=0)
```

La IA también advirtió que el modelo debía aceptar dos canales de entrada en lugar de uno y que las ventanas debían aplicarse sobre los HU originales, no sobre una imagen que ya estuviera normalizada.

#### Modificaciones realizadas por el equipo

La propuesta de incluir una segunda ventana fue una decisión nuestra, porque queríamos darle al modelo información adicional sobre el contraste interno del hueso.

No adoptamos el rango inicial del ejemplo. Elegimos configurar una ventana trabecular de **100–700 HU**, que permite representar con mayor contraste variaciones de intensidad dentro de ese intervalo.

**Código final implementado** (`src/config.py`, líneas 193–205):

```python
M3D_RUN_CBAM = "unet3d_cbam_trab"
M3D_RUN_SE = "unet3d_se_trab"
M3D_RUN_NO_CBAM = "unet3d_sin_atencion"
M3D_RUN_PREV = "unet3d_cbam"

M3D_ATTENTION = "cbam"
M3D_TRAB_WINDOW = (100.0, 700.0)
M3D_DECODER_ATTENTION = ("p3",)
```

La función utilizada para realizar el ventaneo se encuentra en `src/dataset/preprocessing.py`, líneas 105–110:

```python
def window_hu(
    hu: np.ndarray,
    window=config.HU_WINDOW
) -> np.ndarray:
    low, high = window

    if not high > low:
        raise ValueError(f"Ventana HU inválida: {window}")

    return (
        (np.clip(hu.astype(np.float32), low, high) - low)
        / (high - low)
    ).astype(np.float32)
```

#### Comparación y análisis crítico

**Aciertos:** La IA nos ayudó a entender cómo representar un mismo volumen utilizando dos ventanas HU, sin perder la correspondencia espacial entre los canales. También fue útil la aclaración de que el ventaneo debía hacerse antes de normalizar los datos.

**Cambios del equipo:** Decidimos utilizar 100–700 HU para el segundo canal, en lugar de la propuesta ilustrativa de 200–1000 HU. La razón es que un intervalo más estrecho aumenta la separación normalizada entre intensidades que se encuentran dentro de la ventana.

Por ejemplo, dos vóxeles con valores de 200 y 400 HU tienen una diferencia normalizada mayor dentro de una ventana de 100–700 HU que dentro de una ventana de 200–1000 HU. Consideramos que este contraste adicional podía ser útil para distinguir estructuras óseas internas.

También mantuvimos la ventana general de 50–1300 HU, porque no queríamos que la segunda ventana reemplazara la información de las estructuras de mayor densidad. Nuestra intención fue complementar ambos canales.

**Limitaciones:** La IA inicialmente nos dio un rango general que no necesariamente era el más adecuado para nuestro objetivo. Sin embargo, tampoco podemos asumir que 100–700 HU es mejor únicamente por ser más estrecho. Al recortar intensidades fuera del intervalo, se puede perder información útil, por lo que la mejora debe evaluarse experimentalmente.

**Resultado alcanzado:** Incorporamos la configuración de una segunda ventana trabecular mediante `M3D_TRAB_WINDOW = (100.0, 700.0)` y definimos variantes del modelo que permiten trabajar con ese contraste adicional. Esto constituye una modificación propia del preprocesamiento. Para cuantificar su beneficio, debemos comparar el modelo de uno y dos canales manteniendo constantes los demás parámetros.

## 14. Cambio del modelo 3D inicial

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-09

**Prompt a Claude:** «Usar las cajas predichas como zona de búsqueda de una cabeza de máscara con ROI Align, como Mask R-CNN; mejorar el contraste del preprocesamiento; y poner atención en las primeras capas del decoder. Haz una propuesta y figuras para decidir. Lo que decidimos como equipo: (1) cabeza de máscara 2D por corte, sobre cada caja axial, entrenada junto con el resto pero sin que su pérdida llegue al backbone, para no afectar al detector; (2) un segundo canal de entrada con la ventana trabecular de 100 a 700 HU, que fue la que ganó en la figura; (3) CBAM con γ en la primera capa del decoder y en la detección, y quitarlo del bloque 3 del encoder, donde γ había quedado en cero; (4) un segundo modelo igual pero con squeeze-excitation en los mismos lugares, para la ablación; (5) el optimizador sigue siendo AdamW; (6) la clasificación pasa a ser la rama de clase de cada caja, que elige la máscara.»

**Respuesta:** Una propuesta con opciones para cada punto, una figura que comparaba opciones de contraste en tres CT (ecualización global, CLAHE y ventanas como canales) y la implementación de lo que el equipo eligió.

**Código final:**

```python
# src/config.py
M3D_TRAB_WINDOW = (100.0, 700.0)
M3D_DECODER_ATTENTION = ("p3",)
M3D_CBAM_BLOCKS = (4,)
M3D_MASK_DETACH = True
M3D_RUN_SE = "unet3d_se_trab"               # modelo de comparación con squeeze-excitation
```

**Decisión del equipo y comparación:** Las decisiones del equipo en cada punto:
- Cabeza de máscara 2D por corte o 3D por región: elegimos 2D.
- Entrenarla aparte, junto con gradiente o junto sin gradiente: elegimos junto y sin gradiente al backbone.
- Contraste (ecualización global, CLAHE o ventanas como canales): elegimos la ventana trabecular, que fue la mejor en la figura.
- Qué clasifica la cabeza de clasificación (cada hueso, cada fragmento o cada caja): elegimos cada caja.

**Error de la IA:** propuso la ecualización global del histograma, y la figura mostró que realzaba el tejido blando y empeoraba la separación del hueso. **Limitación:** no entrenamos un modelo por cada cambio, así que no sabemos cuánto aporta cada uno por separado.

## 15. Corrección del decoder para evitar el colapso de clases durante la segmentación

**Herramienta:** ChatGPT · **Fecha de la decisión:** 2026-10-09

#### Prompt enviado a ChatGPT

> Estamos entrenando nuestro modelo 3D para segmentar sacro, coxal izquierdo y coxal derecho, pero encontramos un problema: algunas clases dejan de aprender correctamente y la salida del decoder parece quedarse sin activaciones útiles. Actualmente utilizamos ReLU. ¿Qué podemos revisar para evitar que una clase colapse durante el entrenamiento? ¿Sería conveniente utilizar LeakyReLU o agregar una salida auxiliar en el decoder?

#### Respuesta de ChatGPT

Uno de los posibles problemas de utilizar ReLU es que las activaciones negativas se convierten en cero y, dependiendo del entrenamiento, algunas neuronas pueden permanecer inactivas.

Como alternativa, se sugiere probar LeakyReLU, que conserva una pendiente pequeña para los valores negativos y permite que el gradiente continúe propagándose.

Se debe tener en cuenta el concepto de supervisión profunda (*deep supervision*), que consiste en agregar una predicción auxiliar en una etapa intermedia del decoder y calcular una pérdida adicional durante el entrenamiento.

Como punto de partida, se podía utilizar una pendiente de 0.1 para LeakyReLU y dar el mismo peso a la salida principal y a la auxiliar.

**Fragmento de código propuesto:**

```python
DECODER_ACT = "leaky"
LEAKY_SLOPE = 0.1

SEG_AUX = True
SEG_AUX_WEIGHT = 1.0

loss_seg = loss_main + SEG_AUX_WEIGHT * loss_aux
```

La IA recomendó revisar las métricas por clase, porque una reducción en la pérdida global no garantiza que el modelo esté aprendiendo correctamente todas las categorías anatómicas.

#### Modificaciones realizadas por el equipo

En nuestro proyecto identificamos un problema de colapso relacionado con las activaciones del decoder, especialmente en las salidas correspondientes al sacro y al coxal derecho.

Revisamos las alternativas y decidimos incorporar LeakyReLU, pero con una pendiente de 0.01, en lugar de utilizar el valor de 0.1 del ejemplo.

Además, agregamos supervisión profunda sobre P1 con un peso auxiliar de 0.5, para que la predicción intermedia ayudara al aprendizaje sin tener el mismo peso que la salida principal.

**Código final implementado** (`src/config.py`, líneas 200–207):

```python
# Configuración del decoder y supervisión auxiliar.
M3D_DECODER_ACT = "leaky"
M3D_LEAKY_SLOPE = 0.01

M3D_SEG_AUX = True
M3D_SEG_AUX_WEIGHT = 0.5

M3D_MIN_BONE_DICE = 0.5
```

La configuración documenta que la pérdida de segmentación combina la salida principal P0 y la salida auxiliar P1:

```python
# Esquema de la pérdida documentado en config.py.
L_seg = L_P0 + 0.5 * L_P1
```

También se configuró un control de Dice mínimo por hueso, para facilitar la detección de clases cuyo desempeño fuera demasiado bajo.

#### Comparación y análisis crítico

**Aciertos:** La IA nos permitió relacionar el problema de entrenamiento con las funciones de activación y la propagación del gradiente. Además, nos ayudó a entender por qué una salida auxiliar podía proporcionar una señal de entrenamiento adicional al decoder.

**Cambios del equipo:** No utilizamos directamente los hiperparámetros iniciales del ejemplo. Elegimos `M3D_LEAKY_SLOPE = 0.01` porque queríamos permitir el paso de gradientes negativos sin darles demasiado peso.

También redujimos la contribución de la pérdida auxiliar de 1.0 a 0.5. Consideramos que la predicción principal debía seguir siendo el objetivo más importante y que la salida intermedia debía funcionar como un apoyo durante el entrenamiento.

Otro cambio fue establecer `M3D_MIN_BONE_DICE = 0.5`, ya que necesitábamos revisar el comportamiento por categoría anatómica y no solamente observar la pérdida total.

**Limitaciones:** Aunque LeakyReLU ayuda a evitar que el gradiente sea cero para entradas negativas, no garantiza por sí sola que desaparezca el colapso de clases. Este problema también puede estar relacionado con el desbalance de etiquetas, las pérdidas, el optimizador y la distribución de los datos.

Por otro lado, agregar supervisión profunda aumenta los cálculos durante el entrenamiento y requiere controlar el peso de la salida auxiliar. Si este es demasiado alto, puede afectar la optimización de la salida principal.

**Resultado alcanzado:** El proyecto incorporó LeakyReLU en el decoder, supervisión auxiliar en P1 y un control de Dice mínimo por categoría. Estas modificaciones permiten abordar el problema identificado durante el entrenamiento, aunque su eficacia debe comprobarse comparando las métricas por clase antes y después de los cambios.

## 16. Resultados finales, fusión, inferencia y verificación

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-10 a 10-11

**Prompt a Claude:** «Ya están entrenados los dos modelos. Evalúalos y actualiza los notebooks 02, 03 y 04 con el modelo nuevo. Compara la segmentación semántica con la cabeza de máscara en la misma rejilla y dime si conviene unirlas.»

**Respuesta:** El script `scripts/comparar_cabezas.py`, la función de fusión `fuse_heads` con tres modos y los notebooks actualizados. Propuso también otros postprocesos de limpieza.

**Código final:**

```python
# src/config.py
M3D_FUSION_MODE = "union_sacro"             # elegido en validación
```

**Decisión del equipo y comparación:** De las tres formas de fusión elegimos la que dio mejor Dice en validación (`union_sacro`); test solo se miró después. Los demás postprocesos quedaron guardados sin implementar. **Acierto:** la fusión tiene el mejor borde en test (HD95 1,9 mm). **Limitación:** la ganancia frente a la cabeza de máscara es pequeña.

## Análisis crítico

### Errores de la IA

- **Claude** propuso una ventana HU que no servía (200 a 1500) y se cambió por la de una fuente de radiología (50 a 1300).
- **Claude** propuso parada temprana y calentamiento de la tasa de aprendizaje sin que el entrenamiento los necesitara; se descartaron.
- **Claude:** el primer notebook de evaluación comparaba métricas distintas y solo en validación, así que no dejaba ver el sobreajuste; hubo que rehacerlo.
- **Claude** propuso la ecualización global del histograma para el contraste; la figura de comparación mostró que realzaba el tejido blando y empeoraba la separación del hueso.
- **Claude:** la primera versión del modelo modificado colapsó al entrenar: la segmentación solo aprendía un hueso, porque la última capa del decoder tuvo neuronas ReLU muertas. Se corrigió con LeakyReLU, supervisión profunda en P1 y el Dice de cada hueso por separado en la validación.
- **ChatGPT** propuso valores genéricos (anclas iguales para las tres vistas, umbrales de NMS y de asignación, tamaño de la cabeza de máscara, rango de la ventana trabecular, pendiente de LeakyReLU y peso de la salida auxiliar) que no estaban ajustados a los datos de PENGWIN; el equipo los cambió en todos los casos.

### Aciertos

- Las técnicas de memoria (gradient checkpointing y precisión mixta) permitieron entrenar el modelo 3D completo en 6 GB.
- La regla para los λ deja cada tarea con el peso que el equipo quiere al empezar, sin depender de la escala de cada pérdida.
- La ventana trabecular se eligió con una figura que comparaba las opciones en tres CT, y la fusión se eligió en validación.
- ChatGPT ayudó a entender cómo construir la cabeza de máscara con ROI Align, la supervisión profunda y el uso de dos ventanas como canales.

### Limitaciones

- Ninguna de las dos IA conocía todo el contexto del proyecto, así que cometieron errores que el equipo tuvo que corregir para que los métodos y los resultados fueran los adecuados.
- Varias decisiones no tienen una comparación controlada (anclas, umbrales, pesos de la pérdida, tamaño de la cabeza de clasificación, canal trabecular): se justifican por los datos o por criterio, no por un experimento que aísle su efecto.

### Balance

ChatGPT nos sirvió para plantear opciones técnicas y entender qué hacía cada parte de las cabezas, pero no era suficiente aceptar parámetros genéricos. En nuestro caso importaba más que los valores fueran compatibles con las cajas de PENGWIN, las características del CT, los recursos de cómputo y la evaluación. El principal aprendizaje fue que una propuesta de IA puede servir como punto de partida sin convertirse en una decisión definitiva. Además, la cabeza adicional de máscara y la ventana trabecular son decisiones del trabajo del equipo que no corresponde atribuir a una sugerencia de IA.
