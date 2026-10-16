# IA_USAGE · Uso de IA generativa en el proyecto PENGWIN

**Proyecto:** detección, segmentación y clasificación de fracturas pélvicas en CT.

**Modelos de IA generativa usados:**

- **Claude Opus 5.5** (Anthropic), con Claude Code: asistente de programación y de análisis. Escribía código y scripts, corría pruebas y evaluaciones y, antes de cada cambio de diseño, presentaba opciones; la decisión la tomaba el equipo.
- **ChatGPT** (OpenAI): consultas puntuales sobre valores de partida de hiperparámetros y sobre cómo implementar componentes.

**Cómo leer este archivo:**

- Cada entrada tiene el prompt, lo que respondió o propuso la IA, el código final y la decisión del equipo con su análisis crítico.
- Las entradas están en **orden cronológico**, según la fecha en que la decisión quedó en el código (registro de decisiones del proyecto). Por eso Claude y ChatGPT aparecen intercalados.
- Los prompts de Claude son los **mensajes textuales** enviados en las sesiones de Claude Code, con su fecha y hora; en ellos van las decisiones que se tomaron. Se copian tal como se escribieron. El contenido pegado (fragmentos de la rúbrica, código, apuntes) se indica entre corchetes. Los de ChatGPT son los que registró el equipo.

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
| 17 | 2026-10-11 | Claude | Dashboard |

## 1. División de los datos y EDA

**Herramienta:** Claude · **Fecha de la decisión:** 2026-09-24 a 09-25

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-09-24 11:19:*

> Entonces pon el MIP de primero, porque ese es el obligatorio. 
>
> De qué forma se puede dividir este trabajo para que 4 personas puedan hacer un commit de esta parte? 
>
> Estaba pensandolo así
>
> * Splits fijos y Ventaneo HU
> * Carga y Preprocesamiento
> * EDA
> * Visualizador
>
>
> Esto ultimo es solo una pregunta.

*2026-09-24 11:30:*

> Pues si se debe ahcer el split de nuevo se hace. Pero que quede completamente aislado del notebook, que no dependa de este mejor. Por reproducibilidad entendería que ese script solo se usa una vez, no?

*2026-09-24 11:35:*

> Sí, que lo unico que quede del splits sea leer el archiv para ver cómo se separaron las clases 
> Es decir como esto, pero leyendo el archivo
>
> [código pegado: la celda del notebook que asignaba los casos a train / val / test]
>
> Solo esta sí dejarla, pero que lea el archivo

*2026-09-25 21:03:*

> ahi esta el notebook actualizado con 200 a 1300. Di que el umbra se definió en base a una pagina de radiologia
> https://radiopaedia.org/articles/hounsfield-unit
>
> No des marcas de tiempo, solo que debo mostrar en cada tramo y ya.

**Qué propuso la IA:** El script `src/dataset/splits.py` y la propuesta de agrupar los cinco tipos de fractura en tres estratos (sacro fracturado, coxal unilateral o sin fractura, coxal bilateral). También propuso una ventana HU de 200 a 1500.

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

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-03 10:00:*

> @Taller_3_Analítica_de_Datos.ipynb @Investigación trabajos similares.pdf
> Basate bastante en esta arquitectura y en la Unet para crear la red 3D. 
>
> dame todas las opciones que tenemos para Meter el volumen 3D, si en parches, completo, etc. Dame fuentes de estudios similares. Puedes usar tu propia investigación, pero ademas de eso

*2026-10-03 10:22:*

> Vale, creo que voy a intentar esa red backbone 3D con deteccion 2D en la RTX3060. Como podría hacerlo en ese computador usando esta misma sesion o abro otra o como?
> Adicional, entonces solo corre el modelo CBAM 2..5D, no me importa el sin CBAM. Cuanto estimarias que se demoraria en modelo 3D en la RTX3060?

*2026-10-03 10:59:*

> mi RTX es de 6 Gb, se tuvieron hacer unos cambios. Será mejor esperar a que termine este para entrenar el 3D en mac con 16 GB aunque sea más lento?

*2026-10-07 16:26:*

> El cuello, pon el intermedio. La vecindad dejala en 6. Si, la perdida en deteccion agrega el cambio. 
>
> Me van a pasar de nuevo la arquitectura mis compañeras, entonces estos cambios los tienes que tener siempre presente para agregarlo cuando vuelva a cambiarlo.. 
>
> el canal del borde no entendi lo de solo o con 4 clases semantica.

**Qué propuso la IA:** La implementación en `src/models/model3d.py`, gradient checkpointing y precisión mixta para que el modelo cupiera en 6 GB, tres tamaños posibles para el cuello de botella, y parada temprana y calentamiento de la tasa de aprendizaje.

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

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-06 21:05:*

> Por ahora solucionemos lo del lambda, creo que lo ideal sería darle más peso a segmentación, un poco más de a clasificación que a detección. No sé qué te parece o que recomiendas. Sobre la rúbrica, la rúbrica también dice esto, no se contradice con lo otro que hallaste?
> "ipeline completo: detección → NMS →
> segmentación de instancia → clasificación
> por pertenencia. "
> y de esto
> "
>
> [texto pegado de la rúbrica]
>
> "??
>
> los fragmentos que se tocan no tengo idea qué hacer con eso. 
>
> Si 16 canales están bien entonces adecúalo a 16, sino recomiendame la cantidad. En la inferencia no sé si lo de las vistas sea un problema realmente. Lo de comparar da igual.

*2026-10-07 08:51:*

> Ya que toca reentrenar, ajusta los pesos, para que detección no esté con un peso tan bajo. que quede más igualado a clasificación

**Qué propuso la IA:** Calcular cada λ como la prioridad de la tarea dividida entre su pérdida inicial, medida con el modelo sin entrenar, y el comando `python -m src.models.model3d calibrate` para medirla.

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

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-07 07:04:*

> Ya tengo el modelo entrenado. Organiza lo que trae el zip en el repo. 
> El Dice hay que calcularlo para cada fragmento, y el mAP .50:95 se quedó cortó a lo pedido. No sé qué hacer. Hazme el notebook de la evaluación del modelo en todas las cabezas, que sea completo, que sea como de paper, aunque de nuestro alcance, nada demasiado complejo.

*2026-10-07 07:36:*

> Eso es demasiado. Todas esas gráficas. Demasiado para un curso de analítica de pregrado. Volviste a falla en lo que se te pidió. Que la curva de loss sea frente a validación, no frente a otras pérdidas.
> Lo mismo como las métricas: entrenamiento vs validación.

**Qué propuso la IA:** El comando `python -m src.models.model3d evaluate` y los notebooks 02 y 03. La primera versión del notebook comparaba métricas distintas y solo en validación.

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

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-09 10:56:*

> Vamos a hacer algunos cambios. Primero hay que entrenar el modelo de nuevo. 
> El profesor dijo al mostrarle los avances y explicarle que por ahora estabamos trabajando con la misma arquitectura con dos cabezas de segmentación: "Hay que refinar el modelo, pasarlo de segmentación semantica, y luego, si se puede... ustedes pueden hacer dos cosas, hacerlo lo quehicimos con SAM usar los bounding box predichos y ese mismo bounding box predicho pasarlo como una zona de búsqueda para que una segunda RPN, pero que esta trabaje para segmentación, agrupamiento, y utilizar lo que sería la parte del ROI Align, con eso tendrían mejores resultados. O sea utilizar todo el proceso de una masket (no entendí aquí), pero hacerlo dentro de su arquitectura de detección y con eso tendría un doble beneficio con segmentación orientada a lo que ya se está haciendo del bounding box. Un posible tip, ustedes verán si lo toman o no. Para tener una mejor extracción de esos objetos que están ¿volantes? (tampoco entendí aqui). De pronto también mejorar un poco el contexto de la resolución espacial, que están usando en el pipeline de preprocesamiento de la imagen?": respondí con lo que teníamos "Utilizaron ecualizador del histograma de la imagen? Hay una librería que se llama MONAI trae ya unas preconfiguraciones para hacer el transform previo al data loader ,para mejorar el contraste, también para data augmentation. Porque estoy viendo que no hay una buena ecualización del histograma, entonces la resolución espacial se pierde un poquito entonces se empieza a ocultar el hueso frente a otro tejido. Entonces hay que mejorarlo un poco más. De hecho hay un método por segmento, que puede darme realce a esas zonas de interés, para que salga más representación para que el modelo lo tenga fácilmente diferenciables frente al resto de objetos. y ya con eso podemos sustraer del modelo". además la atención, en las primeras capas del upsampling ayuda a agrupar pixeles realmente de revelancia a la máscara de interés"
>
> hazme una propuesta de los cambios para presentarla. 
>
> estos fueron otros apuntes un poco más sobre la marcha de otras presentaciones. 
>
> [apuntes pegados de otras presentaciones]

*2026-10-09 12:05:*

> Cual es la decision segun esto: 
> Para la parte 2. Creo que lo discutimos ayer en la reu, según eso era lo que hacíamos
> Para la 3 la opción B me parece más razonable como punto de partida para la prueba, si no llegase a mejorar mucho, la C
> Para la 4  pensaría que dos bloques CBAM ahí bien, pero también probar con atención solo de canal, y si dan resultados similares pues quedarnos solo de canal ya que haría el modelo más liviano
>
> Opinion x2: pensamos que hay que entrenar con CBAM (en las primeras capas del decoder como dijo él y no sé si tambien implementemos CBAM en la detección) y sin CBAM. También el preprocesamiento de contraste pensaría que podríamos probar con ventana trabecular o tal vez CLAHE 3D. Hagamos la cabeza de máscara porque esa fue la principal recomendación del profesor y no creo que sea tan necesario implementar Lion
>
> Que esta decidido entre ambos, y que faltaría por desempatar

*2026-10-09 12:47:*

> Bien, entonces, solo elimina el del bloque 3 del encoder, y que se quede el del bloque 4. Pasame el zip para los dos modelos. Sin CBAM y Squeeze Excitation y con CBAM donde discutimos. Primero el de CBAM, luego sin CBAM

*2026-10-09 13:14:*

> Espera qué. Por qué cabeza de mscara después. Implementala de una vez. 
> Solo en p3 está bien. Una cosa, esa cabeza de clasificacion qué está haciendo? porque creo que no está haciendo nada. De que forma puede ser util y evitar que el F1 score sea 1 durante todo el entrenamiento? Se baja el ponderado del loss o qué?
> Dime como así 2D por corte o 3D por region y cual recomiendas, tomamos la decision

*2026-10-09 13:23:*

> Sí, opción 1, implementa todo y arma los ZIP

**Qué propuso la IA:** Una propuesta con opciones para cada punto, una figura que comparaba opciones de contraste en tres CT (ecualización global, CLAHE y ventanas como canales) y la implementación de lo que el equipo eligió.

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

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-10 07:26:*

> En la carpeta raiz están los zips de los entrenamientos de los dos modelos. Guardalos y archiva o elimina los demas. Reejecuta los notebooks, en 03, que mire tambien la nueva cabeza de mascara, cual es su salida. Esta ayuda de alguna forma a la segmentación?
> Que aqui se miren todas las salidas de las cabezas, y los 4 casos donde les va bieen, mal, etc. Y haz la comparación entre CBAM y SE

*2026-10-11 10:44:*

> Se puede hacer un post procesamiento para que la máscara y y la cabeza segmentadora se unan para sacar como lo mejor de ambos (sacro de segmentacion semantica y los fragmentos pequeños de mascara?

*2026-10-11 11:00:*

> Sí, agrega la fusión a evaluate, pero entonces en el notebook haz una nueva sección donde se compare esto, semantica, cabeza de mascara y la fusión. Si hay otro post procesamiento que se pueda realizar para esto me dices. Mientras corre evaluación, tengo el zip del dashboard, entonces para que lo.  implementes. y lo corras con el modelo CBAM. Si haces cambios para que funcione me dices.

**Qué propuso la IA:** El script `scripts/comparar_cabezas.py`, la función de fusión `fuse_heads` con tres modos y los notebooks actualizados. Propuso también otros postprocesos de limpieza.

**Código final:**

```python
# src/config.py
M3D_FUSION_MODE = "union_sacro"             # elegido en validación
```

**Decisión del equipo y comparación:** De las tres formas de fusión elegimos la que dio mejor Dice en validación (`union_sacro`); test solo se miró después. Los demás postprocesos quedaron guardados sin implementar. **Acierto:** la fusión tiene el mejor borde en test (HD95 1,9 mm). **Limitación:** la ganancia frente a la cabeza de máscara es pequeña.

## 17. Dashboard

**Herramienta:** Claude · **Fecha de la decisión:** 2026-10-11

**Prompts enviados a Claude** (textuales, con la hora de Colombia):

*2026-10-11 11:27:*

> Guarda estas sugerencias para después. 
>
> En el dashboard, asegurate que se forme la linea entre el fragmento y el principal, y el color de las distancias hazlo de un color más visible, porque no se ve, amarillo, o no sé. Esto en el 3D, y asegurate de que los puntos de medicion empiecen en el borde y no en el centro del hueso porque ahora mismo aparecen en el centro. 
>
> y aparece P en lugares donde no hay distancias, no sé si eso es ncesario en el corte a corte

*2026-10-11 11:39:*

> Se volvió el cubo a las medidas originales? el tamaño original de voxeles en lugar de que sea isotropico? porque hay que hacer eso. Los puntos siguen sin estar bien puestos. no estan en contacto con la superficie

*2026-10-11 11:52:*

> Ahora está perfecto. Hay forma de que se mejore la velocidad de esto? el profesor decia que hacer que las coass se hicieran de lado del cliente, pero no sé a qué se refería y no sé si eso ayuda. Tal vez guardar el caché, para que una vez que se hace la prediccion se guarde y no se deba hacer dos veces para cada vista.

**Qué propuso la IA:** `src/inference.py`, la línea y la etiqueta de la distancia en los visualizadores 2 y 3, guardar las predicciones en disco y enviar las imágenes como PNG.

**Código final:**

```python
# dashboard/estado.py
CACHE_DISCO = config.RESULTS_DIR / "dashboard_cache"       # predicciones guardadas (se puede borrar sin riesgo)
```

**Decisión del equipo y comparación:** Aceptamos guardar las predicciones en disco y enviar las imágenes como PNG. **Acierto:** la segunda vez que se abre un caso carga de inmediato y la distancia es visible en los visualizadores 2 y 3, como pide la rúbrica.

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
