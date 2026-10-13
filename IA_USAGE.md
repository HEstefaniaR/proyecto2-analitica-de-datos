# IA_USAGE · Uso de IA generativa en el proyecto

- **Modelo de IA:** Claude Opus 5.5, usado con Claude Code (Anthropic).
- **Cómo se usó:** como asistente de programación y de análisis. La IA escribía código y scripts, corría pruebas y evaluaciones. Antes de cada cambio de diseño presentaba opciones; la decisión la tomaba el equipo.
- **Prompts:** están en [`IA_USAGE_prompts.md`](IA_USAGE_prompts.md), resumidos por tema y con las decisiones del equipo en cada uno.

## Resumen por tema

| # | Tema | Qué se pidió | Qué aportó la IA | Qué decidió o cambió el equipo |
|---|---|---|---|---|
| 1 | División de los datos y EDA | División train / validación / test en un script aparte, 70 / 15 / 15, estratificada y con semilla fija. | `src/dataset/splits.py` y la propuesta de tres estratos por tipo de fractura. | Se aceptaron los estratos. Se rechazó la ventana HU que propuso la IA (200 a 1500) y se usó la de una fuente de radiología (50 a 1300). |
| 2 | Modelo 3D multitarea | Backbone U-Net 3D con detección 2D por corte, para entrenar en una GPU de 6 GB, 140 épocas. | Implementación en `src/models/model3d.py`; gradient checkpointing y precisión mixta para que cupiera en memoria. | Se usaron las dos técnicas de memoria. De tres tamaños del cuello de botella se eligió el intermedio (128 canales). Se descartaron la parada temprana y el calentamiento de la tasa de aprendizaje. |
| 3 | Pérdida multitarea | Calibrar los λ, con más peso a la segmentación. | La regla λ = prioridad / pérdida inicial y el comando `calibrate`. | Se aceptó la regla. Las prioridades las fijó el equipo: segmentación 2, detección 1,5 y clasificación 1,5. |
| 4 | Evaluación y notebooks | Evaluación final en validación y test con las métricas de la rúbrica; HD95 y ASSD por cabeza. | `model3d evaluate` y los notebooks 02 y 03. | Se rechazó la primera versión del notebook, que comparaba métricas distintas solo en validación; se aceptó la segunda, con curvas de entrenamiento frente a validación. |
| 5 | Cambio del modelo 3D inicial | Cabeza de máscara con ROI Align (como Mask R-CNN), mejor contraste y atención en el decoder. | Propuesta con opciones, figura de comparación del contraste e implementación. | Cabeza de máscara 2D por corte, entrenada junto y sin gradiente al backbone; ventana trabecular (100 a 700 HU) como segundo canal; CBAM en la primera capa del decoder y en la detección; modelo con squeeze-excitation para la ablación; AdamW; clasificación por caja. |
| 6 | Resultados finales, fusión, inferencia y verificación | Evaluar los dos modelos, actualizar los notebooks y comparar la segmentación semántica con la cabeza de máscara. | `scripts/comparar_cabezas.py`, la función de fusión y los notebooks actualizados. | De tres formas de fusión se eligió la mejor en validación (`union_sacro`). Otros postprocesos propuestos quedaron sin implementar. |
| 7 | Dashboard | Integrar el dashboard con el modelo final, mostrar la medición y hacerlo más rápido. | `src/inference.py`, la línea y la etiqueta de la distancia, y la aceleración. | Se aceptó guardar las predicciones en disco y enviar las imágenes como PNG. |

## Análisis crítico

**Errores de la IA:**
- Propuso una ventana HU que no servía (200 a 1500): se cambió por la de una fuente de radiología.
- Propuso parada temprana y calentamiento de la tasa de aprendizaje sin que el entrenamiento los necesitara; se descartaron.
- El primer notebook de evaluación comparaba métricas distintas entre síy solo en validación, así que no dejaba ver el sobreajuste. Hubo que rehacerlo.
- Propuso la ecualización global del histograma. para el contraste; la figura de comparación mostró que realzaba el tejido blando y empeoraba la separación del hueso.
- La primera versión del modelo modificado colapsó al entrenar ya que la segmentación solo aprendía un hueso, porque la última capa del decoder tuvieron un problema de dying ReLU.  Se corrigió con LeakyReLU y el Dice de cada hueso por separado en la validación.

**Aciertos:**
- Las técnicas de memoria (gradient checkpointing y precisión mixta) permitieron entrenar el modelo 3D completo en 6 GB.
- La regla para los λ deja cada tarea con el peso que el equipo quiere al empezar, sin depender de la escala de cada pérdida.
- La ventana trabecular se eligió con una figura que comparaba las opciones en tres CT, y la fusión se eligió en validación.

**Limitaciones:**

- La IA no conocía todo el contexto del proyecto, por lo que cometió errores que debían ser corregidos para que los métodos y los resultados fueran los adecuados. 

