# Prompts

Se utilizó el modelo de IA Claude Opus 5.5.


---

## 1. División de los datos y EDA 
**Prompt.** «Necesito que la división train / validación / test no dependa del notebook del EDA. Hazla en un script aparte que se corra una sola vez, por caso, 70 / 15 / 15, estratificada por tipo de fractura y con semilla fija, y que guarde un archivo. El notebook solo debe leer ese archivo para mostrar cómo quedaron las clases. En el visualizador pon primero el MIP, que es el obligatorio según la rúbrica. La ventana HU es de 50 a 1300, según la fuente de radiología que te paso.»


- **Decisiones del equipo:** la IA propuso tres estratos (sacro fracturado, coxal unilateral o sin fractura, coxal bilateral) y se aceptaron. Sin embargo la IA propuso un umbral HU de ventaneo de 200 a 1500 que se rechazó y se cambió por el de una fuente de radiología (50 a 1300). 


## 2. Modelo 3D multitarea 

**Prompt.** «Voy a intentar un backbone 3D con detección 2D por corte, entrenando en una RTX 3060 de 6 GB. Usa el siguiente backbone: U-Net 3D con un encoder de cuatro bloques (16, 32, 64 y 128 canales, con InstanceNorm), un cuello con dos convoluciones dilatadas y un decoder que vuelve a la resolución completa, con sus cabezas de segmentación y de clasificación. La detección sigue siendo la nuestra, por corte, con anclas y NMS propio.  Entrena 140 épocas.»


- **Decisiones del equipo:**  la IA propuso gradient checkpointing y precisión mixta ya que en la tarjeta gráfica que usamos para entrenar no cabía el modelo sin esto y se utilizó. Para el cuello de botella dijo tres tamaños y se eligió el intermedio (128 canales). Además propuso earlystop, calentamiento del lr, pero eso se descartó.

## 3. Pérdida multitarea
**Prompt.** «Los λ de la pérdida no pueden quedarse en 1. Lo ideal es darle más peso a la segmentación, y que la detección y la clasificación pesen igual.»


- **Decisiones del equipo:** la IA propuso calcular cada λ como prioridad / pérdida inicial medida con el modelo sin entrenar. Se aceptó y elegimos darle prioridad a las cabezas de la siguiente forma: segmentación 2, detección 1,5 y clasificación 1,5.

## 4. Evaluación y notebooks

**Prompt.** «Haz la evaluación final del mejor modelo en validación y test, con las métricas de la rúbrica. Cada curva debe comparar entrenamiento contra validación. Agrega HD95 y ASSD para cada cabeza de segmentación, igual que Dice e IoU.»


- **Decisiones del equipo:** la primera versión del notebook se rechazó ya que presentaba gráficas que comparaban métricas diferentes solo en validación. La segunda versión se aceptó. 

## 5. Cambio del modelo 3D inicial

**Prompt.** «Usar las cajas predichas como zona de búsqueda de una cabeza de máscara con ROI Align, como Mask R-CNN; mejorar el contraste del preprocesamiento; y poner atención en las primeras capas del decoder. Haz una propuesta y figuras para decidir. Lo que decidimos como equipo: (1) cabeza de máscara 2D por corte, sobre cada caja axial, entrenada junto con el resto pero sin que su pérdida llegue al backbone, para no afectar al detector; (2) un segundo canal de entrada con la ventana trabecular de 100 a 700 HU, que fue la que ganó en la figura; (3) CBAM con γ en la primera capa del decoder y en la detección, y quitarlo del bloque 3 del encoder, donde γ había quedado en cero; (4) un segundo modelo igual pero con squeeze-excitation en los mismos lugares, para la ablación; (5) el optimizador sigue siendo AdamW; (6) la clasificación pasa a ser la rama de clase de cada caja, que elige la máscara.»

- **Decisiones del equipo**
  - Cabeza de máscara 2D por corte o 3D por región: se eligió 2D.
  - Entrenarla aparte, junto con gradiente o junto sin gradiente: se eligió junto y sin gradiente al backbone.
  - Contraste (ecualización global, CLAHE o ventanas como canales): se eligió la ventana trabecular.
  - Qué clasifica la cabeza de clasificación (cada hueso, cada fragmento o cada caja): se eligió cada caja.



## 6. Resultados finales, fusión, inferencia y verificación 

**Prompt.** «Ya están entrenados los dos modelos. Evalúalos y actualiza los notebooks 02, 03 y 04 con el modelo nuevo. Compara la segmentación semántica con la cabeza de máscara en la misma rejilla y dime si conviene unirlas.»


- **Decisiones del equipo:** de tres formas de fusión se eligió la que dio mejor resultado en validación (`union_sacro`). Otros postprocesos que propuso la IA quedaron guardados sin implementar.

## 7. Dashboard 

**Prompt.** «Integra el dashboard con el modelo final para los tres visualizadores. En la reconstrucción 3D y en el corte quiero ver la medición: la línea entre los dos puntos más cercanos y la etiqueta con la distancia. Hazlo más rápido, porque cada movimiento del slider tarda mucho.»


- **Decisiones del equipo:** se aceptó guardar las predicciones en disco y enviar las imágenes como PNG.


