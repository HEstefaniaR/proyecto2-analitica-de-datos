# proyecto2-analitica-de-datos

## Reproducción

Entorno con `uv` (`pyproject.toml` + `uv.lock`; `requirements.txt` exportado de ahí). Semilla fija: `config.SEED = 99`.
Todos los comandos se ejecutan desde la raíz del proyecto.

```bash
uv sync
```

### Datos y splits

```bash
uv run python -m src.dataset.splits
```

### Detección: tres vistas, backbone FundidoraFPN + CBAM con γ, cabeza con anclas

Todos los parámetros están en `src/config.py`.

- **Preprocesamiento** (`src/dataset/preprocessing.py`, igual en entrenamiento e inferencia): reorientación a LPS,
  cubo isótropo de 400 mm con 256³ vóxeles (1,5625 mm; relleno con aire, centrado en el FOV y en los cortes con hueso)
  y ventana HU 50-1300 → [0, 1].
- **Modelo** (`src/models/detection.py`): cortes 2.5D (corte y vecinos como 3 canales) de las vistas axial, coronal y
  sagital; un backbone FundidoraFPN compartido por las tres, con CBAM + γ aprendible (`y = x + γ·CBAM(x)`, γ inicia en 0)
  en los bloques 3 y 4 del encoder; una cabeza con anclas por vista (como la RPN del taller 3) con CBAM + γ; NMS propio;
  fusión de las tres vistas en una caja 3D por región (pesos de QuickNAT). La vista sagital predice sacro y "coxal" y el
  lado se asigna por la posición del corte.
- **Modelo 3D** (`src/models/model3d.py`): backbone Fundidora 3D (Conv3d + InstanceNorm, CBAM 3D + γ en los bloques 3
  y 4, decoder U-Net 3D hasta P3 y P2) sobre el cubo completo, con la misma detección 2D por corte del modelo 2.5D (cabeza
  con anclas por vista, pérdida, NMS y métricas). Un paso = un CT completo; gradient checkpointing para GPU de 6 GB.
- **Salidas**: `results/detection/tres_vistas_cbam/` y `results/detection/tres_vistas_sin_cbam/` (ablación sin atención);
  `results/detection/unet3d_cbam/` para el modelo 3D.

1. Caché del cubo (una vez; ~3,3 GB en `data/cache_cubo_400mm_256/`):

```bash
uv run python -m src.models.detection cache
```

2. Prueba de correctitud (overfit de un batch fijo de 8 cortes por vista):

```bash
uv run python -m src.models.detection overfit
```

```bash
uv run python -m src.models.model3d overfit
```

Memoria y tiempo de un paso del modelo 3D (antes de entrenarlo en una GPU nueva):

```bash
uv run python -m src.models.model3d bench
```

3. Entrenamiento desde cero, sin congelar nada (si se interrumpe, el mismo comando con `--resume` continúa desde `last.pth`):

```bash
uv run python -m src.models.detection train
```

```bash
uv run python -m src.models.detection train --no-cbam
```

```bash
uv run python -m src.models.model3d train
```

Evaluación del modelo 3D en validación y test (`results/detection/<run>/eval/`):

```bash
uv run python -m src.models.model3d evaluate --run unet3d_cbam_trab
```

Inferencia sobre un CT cualquiera (`.mha`, `.nii`, `.nii.gz` o carpeta DICOM, con cualquier número de cortes). Escribe
en `results/inferencia/<nombre>/` las máscaras de los tres huesos en la rejilla del CT, las cajas de cada corte
(`cajas.csv`), el resumen por corte (`cortes.csv`) y `resumen.json`. La máscara recomendada es
`mascara_fusion.nii.gz` (coxales de la cabeza de máscara y sacro de la semántica):

```bash
uv run python -m src.models.model3d infer ruta/al/ct.nii.gz
```

4. Evaluación en validación (métricas por vista, fusión 3D, Grad-CAM, latencia) y prueba de reconstrucción del extractor:

```bash
uv run python -m src.models.detection evaluate --run tres_vistas_cbam
```

```bash
uv run python -m src.models.detection evaluate --run tres_vistas_sin_cbam
```

```bash
uv run python -m src.models.detection probe --run tres_vistas_cbam
```

5. Tabla comparativa (`results/detection/comparison.csv`):

```bash
uv run python -m src.models.detection summary
```

Salidas en `results/detection/<run>/`: `best.pth` (pesos), `last.pth` (estado para reanudar; no se sube al repositorio),
`config.json`, `history.csv` (pérdidas y métricas de train y validación y γ de cada capa por época), `metrics.json`,
`per_slice.csv`, `fusion_3d.csv`, `predictions_<vista>.png`, `gradcam_<vista>.png`, `gradcam_scores.csv` y
`probe_p2.json` / `probe_p2.png`. El notebook `notebooks/02_backbone_deteccion.ipynb` solo lee estos archivos.
