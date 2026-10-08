# proyecto2-analitica-de-datos

## Reproducción

Entorno con `uv` (`pyproject.toml` + `uv.lock`; `requirements.txt` exportado de ahí). Semilla fija: `config.SEED = 99`.
Todos los comandos se ejecutan desde la raíz del proyecto.

```bash
uv sync
```

### Detección: backbone FundidoraPC 3D + CBAM 3D con γ, detección 2D por corte

Todos los parámetros están en `src/config.py`.

- **Preprocesamiento** (`src/dataset/preprocessing.py`, igual en entrenamiento e inferencia): reorientación a LPS,
  cubo isótropo de 400 mm con 256³ vóxeles (1,5625 mm; relleno con aire, centrado en el FOV y en los cortes con hueso)
  y ventana HU 50-1300 → [0, 1].
- **Modelo** (`src/models/model3d.py`): el cubo completo entra a un backbone FundidoraPC 3D (Conv3d + InstanceNorm +
  ReLU, CBAM 3D + γ aprendible en los bloques 3 y 4, `y = x + γ·CBAM(x)` con γ que inicia en 0, cuello dilatado y
  decoder U-Net 3D hasta P3 y P2). La detección es 2D por corte: para cada corte axial, coronal o sagital se toman P3 y P2
  en esa posición y una cabeza con anclas por vista (como la RPN del taller 3, con CBAM + γ) predice las cajas; NMS
  propio. La vista sagital predice sacro y "coxal" y el lado se asigna por la posición del corte. Un paso = un CT
  completo; gradient checkpointing para caber en una GPU de 6 GB.
- **Componentes compartidos** (`src/models/detection.py`): caché del cubo, anclas, cabeza con anclas, asignación,
  pérdida (BCE + SmoothL1), NMS y métricas.

1. Caché del cubo (una vez; ~3,3 GB en `data/cache_cubo_400mm_256/`):

```bash
uv run python -m src.models.detection cache
```

2. Memoria y tiempo de un paso de entrenamiento:

```bash
uv run python -m src.models.model3d bench
```

3. Prueba de correctitud (overfit de 2 CT sin aumento; salida en `results/detection/overfit_unet3d_cbam/metrics.json`):

```bash
uv run python -m src.models.model3d overfit
```

El notebook `notebooks/02_backbone_deteccion.ipynb` solo lee estos resultados.
