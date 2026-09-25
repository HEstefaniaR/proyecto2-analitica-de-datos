"""Visualizador 1: reconstrucción 3D del volumen crudo (MIP), umbral HU, sin modelo.

Ejecutar con `uv run streamlit run dashboard/app.py` desde la raíz del proyecto.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src import config
from src.dataset import preprocessing as pre

# Umbral y clip del visualizador: independientes de la ventana HU del pipeline del modelo
# (config.HU_WINDOW), porque este visualizador trabaja sobre el volumen sin procesar.
HU_THRESHOLD_DEFAULT = 100
HU_CLIP_MAX = 1300
MESH_STEP_DEFAULT = 3   # step_size de marching_cubes; menor = malla más detallada y más pesada
MIN_COMPONENT_MM3_DEFAULT = 1000   # descarta estructuras umbralizadas más pequeñas que esto (ruido disperso)
BONE_COLOR = "#e3d5b8"

st.set_page_config(page_title="PENGWIN — Visualizador 1", layout="wide")
st.title("Visualizador 1 · Volumen crudo (MIP)")
st.warning(
    "Uso exclusivamente académico. No es un dispositivo médico, no ha sido validado "
    "clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales."
)


def list_dataset_cases() -> dict[str, Path]:
    return {p.stem: p for d in config.IMAGE_DIRS for p in sorted(d.glob("*.mha"))}


def save_upload(file) -> Path:
    suffix = "".join(Path(file.name).suffixes) or ".mha"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.getbuffer())
    tmp.close()
    return Path(tmp.name)


@st.cache_data(show_spinner="Cargando volumen…")
def load_volume(path_str: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Carga y reorienta el volumen (mismo paso de orientación del pipeline). Sin crop ni
    ventaneo: ese preprocesamiento es específico del modelo, no del visualizador crudo."""
    img = pre.reorient(pre.load_volume(path_str))
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()


def bone_mask(volume: np.ndarray, threshold: float, min_voxels: int) -> np.ndarray:
    """Máscara binaria del umbral HU, sin las componentes conexas menores que `min_voxels`.

    A ese umbral, el volumen queda con decenas de miles de componentes conexas: unas pocas
    corresponden a hueso y el resto son vóxeles sueltos de ruido. Descartarlas por tamaño
    limpia tanto la vista MIP como la reconstrucción 3D sin usar ningún modelo.
    """
    mask = volume >= threshold
    if min_voxels <= 0:
        return mask
    labels, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    return np.isin(labels, 1 + np.flatnonzero(sizes >= min_voxels))


def orthogonal_mips(volume: np.ndarray, mask: np.ndarray, clip_max: float):
    """Proyección de máxima intensidad en los tres planos, restringida a `mask`."""
    vol = np.where(mask, np.clip(volume, 0, clip_max), 0)
    return vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)   # axial, coronal, sagital


def build_isosurface(volume: np.ndarray, mask: np.ndarray, spacing_xyz, threshold: float, step_size: int) -> go.Figure:
    """Reconstrucción 3D por isosuperficie (marching cubes), restringida a `mask`.

    `step_size` se pasa directamente a marching_cubes (no se remuestrea el volumen antes), y la
    malla se envía al navegador en float32/int32 para no exceder el límite de tamaño de mensaje
    de Streamlit.
    """
    vol = np.where(mask, volume, threshold - 1)
    verts, faces, _, _ = marching_cubes(vol, level=threshold, spacing=spacing_xyz[::-1], step_size=step_size)
    verts, faces = verts.astype(np.float32), faces.astype(np.int32)
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=BONE_COLOR, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.25, roughness=0.6, fresnel=0.1),
        lightposition=dict(x=200, y=200, z=300),
    )])
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


with st.sidebar:
    st.header("Volumen")
    fuente = st.radio("Fuente", ["Caso del dataset", "Subir archivo"])
    if fuente == "Caso del dataset":
        casos = list_dataset_cases()
        case_id = st.selectbox("Caso", sorted(casos))
        volume_path = str(casos[case_id])
    else:
        archivo = st.file_uploader("Volumen CT (.mha, .nii, .nii.gz)", type=["mha", "nii", "gz"])
        volume_path = str(save_upload(archivo)) if archivo else None

    st.header("Parámetros")
    umbral_hu = st.slider("Umbral HU (hueso)", min_value=-200, max_value=600, value=HU_THRESHOLD_DEFAULT, step=25)
    min_component_mm3 = st.slider(
        "Volumen mínimo de estructura (mm³)", min_value=0, max_value=5000, value=MIN_COMPONENT_MM3_DEFAULT, step=100,
        help="Descarta del umbral las estructuras desconectadas más pequeñas que este volumen "
             "(vóxeles de ruido disperso), sin afectar al hueso.",
    )
    mesh_step = st.select_slider(
        "Resolución de la malla 3D", options=[2, 3, 4, 6], value=MESH_STEP_DEFAULT,
        help="Paso de muestreo de la isosuperficie. Valores más bajos dan más detalle pero una "
             "malla más pesada de transmitir al navegador.",
    )

if volume_path is None:
    st.info("Selecciona un caso del dataset o sube un volumen para visualizar.")
    st.stop()

try:
    volumen, spacing = load_volume(volume_path)
except ValueError as e:
    st.error(str(e))
    st.stop()

voxel_mm3 = float(np.prod(spacing))
min_voxels = round(min_component_mm3 / voxel_mm3)
mask = bone_mask(volumen, umbral_hu, min_voxels)

tab_mip, tab_3d = st.tabs(["MIP 2D (3 planos)", "Reconstrucción 3D"])

with tab_mip:
    axial, coronal, sagital = orthogonal_mips(volumen, mask, HU_CLIP_MAX)
    aspect_coronal = spacing[2] / spacing[0]   # z / x
    aspect_sagital = spacing[2] / spacing[1]   # z / y
    # Filas del array = Y (posterior) en el axial y Z (superior) en el coronal/sagital. El eje y
    # de Heatmap por defecto pone la fila 0 abajo, así que solo el axial necesita invertirlo
    # (para que quede anterior arriba); coronal y sagital ya quedan con la cabeza arriba sin invertir.
    c1, c2, c3 = st.columns(3)
    for col, (nombre, imagen, aspecto, invertir) in zip(
        (c1, c2, c3),
        [("Axial", axial, 1.0, True), ("Coronal", coronal, aspect_coronal, False), ("Sagital", sagital, aspect_sagital, False)],
    ):
        f = go.Figure(go.Heatmap(z=imagen, colorscale="gray", showscale=False))
        f.update_layout(
            title=nombre, margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(scaleanchor="x", scaleratio=aspecto, autorange="reversed" if invertir else True),
        )
        col.plotly_chart(f, width="stretch")

with tab_3d:
    fig = build_isosurface(volumen, mask, spacing, umbral_hu, mesh_step)
    st.plotly_chart(fig, width="stretch")
