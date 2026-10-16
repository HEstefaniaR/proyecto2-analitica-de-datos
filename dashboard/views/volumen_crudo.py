"""Visualizador de volumen crudo: corte a corte, MIP y reconstrucción por umbral HU, sin modelo."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from dashboard import estado
from src import config


def _gris(imagen: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Imagen en grises de 8 bits con la ventana [lo, hi]: viaja al navegador como PNG, no como matriz en JSON."""
    return ((np.clip(imagen, lo, hi) - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)


@st.cache_data(show_spinner=False, max_entries=4)
def _mascara_cacheada(ruta: str, rango_hu: tuple[float, float], min_voxels: int) -> np.ndarray:
    """Máscara de hueso por umbral de un caso: se calcula una vez por combinación de parámetros."""
    return bone_mask(estado.cargar_volumen(ruta)[0], rango_hu, min_voxels)


@st.cache_data(show_spinner=False, max_entries=4)
def _isosuperficie_cacheada(ruta: str, rango_hu: tuple[float, float], min_voxels: int, step_size: int) -> go.Figure:
    volumen, spacing = estado.cargar_volumen(ruta)
    return build_isosurface(volumen, _mascara_cacheada(ruta, rango_hu, min_voxels), spacing, rango_hu, step_size)


def bone_mask(volume: np.ndarray, rango_hu: tuple[float, float], min_voxels: int) -> np.ndarray:
    """Máscara binaria del rango HU [lo, hi], sin las componentes conexas menores que `min_voxels`.

    Dentro de ese rango, el volumen queda con decenas de miles de componentes conexas: unas pocas
    corresponden a hueso y el resto son vóxeles sueltos de ruido. Descartarlas por tamaño
    limpia tanto la vista MIP como la reconstrucción 3D sin usar ningún modelo.
    """
    lo, hi = rango_hu
    mask = (volume >= lo) & (volume <= hi)
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


def build_isosurface(volume: np.ndarray, mask: np.ndarray, spacing_xyz, rango_hu: tuple[float, float], step_size: int) -> go.Figure:
    """Reconstrucción 3D por isosuperficie (marching cubes), restringida a `mask`.

    La superficie se extrae en el límite inferior del rango: es la frontera del conjunto
    {lo ≤ HU ≤ hi}. `step_size` se pasa directamente a marching_cubes (no se remuestrea el volumen
    antes), y la malla se envía al navegador en float32/int32 para no exceder el límite de tamaño
    de mensaje de Streamlit.
    """
    lo, _ = rango_hu
    vol = np.where(mask, volume, lo - 1)
    verts, faces, _, _ = marching_cubes(vol, level=lo, spacing=spacing_xyz[::-1], step_size=step_size)
    verts, faces = verts.astype(np.float32), faces.astype(np.int32)
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=config.DASH_BONE_COLOR, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.25, roughness=0.6, fresnel=0.1),
        lightposition=dict(x=200, y=200, z=300),
    )])
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


@st.fragment
def slice_viewer(volume: np.ndarray, spacing_xyz) -> None:
    """Corte axial del volumen crudo con un slider en z y una ventana HU.

    Es un fragmento de Streamlit: mover los sliders solo vuelve a ejecutar esta función, no el
    cálculo de la máscara de hueso ni la malla 3D del resto de la página.
    """
    n_z = volume.shape[0]
    c1, c2 = st.columns([3, 2])
    z = c1.slider("Corte axial (z)", min_value=0, max_value=n_z - 1, value=n_z // 2)
    lo, hi = c2.slider("Ventana HU", min_value=-1000, max_value=3000, value=config.DASH_SLICE_WINDOW, step=50)
    # El corte viaja como PNG de 8 bits. Filas = Y (posterior): la imagen deja anterior arriba, como en el MIP axial
    f = px.imshow(_gris(volume[z], lo, hi), binary_string=True, binary_compression_level=4)
    f.update_traces(hovertemplate="x %{x} · y %{y}<extra></extra>")
    f.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=650,
                    yaxis=dict(scaleanchor="x", scaleratio=spacing_xyz[1] / spacing_xyz[0], constrain="domain"),
                    xaxis=dict(constrain="domain"))
    st.plotly_chart(f, width="stretch")
    st.caption(f"Corte {z + 1} de {n_z} · {z * spacing_xyz[2]:.1f} mm desde el primer corte · "
               f"píxel {spacing_xyz[0]:.2f} x {spacing_xyz[1]:.2f} mm, corte de {spacing_xyz[2]:.2f} mm · ventana {lo} a {hi} HU")


def render(volumen: np.ndarray, spacing_xyz, ruta: str, etiqueta: str = "") -> None:
    """Vista completa del CT crudo: parámetros, corte a corte, MIP y reconstrucción."""
    st.header("Volumen crudo")
    st.markdown(f"<span style='font-size:1.1rem'><b>Caso: {etiqueta}</b></span>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Volumen crudo")
        umbral_lo, umbral_hi = st.slider(
            "Umbral HU (hueso)", min_value=-200, max_value=600,
            value=config.DASH_HU_RANGE, step=25,
            help="Rango de hueso [lo, hi]: cuenta como hueso lo que cae entre las dos asas.",
        )
        min_component_mm3 = st.slider(
            "Volumen mínimo de estructura (mm³)", min_value=0, max_value=5000, value=config.DASH_MIN_COMPONENT_MM3, step=100,
            help="Descarta del umbral las estructuras desconectadas más pequeñas que este volumen "
                 "(vóxeles de ruido disperso), sin afectar al hueso.",
        )
        mesh_step = st.select_slider(
            "Resolución de la malla 3D", options=[2, 3, 4, 6], value=config.DASH_MESH_STEP,
            help="Paso de muestreo de la isosuperficie. Valores más bajos dan más detalle pero una "
                 "malla más pesada de transmitir al navegador.",
        )

    voxel_mm3 = float(np.prod(spacing_xyz))
    min_voxels = round(min_component_mm3 / voxel_mm3)
    with st.spinner("Procesando el volumen: máscara de hueso y proyecciones…"):
        mask = _mascara_cacheada(ruta, (umbral_lo, umbral_hi), min_voxels)

    tab_cortes, tab_mip, tab_3d = st.tabs(["Corte a corte", "MIP 2D (3 planos)", "Reconstrucción 3D (umbral)"])

    with tab_cortes:
        slice_viewer(volumen, spacing_xyz)

    with tab_mip:
        axial, coronal, sagital = orthogonal_mips(volumen, mask, config.DASH_HU_CLIP_MAX)
        aspect_coronal = spacing_xyz[2] / spacing_xyz[0]   # z / x
        aspect_sagital = spacing_xyz[2] / spacing_xyz[1]   # z / y
        # Filas del array = Y (posterior) en el axial y Z (superior) en el coronal/sagital. El eje y
        # de Heatmap por defecto pone la fila 0 abajo, así que solo el axial necesita invertirlo
        # (para que quede anterior arriba); coronal y sagital ya quedan con la cabeza arriba sin invertir.
        c1, c2, c3 = st.columns(3)
        for col, (nombre, imagen, aspecto, invertir) in zip(
            (c1, c2, c3),
            [("Axial", axial, 1.0, True), ("Coronal", coronal, aspect_coronal, False), ("Sagital", sagital, aspect_sagital, False)],
        ):
            # La imagen pone la fila 0 arriba: coronal y sagital se voltean para dejar la cabeza arriba
            f = px.imshow(_gris(imagen if invertir else imagen[::-1], 0, config.DASH_HU_CLIP_MAX), binary_string=True,
                          binary_compression_level=4)
            f.update_traces(hoverinfo="skip", hovertemplate=None)
            f.update_layout(
                title=nombre, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(constrain="domain"),
                yaxis=dict(scaleanchor="x", scaleratio=aspecto, constrain="domain"),
            )
            col.plotly_chart(f, width="stretch")

    with tab_3d:
        with st.spinner("Reconstruyendo la superficie 3D del hueso…"):
            fig = _isosuperficie_cacheada(ruta, (umbral_lo, umbral_hi), min_voxels, mesh_step)
            st.plotly_chart(fig, width="stretch")
