"""Visualizador de reconstrucción 3D final: fragmentos del modelo como mallas interactivas.

Apila las máscaras predichas (remuestreadas a la rejilla original del CT), levanta cada fragmento
con marching cubes y lo dibuja como malla Mesh3d coloreada por el macro-hueso asignado, con la
distancia de separación visible en la etiqueta 3D de cada fragmento.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage.measure import marching_cubes

from dashboard import estado
from src import config


COLOR_MEDIDA = "#ffd700"      # línea y etiqueta de la distancia de separación


def _malla(mascara: np.ndarray, spacing_zyx, step: int):
    """Vértices (mm, z/y/x) y caras de la isosuperficie de un fragmento.

    El fragmento se recorta a su bounding box con un vóxel de margen: la superficie no toca el
    borde del bloque y marching cubes no genera caras abiertas. Los vértices se devuelven en mm
    absolutos del CT con el espaciado del header (índice de vóxel × espaciado), la misma referencia de los puntos
    de la medición. Si el muestreo grueso salta la superficie
    (fragmentos finos), se reintenta con paso menor hasta 1.
    """
    idx = np.argwhere(mascara)
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, mascara.shape)   # +2: el recorte es exclusivo por arriba
    bloque = mascara[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    for paso in range(step, 0, -1):
        try:
            verts, faces, _, _ = marching_cubes(bloque, level=0.5, spacing=spacing_zyx, step_size=paso)
        except RuntimeError:
            continue
        # `marching_cubes(spacing=...)` ya devuelve los vértices en mm dentro del bloque: solo falta trasladarlos
        # al origen del bloque, también en mm
        return (verts + lo * np.asarray(spacing_zyx)).astype(np.float32), faces.astype(np.int32)
    raise RuntimeError("fragmento sin superficie ni con paso 1: máscara degenerada")


RADIO_PARCHE_MM = 12.0        # medio lado del parche de detalle alrededor del extremo de la medición en el principal


def _parche(mascara: np.ndarray, punto_idx, spacing_zyx):
    """Superficie del principal con el máximo detalle (paso 1) en una caja alrededor de `punto_idx` (z, y, x).

    La malla del principal se dibuja con el paso del control (simplificada); alrededor del extremo de la medición
    se superpone este parche fino para que el punto toque la superficie que se ve. El bloque no se rellena con
    ceros: la superficie queda abierta en las caras de la caja, sin tapas dentro del hueso."""
    sp = np.asarray(spacing_zyx)
    r = np.maximum(np.ceil(RADIO_PARCHE_MM / sp).astype(int), 3)
    lo = np.maximum(np.array(punto_idx) - r, 0)
    hi = np.minimum(np.array(punto_idx) + r + 1, mascara.shape)
    bloque = mascara[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if bloque.min() == bloque.max():
        return None
    verts, faces, _, _ = marching_cubes(bloque, level=0.5, spacing=tuple(sp))
    return (verts + lo * sp).astype(np.float32), faces.astype(np.int32)


def _al_vertice(punto_mm: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vértice de la malla más cercano a un punto (mm, z/y/x)."""
    return verts[np.argmin(((verts - punto_mm) ** 2).sum(1))]


def _figura(res: dict, step: int) -> go.Figure:
    """Malla por fragmento coloreada por región y, en cada conminuto, la línea de la medición (de borde a borde)
    con su distancia de separación."""
    spacing_zyx = res["spacing"][::-1]
    roles = estado.roles_fragmentos(res["fragmentos"])
    fig = go.Figure()
    regiones_vistas: set[str] = set()
    anotaciones: list[dict] = []
    mallas = {}
    for fid, fila in enumerate(res["fragmentos"], start=1):
        # Los conminutos son pequeños: siempre con el máximo detalle, para que no se pierdan ni se deformen
        mallas[fid] = _malla(res["fragmentos_id"] == fid, spacing_zyx, step if fila["principal"] else 1)
    principal_de = {fila["hueso"]: fid for fid, fila in enumerate(res["fragmentos"], start=1) if fila["principal"]}
    for fid, fila in enumerate(res["fragmentos"], start=1):
        verts, faces = mallas[fid]
        rol = roles[fid]
        region = fila["region"]
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=rol["color"], name=config.REGION_LABELS[region],
            showlegend=region not in regiones_vistas,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.25, roughness=0.6, fresnel=0.1),
            lightposition=dict(x=200, y=200, z=300),
        ))
        regiones_vistas.add(region)
        if fila["principal"] or fila.get("punto") is None:
            continue
        # Medición del conminuto: línea entre los dos puntos más cercanos (borde del fragmento y borde del
        # principal), con la distancia en una etiqueta de fondo oscuro para que se lea sobre cualquier tema.
        # Los extremos se miden entre centros de vóxel del borde y la superficie pasa medio vóxel más afuera: cada
        # extremo se lleva al vértice más cercano de una malla de paso 1 (la del conminuto y un parche fino del
        # principal alrededor del punto), así el marcador toca la superficie dibujada moviéndose menos de un vóxel.
        fid_p = principal_de[fila["hueso"]]
        parche = _parche(res["fragmentos_id"] == fid_p, fila["punto_principal"], spacing_zyx)
        p = _al_vertice(np.array(fila["punto"]) * spacing_zyx, mallas[fid][0])
        q = np.array(fila["punto_principal"]) * spacing_zyx
        if parche is not None:
            pv, pf = parche
            q = _al_vertice(q, pv)
            fig.add_trace(go.Mesh3d(x=pv[:, 2], y=pv[:, 1], z=pv[:, 0], i=pf[:, 0], j=pf[:, 1], k=pf[:, 2],
                                    color=roles[fid_p]["color"], showlegend=False, hoverinfo="skip",
                                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.25, roughness=0.6, fresnel=0.1),
                                    lightposition=dict(x=200, y=200, z=300)))
        fig.add_trace(go.Scatter3d(
            x=[p[2], q[2]], y=[p[1], q[1]], z=[p[0], q[0]], mode="lines+markers",
            line=dict(color=COLOR_MEDIDA, width=8), marker=dict(size=5, color=COLOR_MEDIDA, line=dict(color="#000", width=1)),
            showlegend=False, hoverinfo="text",
            text=[f"{rol['etiqueta']} · borde del fragmento", f"{rol['etiqueta']} · borde del principal"],
        ))
        m = (p + q) / 2
        anotaciones.append(dict(
            x=float(m[2]), y=float(m[1]), z=float(m[0]), text=f"<b>{rol['etiqueta']} · {fila['distancia_mm']:.1f} mm</b>",
            showarrow=True, arrowcolor=COLOR_MEDIDA, arrowwidth=2, ax=40, ay=-40,
            font=dict(size=15, color=COLOR_MEDIDA), bgcolor="rgba(0,0,0,0.75)", bordercolor=COLOR_MEDIDA, borderpad=3,
        ))

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                   annotations=anotaciones),
        margin=dict(l=0, r=0, t=0, b=0), height=700,
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=6)
def _figura_cacheada(ruta: str, step: int) -> go.Figure:
    """La figura 3D de un caso con un paso de malla: se construye una vez (marching cubes de cada fragmento) y se
    reutiliza al volver a la vista o al cambiar otro control."""
    return _figura(estado.resultado_inferencia(ruta), step)


def render(volumen: np.ndarray, spacing_xyz, ruta: str, etiqueta: str = "") -> None:
    """Vista completa: inferencia bajo demanda (compartida con la vista de cortes) y mallas."""
    st.header("Reconstrucción 3D")
    st.markdown(f"<span style='font-size:1.1rem'><b>Caso: {etiqueta}</b></span>", unsafe_allow_html=True)
    res = estado.pedir_inferencia(ruta)
    if res is None:
        st.stop()

    with st.sidebar:
        st.subheader("Reconstrucción 3D")
        step = st.select_slider(
            "Resolución de la malla", options=[2, 3, 4, 6], value=config.DASH_MESH_STEP,
            help="Paso de muestreo de marching cubes sobre la máscara de cada fragmento. Valores "
                 "más bajos dan más detalle pero una malla más pesada de transmitir al navegador.",
        )

    with st.spinner("Reconstruyendo la superficie 3D de los fragmentos…"):
        fig = _figura_cacheada(ruta, step)
        st.plotly_chart(fig, width="stretch")
    st.caption("Fragmentos del pipeline coloreados por macro-hueso asignado. El principal de cada hueso lleva el "
               "color pleno; los conminutos (C1, C2…), un tono más claro. La línea amarilla une los dos puntos más "
               "cercanos entre el borde del conminuto y el borde del principal, con la distancia de separación en mm "
               "(espaciado del header). La distancia se mide sobre los vóxeles de la máscara; los extremos de la "
               "línea se apoyan en la superficie (alrededor de cada extremo el principal se dibuja con detalle fino). Mallas por marching cubes sobre las máscaras del modelo remuestreadas a "
               "la rejilla original.")

    with st.expander("Fragmentos"):
        st.dataframe(estado.df_fragmentos(res), hide_index=True)
