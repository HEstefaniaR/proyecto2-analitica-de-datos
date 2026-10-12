"""Visualizador de inferencia corte por corte: overlay 2D del pipeline sobre el CT crudo.

Detección (cajas), segmentación (máscara de cada fragmento) y clasificación (región por color),
con la distancia de separación visible junto a cada fragmento conminuto que cruza el corte.
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import binary_erosion

from dashboard import estado
from src import config

ALPHA_RELLENO = 0.35
COLOR_MEDIDA = "#ffd700"      # etiqueta y línea de la distancia de separación


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def _figura_corte(volumen: np.ndarray, res: dict, z: int) -> go.Figure:
    """Corte axial del CT con el overlay del pipeline.

    El CT (ventana del modelo), el relleno semitransparente de cada fragmento (mismo matiz por hueso,
    luminosidad por fragmento) y su contorno se componen en el servidor en UNA imagen RGB que viaja al
    navegador como PNG (decenas de KB en lugar de varios MB de matrices en JSON). Encima van, como trazas,
    las cajas detectadas que cruzan el corte, la etiqueta de distancia de cada fragmento conminuto presente
    (el principal no lleva etiqueta) y los extremos de la medición que caen en el corte.
    """
    spacing = res["spacing"]
    lo, hi = config.DASH_SLICE_WINDOW
    gris = (np.clip(volumen[z], lo, hi) - lo) / (hi - lo) * 255
    rgb = np.repeat(gris[..., None], 3, axis=2).astype(np.float32)
    fid_corte = res["fragmentos_id"][z]
    roles = estado.roles_fragmentos(res["fragmentos"])
    etiquetas = []
    for fid in np.unique(fid_corte[fid_corte > 0]).tolist():
        rol, color = roles[fid], _rgb(roles[fid]["color"])
        mascara = fid_corte == fid
        rgb[mascara] = (1 - ALPHA_RELLENO) * rgb[mascara] + ALPHA_RELLENO * color
        rgb[mascara & ~binary_erosion(mascara, iterations=2)] = color            # contorno de 2 px
        fila = res["fragmentos"][fid - 1]
        if not fila["principal"]:                    # el principal no tiene distancia: solo su color
            ys, xs = np.nonzero(mascara)
            etiquetas.append((float(xs.max()), float(ys.min()), f"<b>{rol['etiqueta']} · {fila['distancia_mm']:.1f} mm</b>"))

    f = px.imshow(rgb.astype(np.uint8), binary_string=True, binary_compression_level=4)
    f.update_traces(hovertemplate="x %{x} · y %{y}<extra></extra>")
    for x, y, texto in etiquetas:
        f.add_annotation(x=x, y=y, text=texto, showarrow=True, arrowcolor=COLOR_MEDIDA, arrowwidth=2, ax=45, ay=-30,
                         font=dict(size=15, color=COLOR_MEDIDA), bgcolor="rgba(0,0,0,0.75)", bordercolor=COLOR_MEDIDA,
                         borderpad=3)

    # Extremos de cada medición que caen en este corte (la medición es 3D: pueden estar en cortes distintos)
    for fila in res["fragmentos"]:
        p, q = fila.get("punto"), fila.get("punto_principal")
        if p is None or z not in (p[0], q[0]):
            continue
        if p[0] == q[0]:
            f.add_trace(go.Scatter(x=[p[2], q[2]], y=[p[1], q[1]], mode="lines+markers",
                                   line=dict(color=COLOR_MEDIDA, width=3), marker=dict(size=8, color=COLOR_MEDIDA),
                                   showlegend=False, hoverinfo="text",
                                   text=[f"medición · {fila['distancia_mm']:.1f} mm"] * 2))
        else:
            aqui, otro, cual = (p, q, "principal") if z == p[0] else (q, p, "fragmento")
            f.add_trace(go.Scatter(x=[aqui[2]], y=[aqui[1]], mode="markers",
                                   marker=dict(size=10, color=COLOR_MEDIDA, symbol="x"), showlegend=False,
                                   hoverinfo="text",
                                   text=[f"extremo de la medición ({fila['distancia_mm']:.1f} mm); el del {cual} "
                                         f"está en el corte {otro[0] + 1}"]))

    for caja in res["cajas_por_corte"].get(z, ()):
        xs = [caja["x0"], caja["x1"], caja["x1"], caja["x0"], caja["x0"]]
        ys = [caja["y0"], caja["y0"], caja["y1"], caja["y1"], caja["y0"]]
        f.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                               line=dict(color="#ffd700", width=2, dash="dash"), opacity=0.9,
                               showlegend=False, hoverinfo="text",
                               text=[f"detección · {config.REGION_LABELS.get(caja.get('hueso'), 'sin clase')} · "
                                     f"score {caja['score']:.2f}"]))

    alto, ancho = fid_corte.shape
    f.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=650,
                    xaxis=dict(range=[-0.5, ancho - 0.5], constrain="domain"),
                    yaxis=dict(range=[alto - 0.5, -0.5], scaleanchor="x", scaleratio=spacing[1] / spacing[0],
                               constrain="domain"))
    return f


@st.fragment
def _visor(volumen: np.ndarray, res: dict) -> None:
    """Slider y corte. Es un fragmento de Streamlit: mover el slider solo vuelve a ejecutar esta función, no la
    carga del volumen, la inferencia ni la tabla."""
    n_z = volumen.shape[0]
    z = st.slider("Corte axial (z)", 0, n_z - 1, n_z // 2)
    st.plotly_chart(_figura_corte(volumen, res, z), width="stretch")
    lo, hi = config.DASH_SLICE_WINDOW
    salida = {"fusion": "fusión de la cabeza semántica y la cabeza de máscara", "mask": "cabeza de máscara",
              "semantic": "cabeza semántica"}[res.get("salida", "semantic")]
    st.caption(f"Corte {z + 1} de {n_z} · ventana {lo} a {hi} HU (la del modelo) · máscara: {salida}, "
               "remuestreada a la rejilla original (vecino más cercano)")


def render(volumen: np.ndarray, spacing_xyz, ruta: str, etiqueta: str = "") -> None:
    """Vista completa: inferencia bajo demanda, overlay por corte y tabla de fragmentos."""
    st.header("Inferencia por corte")
    st.markdown(f"<span style='font-size:1.1rem'><b>Caso: {etiqueta}</b></span>", unsafe_allow_html=True)
    res = estado.pedir_inferencia(ruta)
    if res is None:
        st.stop()

    t = res["tiempos"]
    c1, c2, c3 = st.columns(3)
    origen = "resultado guardado en disco" if res.get("desde_disco") else f"dispositivo: {res['device']}"
    c1.metric("Pipeline", f"{t['pipeline']:.0f} s", origen)
    c2.metric("Preproceso", f"{t['preproceso']:.1f} s", f"distancias: {t['distancias']:.1f} s")
    c3.metric("Fragmentos", str(len(res["fragmentos"])))

    st.markdown("🟥 Sacro 🟦 Coxal izq. 🟩 Coxal der. 🟨 Región detectada (línea discontinua) y distancia de "
                "separación de los fragmentos conminutos (C1, C2…)")
    _visor(volumen, res)

    st.subheader("Fragmentos")
    st.dataframe(estado.df_fragmentos(res), hide_index=True)
    st.caption("Distancia: mínimo borde a borde entre el fragmento y el principal de su mismo hueso "
               "(EDT 3D en mm con el espaciado del header).")
