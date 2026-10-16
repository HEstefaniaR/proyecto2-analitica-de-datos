"""Dashboard PENGWIN: CT crudo y predicciones del modelo 3D en tres visualizadores.

Ejecutar con `uv run streamlit run dashboard/app.py` desde la raíz del proyecto. Las cargas de
volumen, modelo e inferencia quedan cacheadas por caso (`dashboard.estado`): elegir un caso carga
el CT al instante y la inferencia corre solo cuando el usuario la pide, compartida por las dos
vistas del modelo.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # una operación del CBAM 3D no existe en la GPU del Mac

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import estado
from dashboard.views import inferencia_corte, reconstruccion_3d, volumen_crudo

VISTAS = {"Volumen crudo": volumen_crudo,
          "Inferencia por corte": inferencia_corte,
          "Reconstrucción 3D": reconstruccion_3d}

st.set_page_config(page_title="PENGWIN", layout="wide")
st.markdown(
    '<div style="text-align:center;background:rgba(254,222,54,0.2);border:1px solid rgba(254,222,54,0.45);'
    'border-radius:0.5rem;padding:0.9rem 1rem;">⚠️ <b>Uso exclusivamente académico.</b> No es un dispositivo '
    "médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Caso")
    fuente = st.radio("Fuente", ["Caso del dataset", "Subir CT"])
    ruta, etiqueta = None, None
    if fuente == "Caso del dataset":
        casos = estado.casos_dataset()
        opciones = ["—"] + [f"{c['id']} · {c['split']}" for c in casos]
        eleccion = st.selectbox("Caso", opciones)
        if eleccion != "—":
            caso = casos[opciones.index(eleccion) - 1]
            ruta, etiqueta = caso["ruta"], eleccion
        st.caption("Casos de test: no se usaron para entrenar ni para elegir el modelo")
    else:
        formato = st.radio("Formato", ["Volumen (un archivo)", "Serie de cortes (carpeta)"])
        modo = "serie" if formato.startswith("Serie") else "volumen"
        subida = estado.subida_por_partes(modo)
        if subida and modo == "volumen":
            ruta, etiqueta = subida["ruta"], subida["nombre"]
        elif subida:
            try:
                info = estado.volumen_desde_cortes(subida["ruta"])
                if info["dz_archivo"] is None:
                    st.warning("Los cortes no traen la distancia entre ellos (eje z). Por defecto se usa 1,0 mm; "
                               "cámbiala si la conoces.")
                    dz = st.number_input("Distancia entre cortes (mm)", min_value=0.1, max_value=10.0,
                                         value=1.0, step=0.1, format="%.2f")
                    info = estado.volumen_desde_cortes(subida["ruta"], dz)
                ruta = info["ruta"]
                etiqueta = f"{subida['nombre']} ({info['cortes']} cortes, z = {info['dz']:.2f} mm)"
                st.caption(f"{info['cortes']} cortes unidos en un volumen")
            except ValueError as e:
                st.error(str(e))

    vista = st.radio("Visualizador", list(VISTAS))

if ruta is None:
    st.stop()

volumen, spacing = estado.cargar_volumen(ruta)
VISTAS[vista].render(volumen, spacing, ruta, etiqueta)
