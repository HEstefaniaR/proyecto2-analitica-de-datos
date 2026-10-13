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
    fuente = st.radio("Fuente", ["Caso del dataset", "Subir archivo"])
    ruta, etiqueta = None, None
    if fuente == "Caso del dataset":
        casos = estado.casos_dataset()
        opciones = ["—"] + [f"{c['id']} · {c['split']}" for c in casos]
        eleccion = st.selectbox("Caso", opciones)
        if eleccion != "—":
            caso = casos[opciones.index(eleccion) - 1]
            ruta, etiqueta = caso["ruta"], eleccion
        st.caption("val/test: casos no usados en el entrenamiento")
    else:
        archivo = st.file_uploader("Volumen CT (.mha, .nii, .nii.gz)", type=["mha", "nii", "gz"])
        ruta_archivo = estado.ruta_upload(archivo)
        st.caption("Archivo pesado o red lenta (más de ~150 MB por el túnel): súbelo por partes.")
        por_partes = estado.subida_por_partes()
        if ruta_archivo:
            ruta, etiqueta = ruta_archivo, archivo.name
        elif por_partes:
            ruta, etiqueta = por_partes

    vista = st.radio("Visualizador", list(VISTAS))

if ruta is None:
    st.stop()

volumen, spacing = estado.cargar_volumen(ruta)
VISTAS[vista].render(volumen, spacing, ruta, etiqueta)
