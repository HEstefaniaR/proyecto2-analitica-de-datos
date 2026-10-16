"""Estado compartido del dashboard: cargas cacheadas, modelo, inferencia y casos del dataset.

Las funciones con caché de Streamlit viven aquí para que las tres vistas compartan el mismo
volumen y el mismo resultado de inferencia por caso: la inferencia corre una vez por caso y las
dos vistas del modelo (corte a corte y reconstrucción 3D) leen el mismo resultado.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


# --------------------------------------------------------------------------- casos y volumen
@st.cache_data(show_spinner=False)
def casos_dataset() -> list[dict]:
    """Casos .mha del conjunto de test (splits.csv), ordenados por id: los únicos que el modelo no vio al entrenar
    ni se usaron para elegirlo."""
    import pandas as pd

    rutas = {p.stem: p for d in config.IMAGE_DIRS for p in sorted(d.glob("*.mha"))}
    tabla = pd.read_csv(config.SPLITS_DIR / "splits.csv", dtype={"case_id": str})
    splits = dict(zip(tabla.case_id, tabla.split))
    return [{"id": cid, "split": splits[cid], "ruta": str(ruta)} for cid, ruta in sorted(rutas.items())
            if splits.get(cid) == "test"]


_SUBIDA_DIR = Path(__file__).resolve().parent / "subida_partes"
_EXT_CORTE = (".mha", ".nii", ".nii.gz")


def _nombre_seguro(relativo: str) -> str:
    """Nombre plano para guardar un corte de la carpeta subida (sin separadores de ruta)."""
    return re.sub(r"[^\w.\-]", "_", relativo.replace("\\", "/").replace("/", "__")).lstrip(".") or "corte"


def subida_por_partes(modo: str = "volumen", trozo_mb: float = 4.0) -> dict | None:
    """Subida de un CT por partes por el websocket de Streamlit (`dashboard/subida_partes/index.html`).

    `modo`: "volumen" (un archivo .mha / .nii / .nii.gz) o "serie" (una carpeta con un archivo por corte).
    El cargador normal de Streamlit manda el archivo en una sola petición, que detrás del túnel de Cloudflare se corta
    por tiempo con archivos grandes o redes lentas. Aquí el navegador manda un mensaje, Python lo guarda en un temporal
    y lo confirma (`ack`), y solo entonces llega el siguiente. Devuelve {"ruta", "nombre"} cuando terminó: la ruta del
    archivo (volumen) o de la carpeta con los cortes (serie).
    """
    import streamlit.components.v1 as components

    comp = components.declare_component("subida_por_partes", path=str(_SUBIDA_DIR))
    s = st.session_state.setdefault(f"_subida_{modo}", {"id": None, "ruta": None, "recibidas": -1, "ack": None,
                                                        "nombre": None, "lista": False, "cortes": 0})
    valor = comp(modo=modo, ack=s["ack"], trozo_mb=trozo_mb, key=f"subida_{modo}", default=None)
    if isinstance(valor, (bytes, bytearray)) and len(valor) > 4:
        largo = int.from_bytes(valor[:4], "big")
        cab = json.loads(bytes(valor[4:4 + largo]).decode("utf-8"))
        datos = memoryview(valor)[4 + largo:]
        if cab["id"] != s["id"] and cab["i"] == 0:             # subida nueva: temporal nuevo
            if modo == "serie":
                ruta = tempfile.mkdtemp(prefix="cortes_")
            else:
                sufijo = "".join(Path(cab["nombre"]).suffixes[-2:]) or ".mha"
                with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
                    ruta = tmp.name
            s.update(id=cab["id"], ruta=ruta, recibidas=-1, nombre=cab["nombre"], lista=False, cortes=0)
        if cab["id"] == s["id"] and cab["i"] == s["recibidas"] + 1:
            if modo == "serie":                                # cada mensaje trae uno o más cortes completos
                pos = 0
                for a in cab["archivos"]:
                    (Path(s["ruta"]) / _nombre_seguro(a["nombre"])).write_bytes(datos[pos:pos + a["tam"]])
                    pos += a["tam"]
                s["cortes"] += len(cab["archivos"])
                completo = s["cortes"] == cab["total"]
            else:
                with open(s["ruta"], "ab") as f:
                    f.write(datos)
                completo = Path(s["ruta"]).stat().st_size == cab["tam"]
            s["recibidas"] = cab["i"]
            s["lista"] = cab["i"] == cab["n"] - 1 and completo
            s["ack"] = f"{cab['id']}:{cab['i']}"
            st.rerun()                                         # el componente recibe el `ack` y manda lo siguiente
    return {"ruta": s["ruta"], "nombre": s["nombre"]} if s["lista"] and Path(s["ruta"]).exists() else None


def _orden_natural(p: Path) -> list:
    """corte_2 antes que corte_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


@st.cache_resource(show_spinner="Uniendo los cortes…", max_entries=3)
def _apilar_cortes(carpeta: str):
    """Apila los cortes de la carpeta en un volumen 3D. Devuelve (imagen, distancia z de los archivos o None).

    Si los cortes son volúmenes de un solo corte (traen su posición), se ordenan por esa posición y la distancia
    entre cortes sale de ella. Si son imágenes 2D (no la traen), se ordenan por nombre en orden natural y la
    distancia queda en None: la pide el dashboard."""
    import SimpleITK as sitk

    archivos = sorted((p for p in Path(carpeta).iterdir() if p.name.lower().endswith(_EXT_CORTE)), key=_orden_natural)
    if not archivos:
        raise ValueError("La carpeta no tiene cortes .mha, .nii o .nii.gz")
    cortes, posiciones, origenes, ref = [], [], [], None
    for p in archivos:
        img = sitk.ReadImage(str(p))
        if img.GetDimension() == 3:
            if img.GetSize()[2] != 1:
                raise ValueError(f"{p.name} tiene {img.GetSize()[2]} cortes: se esperaba un corte por archivo "
                                 "(un volumen completo se sube como «Volumen»)")
            normal = np.array(img.GetDirection()).reshape(3, 3)[:, 2]
            posiciones.append(float(np.dot(img.GetOrigin(), normal)))
            origenes.append(img.GetOrigin())
            if ref is None:
                ref = img
            img = img[:, :, 0]
        elif img.GetDimension() != 2:
            raise ValueError(f"{p.name}: se esperaba una imagen 2D (un corte)")
        cortes.append(img)
    if len({c.GetSize() for c in cortes}) > 1:
        raise ValueError("Los cortes no tienen todos el mismo tamaño")
    dz = None
    if len(posiciones) == len(cortes) and len(cortes) > 1:     # todos traen su posición: orden por posición
        orden = np.argsort(posiciones)
        cortes = [cortes[k] for k in orden]
        origenes = [origenes[k] for k in orden]
        paso = float(np.median(np.diff(np.sort(posiciones))))
        dz = paso if paso > 1e-3 else None
    vol = sitk.JoinSeries(cortes)
    if ref is not None:
        vol.SetDirection(ref.GetDirection())
        vol.SetOrigin(origenes[0])
    return vol, dz


@st.cache_data(show_spinner=False, max_entries=6)
def volumen_desde_cortes(carpeta: str, dz: float | None = None) -> dict:
    """Volumen .mha armado con los cortes de la carpeta: {"ruta", "cortes", "dz", "dz_archivo"}.

    `dz` (mm) es la distancia entre cortes; si es None se usa la de los archivos o, si no la traen, 1,0 mm.
    El .mha queda en una carpeta hermana de la de los cortes, un archivo por distancia."""
    import SimpleITK as sitk

    vol, dz_archivo = _apilar_cortes(carpeta)
    dz = float(dz or dz_archivo or 1.0)
    vol = sitk.Image(vol)                                      # copia: no tocar la imagen cacheada
    sx, sy, _ = vol.GetSpacing()
    vol.SetSpacing((sx, sy, dz))
    destino = Path(carpeta + "_volumen")
    destino.mkdir(exist_ok=True)
    ruta = destino / f"volumen_dz{dz:.3f}.mha"
    if not ruta.exists():
        sitk.WriteImage(vol, str(ruta), True)
    return {"ruta": str(ruta), "cortes": vol.GetSize()[2], "dz": dz, "dz_archivo": dz_archivo}


@st.cache_data(show_spinner="Cargando volumen…", max_entries=3)
def cargar_volumen(ruta: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Carga y reorienta el volumen (mismo paso de orientación del pipeline). Sin crop ni
    ventaneo: ese preprocesamiento es específico del modelo, no del visualizador crudo."""
    import SimpleITK as sitk

    from src.dataset import preprocessing as pre

    img = pre.reorient(pre.load_volume(ruta))
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()


# --------------------------------------------------------------------------- modelo e inferencia
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo(run: str = config.DASH_MODEL_RUN):
    """Checkpoint de producción en memoria: se carga una vez por sesión."""
    from src import inference

    return inference.load_model(run)


@st.cache_resource
def _candado_inferencia() -> threading.Lock:
    """Un solo pipeline a la vez en todo el servidor: si dos personas piden inferencia al mismo tiempo (por ejemplo,
    por el túnel), la segunda espera a que termine la primera en vez de correr las dos sobre la misma GPU (el doble de
    memoria y resultados dañados por la competencia)."""
    return threading.Lock()


CACHE_DISCO = config.RESULTS_DIR / "dashboard_cache"       # predicciones guardadas (se puede borrar sin riesgo)
_MATRICES = ("instancias", "fragmentos_id")


def _clave_disco(ruta: str, run: str) -> str:
    """Identifica una predicción: contenido del CT (tamaño + primeros 4 MB), checkpoint y salida mostrada.

    No usa la ruta del archivo (un CT subido cae cada vez en un temporal distinto) y cambia si cambia el
    `best.pth`, la salida (`DASH_OUTPUT`) o el modo de fusión, así que nunca devuelve un resultado viejo."""
    p, ck = Path(ruta), config.DETECTION_DIR / run / "best.pth"
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(4 << 20))
    partes = (p.stat().st_size, run, ck.stat().st_size, ck.stat().st_mtime_ns, config.DASH_OUTPUT, config.M3D_FUSION_MODE)
    h.update("|".join(str(v) for v in partes).encode())
    return h.hexdigest()[:20]


def _leer_disco(clave: str) -> dict | None:
    npz, meta = CACHE_DISCO / f"{clave}.npz", CACHE_DISCO / f"{clave}.json"
    if not (npz.exists() and meta.exists()):
        return None
    res = json.loads(meta.read_text())
    with np.load(npz) as z:
        res.update({k: z[k] for k in _MATRICES})
    res["spacing"] = tuple(res["spacing"])
    res["desde_disco"] = True
    return res


def _guardar_disco(clave: str, res: dict) -> None:
    CACHE_DISCO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DISCO / f"{clave}.npz", **{k: res[k] for k in _MATRICES})
    (CACHE_DISCO / f"{clave}.json").write_text(json.dumps({k: v for k, v in res.items() if k not in _MATRICES}))


@st.cache_data(show_spinner="Ejecutando pipeline: detección, segmentación, clasificación y distancias...",
               max_entries=3)
def resultado_inferencia(ruta: str, run: str = config.DASH_MODEL_RUN) -> dict:
    """Predicción completa del caso (ver `src.inference.predict_case`).

    Dos niveles de caché: en memoria (Streamlit, por sesión del servidor: las dos vistas del modelo comparten
    el resultado) y en disco (`CACHE_DISCO`: sobrevive a reiniciar el dashboard; un caso ya predicho se abre sin
    volver a correr el modelo)."""
    from src import inference
    from src.dataset import preprocessing as pre

    clave = _clave_disco(ruta, run)
    res = _leer_disco(clave)
    if res is None:
        with _candado_inferencia():
            res = _leer_disco(clave)                               # pudo quedar listo mientras esperaba el candado
            if res is None:
                res = inference.predict_case(pre.reorient(pre.load_volume(ruta)), cargar_modelo(run))
                _guardar_disco(clave, res)
    # Cajas indexadas por corte: el visor por corte no recorre todas en cada movimiento del slider
    por_corte: dict[int, list] = {}
    for caja in res["cajas"]:
        por_corte.setdefault(int(caja["z0"]), []).append(caja)
    res["cajas_por_corte"] = por_corte
    return res


def inferencia_en_disco(ruta: str, run: str = config.DASH_MODEL_RUN) -> bool:
    """¿Ya hay una predicción guardada de este caso con el modelo y la salida actuales?"""
    clave = _clave_disco(ruta, run)
    return (CACHE_DISCO / f"{clave}.npz").exists() and (CACHE_DISCO / f"{clave}.json").exists()


def pedir_inferencia(ruta: str) -> dict | None:
    """Botón que dispara la inferencia del caso; `None` mientras el usuario no la pida.

    El resultado queda cacheado por caso, así que las vistas del modelo lo comparten: la primera
    vez corre el pipeline completo y después es lectura de caché. El estado del caso actual viaja
    en session_state, de modo que cambiar de caso vuelve a pedir el botón.
    """
    if st.session_state.get("infer_clave") == ruta or inferencia_en_disco(ruta):
        st.session_state.infer_clave = ruta          # ya predicho (en esta sesión o guardado en disco): sin botón
        return resultado_inferencia(ruta)
    if st.button("Ejecutar inferencia", type="primary"):
        res = resultado_inferencia(ruta)
        st.session_state.infer_clave = ruta
        return res
    return None


# --------------------------------------------------------------------------- color
def tono(hex_color: str, fraccion: float) -> str:
    """Mezcla un color con blanco en la fracción dada (0 = color puro, 1 = blanco).

    Distingue fragmentos del mismo hueso: mismo matiz, luminosidad creciente por componente.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * fraccion) for c in (r, g, b))


def roles_fragmentos(fragmentos: list[dict]) -> dict[int, dict]:
    """Color y etiqueta de cada fragmento (fid → {"color", "etiqueta"}).

    El principal lleva el color base de su región y la etiqueta P; los conminutos, luminosidad
    creciente y etiquetas C1, C2, … numeradas dentro de su hueso. `fid` es la posición en
    `fragmentos` + 1, la misma codificación de `predict_case["fragmentos_id"]`."""
    salida: dict[int, dict] = {}
    conminutos_por_hueso: dict[int, int] = {}
    for i, fila in enumerate(fragmentos):
        if fila["principal"]:
            salida[i + 1] = {"color": config.REGION_COLORS[fila["region"]], "etiqueta": "P"}
        else:
            n = conminutos_por_hueso.get(fila["hueso"], 0) + 1
            conminutos_por_hueso[fila["hueso"]] = n
            salida[i + 1] = {"color": tono(config.REGION_COLORS[fila["region"]], 0.18 * n),
                             "etiqueta": f"C{n}"}
    return salida


def df_fragmentos(res: dict):
    """Tabla resumen de los fragmentos del caso: rol, volumen en mm³ y distancia en mm."""
    import pandas as pd

    mm3 = float(np.prod(res["spacing"]))
    roles = roles_fragmentos(res["fragmentos"])
    filas = [{
        "Fragmento": roles[i + 1]["etiqueta"],
        "Región": config.REGION_LABELS[fila["region"]],
        "Rol": "principal" if fila["principal"] else "conminuto",
        "Volumen (mm³)": round(fila["voxels"] * mm3),
        "Distancia (mm)": fila["distancia_mm"],
    } for i, fila in enumerate(res["fragmentos"])]
    return pd.DataFrame(filas)
