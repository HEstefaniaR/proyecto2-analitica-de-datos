"""Preprocesamiento de volúmenes CT: carga, reorientación, cubo isótropo, ventaneo HU y vistas.

1. `load_volume`: lee un volumen 3D (.mha, .nii, .nii.gz u otro formato de SimpleITK).
2. `reorient`: lleva imagen (y máscara) a `config.TARGET_ORIENTATION`, de modo que todos los
   volúmenes compartan la misma orientación de array.
3. `cube_geometry` + `to_cube`: remuestrea el volumen a un cubo de `config.CUBE_MM` mm con
   `config.INPUT_SIZE` vóxeles por lado (vóxel isótropo de `config.VOXEL_MM` mm), centrado en el
   centro del campo de visión en x, y y en los cortes con hueso en z. Lo que queda fuera del CT se
   rellena con aire (`config.PAD_HU`); la máscara, con 0. Así se corrige también la anisotropía:
   el vóxel queda igual en los tres ejes y las vistas coronal y sagital no se deforman.
4. `window_hu`: recorta los HU a `config.HU_WINDOW` y reescala linealmente a [0, 1].
5. `view_stack`: corta el cubo en una de las tres vistas (axial, coronal o sagital).
6. `cube_to_original`: lleva una máscara del cubo de vuelta a la geometría original del CT, para
   medir distancias en mm con la resolución original.

Uso como comando, para preprocesar volúmenes sin máscara (p. ej. imágenes de prueba):

    uv run python -m src.dataset.preprocessing RUTA [RUTA ...] --out results/preprocessed

Cada RUTA es un archivo de volumen o una carpeta con volúmenes. Por cada volumen escribe un
`<nombre>.npz` con el cubo preprocesado y su geometría.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src import config

EXTENSIONES = (".mha", ".nii", ".nii.gz")


def load_volume(path) -> sitk.Image:
    """Lee un volumen 3D. Lanza ValueError si el archivo no es 3D (p. ej. una imagen 2D)."""
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"{path}: se esperaba un volumen 3D y tiene {img.GetDimension()} dimensiones")
    return img


def reorient(img: sitk.Image, orientation: str = config.TARGET_ORIENTATION) -> sitk.Image:
    """Reorienta el array (permuta y voltea ejes) sin interpolar, así que sirve también para máscaras."""
    return sitk.DICOMOrient(img, orientation)


def bone_slices(hu: np.ndarray, bone_hu: float = config.BONE_HU,
                min_fraction: float = config.MIN_BONE_FRACTION) -> np.ndarray:
    """Cortes axiales (Z, Y, X) con hueso según los HU crudos (no usa la máscara): al menos
    `min_fraction` de sus píxeles con HU >= `bone_hu`."""
    return (hu >= bone_hu).reshape(hu.shape[0], -1).mean(axis=1) >= min_fraction


def cube_geometry(img: sitk.Image, hu: np.ndarray | None = None) -> dict:
    """Geometría del cubo para un volumen ya reorientado: origen, espaciado, dirección y tamaño.

    Centro: el del campo de visión en x e y, y el punto medio entre el primer y el último corte
    axial con hueso en z (si no hay hueso, el centro del volumen). Solo usa los HU, así que sirve
    igual en inferencia.
    """
    hu = sitk.GetArrayViewFromImage(img) if hu is None else hu
    nx, ny, nz = img.GetSize()
    z_bone = np.flatnonzero(bone_slices(hu))
    cz = (z_bone[0] + z_bone[-1]) / 2 if z_bone.size else (nz - 1) / 2
    center = np.array(img.TransformContinuousIndexToPhysicalPoint(((nx - 1) / 2, (ny - 1) / 2, float(cz))))
    direction = np.array(img.GetDirection()).reshape(3, 3)
    half = (config.INPUT_SIZE / 2 - 0.5) * config.VOXEL_MM
    origin = center - direction @ np.full(3, half)
    return {"origin": tuple(float(v) for v in origin), "spacing": (config.VOXEL_MM,) * 3,
            "direction": tuple(float(v) for v in direction.ravel()), "size": (config.INPUT_SIZE,) * 3}


def to_cube(img: sitk.Image, geom: dict, is_mask: bool = False) -> sitk.Image:
    """Remuestrea `img` a la rejilla del cubo `geom`.

    Imagen: suavizado gaussiano previo en los ejes donde el vóxel original es más fino que el del
    cubo (evita aliasing al reducir) e interpolación lineal; fuera del CT, `config.PAD_HU`.
    Máscara: vecino más cercano (no mezcla códigos); fuera del CT, 0.
    """
    if is_mask:
        interp, fill = sitk.sitkNearestNeighbor, 0
    else:
        img = sitk.Cast(img, sitk.sitkFloat32)
        sigma = [max(0.0, (config.VOXEL_MM - s) / 2) for s in img.GetSpacing()]
        if any(s > 0 for s in sigma):
            img = sitk.DiscreteGaussian(img, variance=[s * s for s in sigma], useImageSpacing=True)
        interp, fill = sitk.sitkLinear, float(config.PAD_HU)
    return sitk.Resample(img, list(geom["size"]), sitk.Transform(), interp, geom["origin"], geom["spacing"],
                         geom["direction"], fill, img.GetPixelID())


def window_hu(hu: np.ndarray, window=config.HU_WINDOW) -> np.ndarray:
    """Recorta a [low, high] HU y reescala a [0, 1] (float32)."""
    low, high = window
    if not high > low:
        raise ValueError(f"Ventana HU inválida: {window}")
    return ((np.clip(hu.astype(np.float32), low, high) - low) / (high - low)).astype(np.float32)


def fraction_outside_cube(mask: sitk.Image, geom: dict) -> tuple[float, dict]:
    """Fracción de los vóxeles etiquetados (> 0) de `mask` (reorientada) que caen fuera del cubo, en
    total y por código de fragmento. Se calcula con coordenadas físicas, sin remuestrear."""
    m = sitk.GetArrayFromImage(mask)
    idx = np.argwhere(m > 0)                                          # (N, 3) en orden z, y, x
    if idx.size == 0:
        return 0.0, {}
    codes = m[tuple(idx.T)]
    sp, org = np.array(mask.GetSpacing()), np.array(mask.GetOrigin())
    d_in = np.array(mask.GetDirection()).reshape(3, 3)
    phys = org + (idx[:, ::-1] * sp) @ d_in.T                         # (N, 3) en x, y, z (mm)
    d_cube = np.array(geom["direction"]).reshape(3, 3)
    pos = (phys - np.array(geom["origin"])) @ d_cube / config.VOXEL_MM   # índice continuo en el cubo
    fuera = ((pos < -0.5) | (pos > config.INPUT_SIZE - 0.5)).any(axis=1)
    por_codigo = {int(c): float(fuera[codes == c].mean()) for c in np.unique(codes)}
    return float(fuera.mean()), por_codigo


def preprocess_volume(img: sitk.Image, mask: sitk.Image | None = None) -> dict:
    """Reorientación, cubo isótropo y ventaneo de un volumen (y de su máscara, si la hay).

    Devuelve un diccionario con:
      image        float32 (Z, Y, X) = (256, 256, 256) en [0, 1]
      hu           el mismo cubo antes del ventaneo, en HU (float32)
      mask         uint8 (Z, Y, X) con los códigos de fragmento, o None
      cube         geometría del cubo (origen, espaciado, dirección, tamaño)
      original     la imagen reorientada (sitk.Image), referencia para volver al espacio original
      spacing      espaciado del cubo (x, y, z) en mm
    """
    if mask is not None and (img.GetSize() != mask.GetSize() or not np.allclose(img.GetSpacing(), mask.GetSpacing())):
        raise ValueError("Imagen y máscara difieren en tamaño o espaciado")
    img = reorient(img)
    geom = cube_geometry(img)
    hu = sitk.GetArrayFromImage(to_cube(img, geom))
    out = {"image": window_hu(hu), "hu": hu, "mask": None, "cube": geom, "original": img,
           "spacing": geom["spacing"]}
    if mask is not None:
        out["mask"] = sitk.GetArrayFromImage(to_cube(reorient(mask), geom, is_mask=True)).astype(np.uint8)
    return out


def preprocess_file(path, mask_path=None) -> dict:
    """`preprocess_volume` sobre archivos en disco."""
    return preprocess_volume(load_volume(path), load_volume(mask_path) if mask_path else None)


def cube_to_original(mask_cube: np.ndarray, geom: dict, reference: sitk.Image) -> np.ndarray:
    """Lleva una máscara del cubo (Z, Y, X) a la rejilla de `reference` (el CT reorientado), con
    vecino más cercano. Sirve para medir en mm con la resolución original."""
    m = sitk.GetImageFromArray(mask_cube.astype(np.uint8))
    m.SetOrigin(geom["origin"]); m.SetSpacing(geom["spacing"]); m.SetDirection(geom["direction"])
    out = sitk.Resample(m, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


# --- Vistas -------------------------------------------------------------------------------
# El cubo es (Z, Y, X) en LPS: z crece hacia la cabeza, y hacia atrás y x hacia la izquierda.
#   axial    corte k = cubo[k]          filas = y (adelante arriba), columnas = x
#   coronal  corte k = cubo[:, k, :]    filas = z con la cabeza arriba, columnas = x
#   sagital  corte k = cubo[:, :, k]    filas = z con la cabeza arriba, columnas = y (adelante a la izquierda)

def view_slices(cube: np.ndarray, view: str, idx) -> np.ndarray:
    """Cortes `idx` (int o lista) del cubo en la vista `view` -> (H, W) o (len(idx), H, W)."""
    if view == "axial":
        out = cube[idx]
    elif view == "coronal":
        out = np.moveaxis(cube[:, idx, :], 1, 0) if not np.isscalar(idx) else cube[:, idx, :]
        out = out[..., ::-1, :]
    elif view == "sagittal":
        out = np.moveaxis(cube[:, :, idx], 2, 0) if not np.isscalar(idx) else cube[:, :, idx]
        out = out[..., ::-1, :]
    else:
        raise ValueError(f"Vista desconocida: {view}")
    return np.ascontiguousarray(out)


def view_stack(cube: np.ndarray, view: str) -> np.ndarray:
    """Todos los cortes del cubo en la vista `view` -> (N, H, W)."""
    return view_slices(cube, view, list(range(cube.shape[{"axial": 0, "coronal": 1, "sagittal": 2}[view]])))


def _volumenes(rutas) -> list[Path]:
    """Expande archivos y carpetas a la lista de volúmenes con extensión soportada."""
    salida = []
    for r in map(Path, rutas):
        salida += sorted(p for p in r.iterdir() if p.name.endswith(EXTENSIONES)) if r.is_dir() else [r]
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa volúmenes CT (reorientar, cubo isótropo, ventanear).")
    parser.add_argument("rutas", nargs="+", help="archivos de volumen o carpetas con volúmenes")
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "preprocessed"), help="carpeta de salida")
    args = parser.parse_args()

    salida = Path(args.out)
    salida.mkdir(parents=True, exist_ok=True)
    for ruta in _volumenes(args.rutas):
        res = preprocess_file(ruta)
        nombre = ruta.name
        for ext in EXTENSIONES:
            nombre = nombre.removesuffix(ext)
        g = res["cube"]
        np.savez_compressed(
            salida / f"{nombre}.npz", image=res["image"], spacing_xyz=g["spacing"], origin_xyz=g["origin"],
            direction=g["direction"], hu_window=config.HU_WINDOW, cube_mm=config.CUBE_MM,
            orientation=config.TARGET_ORIENTATION, source=str(ruta))
        print(f"{ruta} -> {salida / (nombre + '.npz')}  {res['image'].shape} (Z, Y, X), vóxel {config.VOXEL_MM:.4f} mm")


if __name__ == "__main__":
    main()
