"""Preprocesamiento de volúmenes CT: carga, reorientación, recorte, ventaneo HU y normalización.

1. `load_volume`: lee un volumen 3D (.mha, .nii, .nii.gz u otro formato de SimpleITK).
2. `reorient`: lleva imagen (y máscara) a `config.TARGET_ORIENTATION`, de modo que todos los
   volúmenes compartan la misma orientación de array.
3. `crop_center`: recorta el plano axial a `config.CROP_MM` mm centrado en el volumen, rellenando
   si el volumen es más pequeño. El espaciado no cambia.
4. `window_hu`: recorta los HU a `config.HU_WINDOW` y reescala linealmente a [0, 1]. Es también la
   normalización min-max.

Uso como comando, para preprocesar volúmenes sin máscara (p. ej. imágenes de prueba):

    uv run python -m src.dataset.preprocessing RUTA [RUTA ...] --out results/preprocessed

Cada RUTA es un archivo de volumen o una carpeta con volúmenes. Por cada volumen escribe un
`<nombre>.npz` con la imagen preprocesada y su geometría.
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


def crop_center(img: sitk.Image, size_mm=config.CROP_MM, fill=config.PAD_HU) -> sitk.Image:
    """Recorta x e y a `size_mm` (mm) centrado en el volumen; rellena con `fill` si no alcanza.

    El tamaño en píxeles es `round(size_mm / espaciado)` en cada eje. z no se recorta. La imagen
    devuelta conserva espaciado, dirección y origen físico consistentes con el recorte.
    """
    size = np.array(img.GetSize())
    spacing = np.array(img.GetSpacing())
    nuevo = size.copy()
    nuevo[:2] = np.round(np.array(size_mm) / spacing[:2]).astype(int)
    inicio = (size - nuevo) // 2                      # negativo si hay que rellenar
    pad_lo, pad_hi = np.maximum(-inicio, 0), np.maximum(inicio + nuevo - size, 0)
    if pad_lo.any() or pad_hi.any():
        img = sitk.ConstantPad(img, pad_lo.tolist(), pad_hi.tolist(), fill)
    return sitk.RegionOfInterest(img, nuevo.tolist(), (inicio + pad_lo).tolist())


def window_hu(hu: np.ndarray, window=config.HU_WINDOW) -> np.ndarray:
    """Recorta a [low, high] HU y reescala a [0, 1] (float32)."""
    low, high = window
    if not high > low:
        raise ValueError(f"Ventana HU inválida: {window}")
    return ((np.clip(hu.astype(np.float32), low, high) - low) / (high - low)).astype(np.float32)


def preprocess_volume(img: sitk.Image, mask: sitk.Image | None = None) -> dict:
    """Aplica reorientación, recorte y ventaneo a un volumen (y a su máscara, si la hay).

    Devuelve un diccionario con:
      image        float32 (Z, Y, X) en [0, 1]
      hu           el mismo recorte antes del ventaneo, en HU (Z, Y, X)
      mask         int16 (Z, Y, X) o None
      spacing      (x, y, z) en mm
      origin, direction    geometría física del volumen preprocesado
    """
    if mask is not None and (img.GetSize() != mask.GetSize() or not np.allclose(img.GetSpacing(), mask.GetSpacing())):
        raise ValueError("Imagen y máscara difieren en tamaño o espaciado")
    img = crop_center(reorient(img), fill=config.PAD_HU)
    hu = sitk.GetArrayFromImage(img)
    out = {"image": window_hu(hu), "hu": hu, "mask": None, "spacing": img.GetSpacing(),
           "origin": img.GetOrigin(), "direction": img.GetDirection()}
    if mask is not None:
        out["mask"] = sitk.GetArrayFromImage(crop_center(reorient(mask), fill=0)).astype(np.int16)
    return out


def preprocess_file(path, mask_path=None) -> dict:
    """`preprocess_volume` sobre archivos en disco."""
    return preprocess_volume(load_volume(path), load_volume(mask_path) if mask_path else None)


def _volumenes(rutas) -> list[Path]:
    """Expande archivos y carpetas a la lista de volúmenes con extensión soportada."""
    salida = []
    for r in map(Path, rutas):
        salida += sorted(p for p in r.iterdir() if p.name.endswith(EXTENSIONES)) if r.is_dir() else [r]
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa volúmenes CT (reorientar, recortar, ventanear).")
    parser.add_argument("rutas", nargs="+", help="archivos de volumen o carpetas con volúmenes")
    parser.add_argument("--out", default="results/preprocessed", help="carpeta de salida (por defecto %(default)s)")
    args = parser.parse_args()

    salida = Path(args.out)
    salida.mkdir(parents=True, exist_ok=True)
    for ruta in _volumenes(args.rutas):
        res = preprocess_file(ruta)
        nombre = ruta.name
        for ext in EXTENSIONES:
            nombre = nombre.removesuffix(ext)
        np.savez_compressed(
            salida / f"{nombre}.npz", image=res["image"], spacing_xyz=res["spacing"], origin_xyz=res["origin"],
            direction=res["direction"], hu_window=config.HU_WINDOW, crop_mm=config.CROP_MM,
            orientation=config.TARGET_ORIENTATION, source=str(ruta))
        print(f"{ruta} -> {salida / (nombre + '.npz')}  {res['image'].shape} (Z, Y, X), espaciado {tuple(round(s, 4) for s in res['spacing'])} mm")


if __name__ == "__main__":
    main()
