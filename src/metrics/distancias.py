"""Distancia mínima entre cada fragmento y el fragmento principal de su mismo hueso, en mm.

La distancia entre dos fragmentos se define como la mínima distancia entre sus vóxeles, medida
con una transformada de distancia euclidiana (EDT, scipy.ndimage.distance_transform_edt) sobre
las máscaras y expresada en milímetros con el espaciado físico del header de la imagen. El
fragmento principal de una región anatómica es el de mayor número de vóxeles.

Convención: la EDT mide entre centros de vóxel. Fragmentos adyacentes que comparten cara quedan
a un paso de rejilla; distancia 0 solo existiría si compartieran vóxeles, lo que no ocurre entre códigos distintos.

Modos:
- "3d" (por defecto): EDT volumétrica con sampling=(sz, sy, sx); el par más cercano puede cruzar
  cortes. Antes de la EDT se recorta la bounding box de la región: el principal queda dentro de
  la caja, así que el recorte no altera ningún mínimo y ahorra tiempo y memoria.
- "2d": EDT por corte axial con sampling=(sy, sx) y mínimo global entre cortes que contienen al
  principal. Aproximación: solo puede sobreestimar, y queda indefinida si ningún corte contiene
  ambos bloques.

Los mismos cálculos corren sobre máscaras de referencia y sobre máscaras predichas cambiando la entrada.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src import config


def load_mask(path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Lee una máscara (.mha/) -> (volumen (Z, Y, X), espaciado (sx, sy, sz) en mm).

    La distancia se calcula sobre la rejilla original del header
    """
    import SimpleITK as sitk  # import diferido: los tests de geometría no necesitan I/O

    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), tuple(float(s) for s in img.GetSpacing())


def codes_inesperados(mask: np.ndarray) -> list[int]:
    """Etiquetas (>0) fuera de la taxonomía 1-30. Vacío = máscara conforme a la taxonomía."""
    validos = {int(c) for codes in config.REGION_CODES.values() for c in codes}
    return sorted(int(c) for c in np.unique(mask) if int(c) > 0 and int(c) not in validos)


def codes_de_region(mask: np.ndarray, region: str) -> list[int]:
    """Códigos presentes de una región, en orden ascendente."""
    presentes = np.unique(mask)
    return [int(c) for c in config.REGION_CODES[region] if c in presentes]


def codigo_principal(mask: np.ndarray, codes: list[int]) -> int:
    """Código del fragmento principal de la región: el de mayor número de vóxeles."""
    return max((int((mask == c).sum()), int(c)) for c in codes)[1]


def distancia_region(mask: np.ndarray, spacing_xyz, region: str, modo: str = "3d") -> dict[int, float]:
    """Distancia en mm de cada fragmento conminuto de `region` al principal de la misma región.

    `mask` debe ser (Z, Y, X) y `spacing_xyz` = (sx, sy, sz) del header. Una región sin fractura (un solo
    fragmento) devuelve {}.
    """
    if modo not in ("3d", "2d"):
        raise ValueError(f"modo debe ser '3d' o '2d', no {modo!r}")
    codes = codes_de_region(mask, region)
    if len(codes) < 2:
        return {}
    principal = codigo_principal(mask, codes)
    conminutos = [c for c in codes if c != principal]
    if modo == "3d":
        # Recorte exacto: el vóxel del principal más cercano a cualquier conminuto está dentro de
        # la bounding box de la región (el principal completo queda dentro), así que la EDT del
        # recorte da los mismos mínimos que la del volumen completo.
        idx = np.argwhere(np.isin(mask, codes))
        (z0, y0, x0), (z1, y1, x1) = idx.min(axis=0), idx.max(axis=0) + 1
        sub = mask[z0:z1, y0:y1, x0:x1]
        edt = ndimage.distance_transform_edt(
            sub != principal, sampling=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]))
        etiquetas = np.where(np.isin(sub, conminutos), sub, 0).astype(np.uint8)
        minimos = np.atleast_1d(ndimage.minimum(edt, labels=etiquetas, index=conminutos))
        return {int(c): float(m) for c, m in zip(conminutos, minimos)}
    mejor = {c: np.inf for c in conminutos}
    for corte in mask:
        presentes = [c for c in conminutos if (corte == c).any()]
        if not presentes or not (corte == principal).any():
            continue
        edt = ndimage.distance_transform_edt(corte != principal, sampling=(spacing_xyz[1], spacing_xyz[0]))
        etiquetas = np.where(np.isin(corte, presentes), corte, 0).astype(np.uint8)
        minimos = np.atleast_1d(ndimage.minimum(edt, labels=etiquetas, index=presentes))
        for c, m in zip(presentes, minimos):
            mejor[c] = min(mejor[c], float(m))
    return mejor


def filas_caso(mask: np.ndarray, spacing_xyz, modo: str = "3d") -> list[dict]:
    """Una fila por fragmento presente (el principal con distancia 0.0) para un volumen.

    La misma tabla se produce sobre la GT y sobre predicciones; cruzarla por
    (caso, región, fragmento) da el error de distancia entre ambas.
    """
    filas = []
    for region in config.REGION_CODES:
        presentes = codes_de_region(mask, region)
        if not presentes:
            continue
        principal = codigo_principal(mask, presentes)
        distancias = distancia_region(mask, spacing_xyz, region, modo=modo)
        for c in presentes:
            filas.append({
                "region": region,
                "fragmento": int(c),
                "principal": bool(c == principal),
                "n_voxels": int((mask == c).sum()),
                "distancia_mm": 0.0 if c == principal else distancias[int(c)],
            })
    return filas
