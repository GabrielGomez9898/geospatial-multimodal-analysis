"""
preprocessing.py

Implementa la seccion D (preprocesamiento) y E (unidad espacial) del protocolo,
para cada uno de los tres modelos. Todo lo que aqui se hace queda documentado
explicitamente, tal como exige el protocolo: "todas las transformaciones
aplicadas a cada modelo deberan documentarse".

Convencion de arrays: todas las imagenes se manejan como numpy arrays con forma
(bandas, alto, ancho), en reflectancia (float32), NO en uint8/DN crudo.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import zoom

from config import (
    ModelSpec,
    SENTINEL2_ALL_BANDS,
    SENTINEL2_20M_BANDS,
)


# ---------------------------------------------------------------------------
# Paso 0 (comun a los tres modelos, antes de cualquier logica especifica):
# alinear las bandas de 20 m a la grilla de 10 m, como exige la seccion D.
# ---------------------------------------------------------------------------

def resample_20m_bands_to_10m(image: np.ndarray, band_order: List[str] = SENTINEL2_ALL_BANDS) -> np.ndarray:
    """
    image: array (13, H, W) con las 13 bandas de Sentinel-2 en `band_order`,
           donde las bandas de 20 m vienen ya en su resolucion nativa (H/2, W/2)
           *interpoladas por el lector* a la forma comun del stack, o bien
           el stack completo ya viene a la resolucion de 10 m con las bandas
           de 20 m duplicadas por pixel (depende de tu lector OSCD).

    Si tu pipeline de lectura ya entrega todas las bandas alineadas a 10 m
    (lo mas comun al usar rasterio con resampling en la lectura), esta funcion
    es un no-op y solo valida las formas. Se deja explicita para que quede
    documentado como paso del protocolo, incluso si el remuestreo real ocurre
    en la etapa de lectura de disco.
    """
    n_bands, h, w = image.shape
    assert n_bands == len(band_order), "El stack no coincide con band_order"
    # Validacion: todas las bandas deben tener exactamente la misma forma (h, w).
    # Si tu lector no garantiza esto, aqui es donde debes remuestrear banda por
    # banda con zoom(banda, factor=2, order=1) para las bandas en SENTINEL2_20M_BANDS.
    return image


# ---------------------------------------------------------------------------
# Seleccion y reordenamiento de bandas por modelo
# ---------------------------------------------------------------------------

def select_and_reorder_bands(
    image: np.ndarray,
    source_band_order: List[str],
    target_bands: List[str],
) -> np.ndarray:
    """Selecciona y reordena bandas de `image` (C, H, W) segun `target_bands`."""
    idx = [source_band_order.index(b) for b in target_bands]
    return image[idx, :, :]


# ---------------------------------------------------------------------------
# Remuestreo de GSD (solo Prithvi lo necesita: 10 m -> 30 m)
# ---------------------------------------------------------------------------

def average_pool_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Remuestreo por agregacion de area (promedio de bloques factor x factor),
    banda por banda. Ver discusion: esto es una aproximacion al producto HLS
    real (que involucra correccion atmosferica y armonizacion Landsat-Sentinel),
    no una replica exacta -- documentar como supuesto simplificador.
    """
    c, h, w = image.shape
    h_crop = (h // factor) * factor
    w_crop = (w // factor) * factor
    image = image[:, :h_crop, :w_crop]
    image = image.reshape(c, h_crop // factor, factor, w_crop // factor, factor)
    return image.mean(axis=(2, 4)).astype(np.float32)


def center_crop_to_multiple(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Recorta al centro para que H y W sean multiplos de patch_size (evita padding)."""
    c, h, w = image.shape
    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size
    top = (h - h_new) // 2
    left = (w - w_new) // 2
    return image[:, top:top + h_new, left:left + w_new]


# ---------------------------------------------------------------------------
# Recorte en chips
# ---------------------------------------------------------------------------

@dataclass
class Chip:
    array: np.ndarray       # (C, chip_size, chip_size)
    row_off: int             # offset en pixeles, en la imagen ya preprocesada para este modelo
    col_off: int


def extract_dense_chips(image: np.ndarray, chip_size: int, stride: int | None = None) -> List[Chip]:
    """
    Grilla densa de chips no solapados (o solapados si stride < chip_size),
    usada para Clay (256) y SatMAE (96). Para Prithvi normalmente esto devolvera
    un unico chip (ver seccion E de la discusion: la escena remuestreada a 30 m
    apenas alcanza para un chip de 192 px).
    """
    stride = stride or chip_size
    c, h, w = image.shape
    chips = []
    for row in range(0, h - chip_size + 1, stride):
        for col in range(0, w - chip_size + 1, stride):
            chips.append(Chip(
                array=image[:, row:row + chip_size, col:col + chip_size],
                row_off=row,
                col_off=col,
            ))
    return chips


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------

def zscore_normalize(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Normalizacion por banda: (x - mean) / std. mean/std deben venir del modelo
    (metadata.yaml de Clay, stats de fMoW-Sentinel para SatMAE, o config.json de
    Prithvi -- NUNCA calculados desde tus propios datos de OSCD, ver discusion)."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    return (image.astype(np.float32) - mean_arr) / std_arr


# ---------------------------------------------------------------------------
# Pipeline especifico por modelo (junta los pasos anteriores)
# ---------------------------------------------------------------------------

def prepare_for_clay(image_10m: np.ndarray, spec: ModelSpec) -> List[Chip]:
    """image_10m: (13, H, W) en band_order = SENTINEL2_ALL_BANDS, ya a 10 m."""
    img = select_and_reorder_bands(image_10m, SENTINEL2_ALL_BANDS, spec.bands)
    # TODO: normalizar con las estadisticas oficiales de configs/metadata.yaml de Clay
    # (dependen del sensor: sentinel-2-l2a). No inventar mean/std aqui.
    return extract_dense_chips(img, spec.chip_size)


def prepare_for_satmae(image_10m: np.ndarray, spec: ModelSpec) -> List[Chip]:
    img = select_and_reorder_bands(image_10m, SENTINEL2_ALL_BANDS, spec.bands)
    # TODO: normalizar con las estadisticas de fMoW-Sentinel publicadas en el repo oficial.
    return extract_dense_chips(img, spec.chip_size)


def prepare_for_prithvi(image_10m: np.ndarray, spec: ModelSpec, mean: List[float], std: List[float]) -> List[Chip]:
    img = select_and_reorder_bands(image_10m, SENTINEL2_ALL_BANDS, spec.bands)
    img_30m = average_pool_downsample(img, factor=3)          # 10 m -> ~30 m
    img_30m = center_crop_to_multiple(img_30m, spec.patch_size)  # -> multiplo de 16 (192 tipicamente)
    img_30m = zscore_normalize(img_30m, mean, std)
    # Con una escena de ~600x600 a 10 m, esto normalmente da 1 solo chip de 192x192.
    return extract_dense_chips(img_30m, spec.chip_size)
