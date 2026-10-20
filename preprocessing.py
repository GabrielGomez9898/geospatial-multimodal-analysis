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
    SATMAE_MEAN_13,
    SATMAE_STD_13,
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


def extract_chips_at(image: np.ndarray, chip_size: int, offsets: List[Tuple[int, int]]) -> List[Chip]:
    """Extrae chips de chip_size x chip_size en las esquinas (row, col) dadas."""
    return [
        Chip(array=image[:, r:r + chip_size, c:c + chip_size], row_off=r, col_off=c)
        for r, c in offsets
    ]


# ---------------------------------------------------------------------------
# Ventanas comunes entre modelos (sin solape)
#
# Cada modelo ve un area de terreno distinta por chip:
#   Clay    256 px x 10 m = 2560 m
#   SatMAE   96 px x 10 m =  960 m
#   Prithvi 224 px x 30 m = 6720 m  (672 px en la grilla de 10 m)
# Para que todos evaluen la MISMA zona y el MISMO numero de chips por escena,
# la grilla de ventanas se arma con la huella mas grande (sin solape), y cada
# modelo recorta su chip centrado dentro de cada ventana. Si una escena no
# alcanza para una sola ventana, queda fuera para TODOS los modelos.
# ---------------------------------------------------------------------------

def footprint_px_10m(spec: ModelSpec) -> int:
    """Lado del chip de `spec` medido en pixeles de la grilla de 10 m."""
    return int(round(spec.chip_size * spec.native_gsd_m / 10.0))


def common_windows(h: int, w: int, align_specs: List[ModelSpec]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Grilla de ventanas no solapadas de lado = huella mas grande entre
    `align_specs`, centrada en la escena (el sobrante se reparte por igual
    entre los bordes, en vez de quedar todo abajo/a la derecha).
    Devuelve (lado_ventana_px_10m, [(row_off, col_off), ...]) en la grilla de 10 m.
    """
    window = max(footprint_px_10m(s) for s in align_specs)
    n_rows, n_cols = h // window, w // window
    top = (h - n_rows * window) // 2
    left = (w - n_cols * window) // 2
    offsets = [(top + i * window, left + j * window) for i in range(n_rows) for j in range(n_cols)]
    return window, offsets


def aligned_chip_offsets(h: int, w: int, spec: ModelSpec, align_specs: List[ModelSpec]) -> List[Tuple[int, int]]:
    """
    Esquinas (en la grilla de 10 m) de los chips de `spec`, uno por ventana
    comun, centrados en el centro de cada ventana.
    """
    window, windows = common_windows(h, w, align_specs)
    fp = footprint_px_10m(spec)
    if fp > window:
        raise ValueError(
            f"La huella de {spec.name} ({fp} px) es mayor que la ventana comun ({window} px); "
            f"incluye {spec.name} en align_specs."
        )
    margin = (window - fp) // 2
    return [(r + margin, c + margin) for r, c in windows]


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


def sentinel_normalize_satmae(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Replica EXACTA de la clase SentinelNormalize del repo oficial de SatMAE
    (util/datasets.py): NO es z-score. Estira cada banda al rango
    [mean - 2*std, mean + 2*std], lo mapea a [0, 255] con clip, lo castea a
    uint8 (para replicar la perdida de precision del pipeline original, que
    pasa por una imagen de 8 bits antes de volver a tensor) y lo vuelve a
    escalar a [0, 1] (equivalente al ToTensor() que sigue en el repo original).
    mean/std deben ser los de SATMAE_MEAN_13/SATMAE_STD_13 (config.py),
    ya recortados al subconjunto de bandas que usa SatMAE.
    """
    mean_arr = np.array(mean, dtype=np.float64).reshape(-1, 1, 1)
    std_arr = np.array(std, dtype=np.float64).reshape(-1, 1, 1)
    min_value = mean_arr - 2 * std_arr
    max_value = mean_arr + 2 * std_arr
    img = (image.astype(np.float64) - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return (img.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Pipeline especifico por modelo (junta los pasos anteriores)
# ---------------------------------------------------------------------------

def prepare_for_clay(image_10m: np.ndarray, spec: ModelSpec,
                     offsets: List[Tuple[int, int]]) -> List[Chip]:
    """image_10m: (13, H, W) en band_order = SENTINEL2_ALL_BANDS, ya a 10 m.
    offsets: esquinas de los chips en la grilla de 10 m (ver aligned_chip_offsets)."""
    img = select_and_reorder_bands(image_10m, SENTINEL2_ALL_BANDS, spec.bands)
    # TODO: normalizar con las estadisticas oficiales de configs/metadata.yaml de Clay
    # (dependen del sensor: sentinel-2-l2a). No inventar mean/std aqui.
    return extract_chips_at(img, spec.chip_size, offsets)


def prepare_for_satmae(image_10m: np.ndarray, spec: ModelSpec,
                       offsets: List[Tuple[int, int]]) -> List[Chip]:
    img = select_and_reorder_bands(image_10m, SENTINEL2_ALL_BANDS, spec.bands)
    # Estadisticas oficiales de fMoW-Sentinel, recortadas al subconjunto y
    # orden de spec.bands (SatMAE usa 10 de las 13 bandas de SENTINEL2_ALL_BANDS).
    band_idx = [SENTINEL2_ALL_BANDS.index(b) for b in spec.bands]
    mean_subset = [SATMAE_MEAN_13[i] for i in band_idx]
    std_subset = [SATMAE_STD_13[i] for i in band_idx]
    img = sentinel_normalize_satmae(img, mean_subset, std_subset)
    return extract_chips_at(img, spec.chip_size, offsets)


def prepare_prithvi_pairs(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    mask: np.ndarray,
    spec: ModelSpec,
    mean: List[float],
    std: List[float],
    offsets: List[Tuple[int, int]],
) -> list[dict]:
    """
    Prepara, para UNA escena, los pares de chips (t1, t2) de Prithvi junto con
    la mascara de cambio, todos alineados sobre el mismo terreno.

    offsets: esquinas de los chips en la grilla de 10 m (ver aligned_chip_offsets).
    Para cada una se corta una ventana de footprint_px_10m(spec) = 672 px a 10 m
    y se reduce a 30 m (224 px) por promedio de bloques 3x3. Cortar ANTES de
    remuestrear hace que la ventana de Prithvi coincida exactamente con la
    ventana comun en la grilla de 10 m, sin depender de que el offset sea
    multiplo de 3.

    La mascara se reduce con el mismo promedio de bloques 3x3 que la imagen y
    queda como FRACCION de cambio (0-1) por pixel de 30 m. Como todos los bloques
    tienen el mismo tamano, promediar fracciones equivale a promediar los pixeles
    originales de 10 m, asi que el umbral posterior (resample_mask_to_grid) da lo
    mismo que si se calculara sobre la mascara original.

    Devuelve una lista de dicts con: chip_t1, chip_t2 (Chip, con row_off/col_off
    en la grilla de 10 m) y mask_chip (array (chip_size, chip_size) con
    fracciones de cambio a 30 m).
    """
    factor = 3  # 10 m -> 30 m
    fp = footprint_px_10m(spec)
    sel1 = select_and_reorder_bands(img_t1, SENTINEL2_ALL_BANDS, spec.bands)
    sel2 = select_and_reorder_bands(img_t2, SENTINEL2_ALL_BANDS, spec.bands)

    pairs = []
    for r, c in offsets:
        win1 = sel1[:, r:r + fp, c:c + fp]
        win2 = sel2[:, r:r + fp, c:c + fp]
        win_mask = mask[None, r:r + fp, c:c + fp].astype(np.float32)
        pairs.append({
            "chip_t1": Chip(array=zscore_normalize(average_pool_downsample(win1, factor), mean, std),
                            row_off=r, col_off=c),
            "chip_t2": Chip(array=zscore_normalize(average_pool_downsample(win2, factor), mean, std),
                            row_off=r, col_off=c),
            "mask_chip": average_pool_downsample(win_mask, factor)[0],
        })
    return pairs