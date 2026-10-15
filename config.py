"""
config.py

Ficha tecnica de los tres modelos fundacionales (Clay v1.5, SatMAE, Prithvi EO 2.0),
consolidada a partir de la documentacion oficial de cada proyecto. Esta es la fuente
unica de verdad para el resto del pipeline: preprocessing.py, extractors.py y
run_pipeline.py leen de aqui en lugar de tener numeros repetidos (y potencialmente
inconsistentes) en cada archivo.

Referencias:
- Clay v1.5: https://clay-foundation.github.io/model/release-notes/specification.html
- SatMAE:    https://github.com/sustainlab-group/SatMAE (fMoW-Sentinel, ViT-Large)
- Prithvi EO 2.0 300M: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Todas las bandas de Sentinel-2 L2A en el orden en que normalmente vienen apiladas
# en OSCD (13 bandas). Los indices de esta lista son los que usan los diccionarios
# de seleccion de bandas de cada modelo mas abajo.
SENTINEL2_ALL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Bandas de 20 m que OSCD exige remuestrear a 10 m para alinear con la mascara
# de referencia (seccion D del protocolo).
SENTINEL2_20M_BANDS = ["B05", "B06", "B07", "B8A", "B11", "B12"]
SENTINEL2_60M_BANDS = ["B01", "B09", "B10"]


@dataclass
class ModelSpec:
    name: str
    chip_size: int              # tamano de entrada (px) que espera el modelo
    patch_size: int              # tamano de patch espacial (px)
    embed_dim: int
    native_gsd_m: float          # GSD con el que se preentreno el modelo
    bands: List[str]             # subconjunto y orden exacto de bandas Sentinel-2
    uses_metadata: bool          # si requiere lat/lon, gsd, fecha, etc.
    joint_temporal: bool         # si t1 y t2 se procesan juntos (True) o por separado (False)
    band_groups: Optional[List[List[str]]] = None  # solo SatMAE: agrupacion espectral
    notes: str = ""


CLAY_V15 = ModelSpec(
    name="clay_v1.5",
    chip_size=256,
    patch_size=8,
    embed_dim=1024,
    native_gsd_m=10.0,  # se le indica el GSD real de la fuente en cada llamada
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    uses_metadata=True,   # gsd, lat/lon, semana/hora
    joint_temporal=False,  # se corre una pasada por fecha
    notes=(
        "El GSD se pasa como argumento explicito al modelo (dynamic embedding block); "
        "no requiere remuestreo del propio Sentinel-2 nativo (10 m)."
    ),
)

SATMAE_FMOW_SENTINEL = ModelSpec(
    name="satmae_vit_large_fmow_sentinel",
    chip_size=96,
    patch_size=8,
    embed_dim=1024,
    native_gsd_m=10.0,  # fMoW-Sentinel ya esta en resolucion nativa de Sentinel-2
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    uses_metadata=False,
    joint_temporal=False,
    band_groups=[
        ["B02", "B03", "B04", "B08"],   # grupo 1: 10 m (RGB + NIR)
        ["B05", "B06", "B07", "B8A"],   # grupo 2: 20 m (red edge + NIR estrecho)
        ["B11", "B12"],                  # grupo 3: 20 m (SWIR)
    ],
    notes=(
        "Se descartan B01, B09, B10 (coastal aerosol, vapor de agua, cirros). "
        "Cada grupo se pasa por su propia capa de patch embedding; el resultado "
        "son 3 mapas de tokens que hay que decidir como combinar (ver metrics.py)."
    ),
)

# Estadisticas oficiales de fMoW-Sentinel (util/datasets.py, clase
# SentinelIndividualImageDataset, repo sustainlab-group/SatMAE), en el orden
# de SENTINEL2_ALL_BANDS (13 bandas). SatMAE NO usa z-score normal: usa
# SentinelNormalize, que estira [mean-2*std, mean+2*std] a [0,255] con clip
# (ver preprocessing.sentinel_normalize_satmae). Sin esto, los embeddings de
# SatMAE quedan casi sin senal (ver discusion: effect_size ~0.5 = azar).
SATMAE_MEAN_13 = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
                   1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
                   1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
SATMAE_STD_13 = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
                  948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
                  1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

PRITHVI_EO2_300M = ModelSpec(
    name="prithvi_eo_2.0_300m",
    chip_size=224,   # tamano de entrenamiento (config oficial: img_size=224). Sin padding ni reescalado.
    patch_size=16,
    embed_dim=1024,
    native_gsd_m=30.0,
    # Orden exacto exigido por el checkpoint oficial: Blue, Green, Red, NIR estrecho, SWIR1, SWIR2
    bands=["B02", "B03", "B04", "B8A", "B11", "B12"],
    uses_metadata=False,  # modelo base (sin sufijo _tl): coords_encoding=[], no acepta lat/lon/fecha
    joint_temporal=True,  # patch embedding 3D: t1 y t2 se procesan en una sola pasada (T=2)
    notes=(
        "Preentrenado a 30 m (HLS). Se remuestrea Sentinel-2 de 10 m a 30 m por "
        "average pooling 3x3 y se usan chips de 224x224 (6720 m de lado). "
        "No se hace padding ni reescalado: las escenas que a 30 m miden menos de "
        "224 px en alguna dimension se EXCLUYEN de la evaluacion de Prithvi "
        "(ver run_all_prithvi.py, que las lista en 'excluded_pairs')."
    ),
)

# Estadisticas de normalizacion oficiales de Prithvi-EO-2.0-300M (config.json del repo).
# OJO: estan en el orden de PRITHVI_EO2_300M.bands, no en el orden de SENTINEL2_ALL_BANDS.
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

ALL_MODEL_SPECS = {
    "clay": CLAY_V15,
    "satmae": SATMAE_FMOW_SENTINEL,
    "prithvi": PRITHVI_EO2_300M,
}