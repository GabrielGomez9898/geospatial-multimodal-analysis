"""
prithvi_extractor.py

Requiere: pip install terratorch
Checkpoint: se descarga automaticamente via terratorch's BACKBONE_REGISTRY, o
manualmente desde https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M

IMPORTANTE: usamos img_size=192 (no el 224 por defecto), porque preprocessing.py
ya remuestreo la escena a ~30 m y la recorto a un multiplo de 16. El positional
encoding de Prithvi es sinusoidal (no aprendido), por lo que el backbone puede
instanciarse con un img_size distinto al de preentrenamiento sin reentrenar
-- confirma este comportamiento contra tu version de terratorch antes de confiar
en los resultados.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch

from preprocessing import Chip
from config import PRITHVI_EO2_300M


def load_prithvi_model(device: str = "cuda"):
    # TODO: confirmar API exacta de terratorch en tu version instalada.
    from terratorch.registry import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300",
        img_size=PRITHVI_EO2_300M.chip_size,   # 192, ver docstring
        num_frames=2,                           # T=2: t1 y t2 en una sola pasada
        bands=PRITHVI_EO2_300M.bands,
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_prithvi_patch_tokens(
    model,
    chip_t1: Chip,
    chip_t2: Chip,
    lat: float = None,
    lon: float = None,
    date_t1=None,
    date_t2=None,
    device: str = "cuda",
) -> np.ndarray:
    """
    A diferencia de Clay/SatMAE, Prithvi recibe t1 y t2 JUNTOS en una sola pasada
    (patch embedding 3D con T=2), asi que esta funcion no se llama una vez por
    fecha sino una vez por PAR de chips.

    Devuelve un array (2, 12, 12, 1024): eje 0 = indice temporal (0=t1, 1=t2),
    ejes 1-2 = grilla espacial de patch tokens (192/16 = 12), eje 3 = embedding.

    IMPORTANTE (ver discusion previa): estos dos "bloques" temporales NO son
    representaciones independientes como en Clay/SatMAE -- ya pasaron por
    atencion cruzada dentro del encoder. Documentar esto en la interpretacion
    de resultados, no solo en el codigo.
    """
    # (C, T, H, W): apilar t1 y t2 en la dimension temporal
    stacked = np.stack([chip_t1.array, chip_t2.array], axis=1)  # (6, 2, 192, 192)
    pixels = torch.from_numpy(stacked).unsqueeze(0).float().to(device)  # (1, 6, 2, 192, 192)

    temporal_coords = None
    location_coords = None
    if lat is not None and lon is not None:
        location_coords = torch.tensor([[lat, lon]], device=device)
    if date_t1 is not None and date_t2 is not None:
        # TODO: formato exacto esperado por terratorch (year, day-of-year) x 2 fechas
        temporal_coords = torch.tensor([[date_t1, date_t2]], device=device)

    # TODO: confirmar nombre exacto del metodo de forward para features del encoder
    # (algunas versiones exponen model.forward_features(), otras requieren
    # model(pixels, temporal_coords=..., location_coords=...) directamente).
    features = model.forward_features(
        pixels,
        temporal_coords=temporal_coords,
        location_coords=location_coords,
    )
    # features: (1, 1 + T*grid*grid, 1024) si hay CLS, o (1, T*grid*grid, 1024) si no.
    # TODO: verificar si este backbone incluye CLS token; si lo incluye, descartarlo
    # igual que en clay_extractor.py.
    grid = PRITHVI_EO2_300M.chip_size // PRITHVI_EO2_300M.patch_size  # 12
    expected_no_cls = 2 * grid * grid
    if features.shape[1] == expected_no_cls + 1:
        features = features[:, 1:, :]
    assert features.shape[1] == expected_no_cls, (
        f"Se esperaban {expected_no_cls} tokens (T=2 x {grid}x{grid}), "
        f"se obtuvieron {features.shape[1]}."
    )

    out = features.reshape(2, grid, grid, PRITHVI_EO2_300M.embed_dim)
    return out.cpu().numpy()
