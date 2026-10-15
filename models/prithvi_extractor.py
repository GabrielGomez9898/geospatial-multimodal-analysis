"""
prithvi_extractor.py

Requiere: pip install terratorch
Checkpoint: se descarga automaticamente la primera vez, via BACKBONE_REGISTRY.

API confirmada contra la documentacion oficial (huggingface.co/ibm-nasa-geospatial/
Prithvi-EO-2.0-300M y ibm.github.io/terratorch/stable/guide/prithvi_eo):

    from terratorch.registry import BACKBONE_REGISTRY
    model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", pretrained=True, ...)
    features = model.forward_features(tensor)

IMPORTANTE -- a diferencia de Clay: el modelo base "prithvi_eo_v2_300" (sin
sufijo _tl) tiene coords_encoding=[] en su config.json oficial, es decir, NO
acepta lat/lon/fecha como entrada en absoluto. Solo las variantes con sufijo
_tl (prithvi_eo_v2_300_tl, etc.) aceptan esos metadatos. Por eso este extractor,
a diferencia de clay_extractor.py, no arma ningun datacube de metadatos.

Lo que SI sigue siendo una hipotesis sin verificar contra una ejecucion real
(iremos ajustando con el smoke test, igual que con Clay y SatMAE):
- El orden exacto de ejes que espera forward_features: aqui se usa (B, C, T, H, W),
  la convencion estandar de PyTorch para Conv3d (que es como estan implementados
  los 3D patch embeddings de Prithvi segun su model card).
- Si forward_features devuelve o no un token CLS al inicio de la secuencia.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch

from preprocessing import Chip
from config import PRITHVI_EO2_300M

PRITHVI_BACKBONE_NAME = "prithvi_eo_v2_300"


def load_prithvi_model(device: str = "cuda", chip_size: int | None = None):
    """chip_size: tamano de entrada con el que se construye el backbone (multiplo
    de patch_size). Por defecto el de config.py (224, el de entrenamiento)."""
    from terratorch.registry import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        PRITHVI_BACKBONE_NAME,
        pretrained=True,
        img_size=chip_size or PRITHVI_EO2_300M.chip_size,
        num_frames=2,                          # T=2: t1 y t2 en una sola pasada
        bands=PRITHVI_EO2_300M.bands,
    )
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_prithvi_patch_tokens(
    model,
    chip_t1: Chip,
    chip_t2: Chip,
    device: str = "cuda",
) -> np.ndarray:
    """
    Como Prithvi procesa t1 y t2 JUNTOS (patch embedding 3D con T=2), esta
    funcion recibe el PAR de chips y hace una sola pasada -- no se llama una
    vez por fecha, a diferencia de Clay/SatMAE.

    Devuelve (2, grid, grid, 1024): eje 0 = indice temporal (0=t1, 1=t2). Con chip de 224, grid = 14.

    IMPORTANTE (ver discusion previa en la conversacion): estos dos "bloques"
    temporales NO son representaciones independientes como en Clay/SatMAE --
    ya interactuaron via atencion dentro del encoder. Documentar esto en la
    interpretacion de resultados.
    """
    # (C, T, H, W): apilar t1 y t2 en la dimension temporal (eje 1, despues de canal)
    stacked = np.stack([chip_t1.array, chip_t2.array], axis=1)  # (C, 2, H, W)
    pixels = torch.from_numpy(stacked).unsqueeze(0).float().to(device)  # (1, C, 2, H, W)

    features = model.forward_features(pixels)
    if isinstance(features, (list, tuple)):
        # Algunos backbones de terratorch devuelven una lista (una entrada por
        # bloque del encoder, para usarse como feature pyramid); si pasa esto,
        # nos quedamos con la ultima capa, que es la salida final del encoder.
        features = features[-1]

    # La grilla se deduce del chip real: 224/16 = 14 tokens por lado (14x14 = 196 por fecha).
    grid = chip_t1.array.shape[-1] // PRITHVI_EO2_300M.patch_size
    expected_no_cls = 2 * grid * grid

    n_tokens = features.shape[1]
    if n_tokens == expected_no_cls + 1:
        features = features[:, 1:, :]  # descartar CLS
    elif n_tokens != expected_no_cls:
        raise ValueError(
            f"Forma inesperada de salida de Prithvi: {features.shape}. "
            f"Se esperaban {expected_no_cls} tokens (T=2 x {grid}x{grid}) "
            f"o {expected_no_cls + 1} (con CLS). Revisa forward_features() "
            f"contra tu version instalada de terratorch."
        )

    out = features.reshape(1, 2, grid, grid, PRITHVI_EO2_300M.embed_dim)[0]
    return out.cpu().numpy()