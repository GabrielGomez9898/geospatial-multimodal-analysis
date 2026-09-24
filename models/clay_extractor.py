"""
clay_extractor.py

Requiere: pip install git+https://github.com/Clay-foundation/model.git
Checkpoint oficial: https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt

IMPORTANTE: verifica el nombre exacto de la clase y de los argumentos contra la
version del repo que clones -- el proyecto ha tenido cambios de API entre v1.0 y
v1.5 (ver su migration-guide). Lo que sigue esta basado en la interfaz publica
documentada en sus tutoriales oficiales (ClayMAEModule + ClayDataModule).
"""

from __future__ import annotations
from typing import List
import math
import numpy as np
import torch

from preprocessing import Chip
from config import CLAY_V15


def normalize_latlon(lat: float, lon: float) -> List[float]:
    """
    Replica exacta de la funcion del tutorial oficial de Clay: lat/lon deben
    llegar al modelo como seno/coseno (4 valores), NO como grados crudos.
    https://clay-foundation.github.io/model/tutorials/wall-to-wall.html
    """
    lat_rad = lat * math.pi / 180
    lon_rad = lon * math.pi / 180
    return [math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)]


def normalize_timestamp(week: int, hour: int) -> List[float]:
    """
    Replica exacta de la funcion del tutorial oficial de Clay: semana/hora
    deben llegar como seno/coseno (4 valores), NO como enteros crudos.
    """
    week_rad = week * 2 * math.pi / 52
    hour_rad = hour * 2 * math.pi / 24
    return [math.sin(week_rad), math.cos(week_rad), math.sin(hour_rad), math.cos(hour_rad)]


def load_clay_model(checkpoint_path: str, metadata_path: str, device: str = "cuda"):
    # TODO: confirmar import exacto en tu version clonada del repo.
    from claymodel.module import ClayMAEModule  # nombre segun docs oficiales

    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        mask_ratio=0.0,   # sin enmascarar: queremos la representacion completa
        shuffle=False,    # sin barajar parches: preservar orden espacial
    )
    module.eval()
    module.to(device)
    return module


@torch.no_grad()
def extract_clay_patch_tokens(
    model,
    chips: List[Chip],
    gsd_m: float,
    wavelengths_um: List[float],
    lat: float,
    lon: float,
    week_hour: tuple,
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Devuelve, por cada chip, un array (32, 32, 1024): la grilla de patch tokens
    de la ultima capa del encoder (sin el token CLS, que se descarta aqui porque
    el protocolo pide conservar la organizacion espacial de los patch tokens).

    gsd_m, wavelengths_um, lat, lon, week_hour son los metadatos que Clay exige
    como parte de su dynamic embedding block (seccion B del protocolo: "El uso
    de metadatos en Prithvi y Clay se conserva").
    """
    outputs = []
    for chip in chips:
        pixels = torch.from_numpy(chip.array).unsqueeze(0).float().to(device)  # (1, C, 256, 256)

        # Formato confirmado contra el tutorial oficial de Clay (wall-to-wall.html):
        # latlon y time van pre-codificados en seno/coseno de 4 valores cada uno,
        # no como (lat,lon)/(week,hour) crudos.
        datacube = {
            "pixels": pixels,
            "gsd": torch.tensor([gsd_m], device=device),
            "waves": torch.tensor(wavelengths_um, device=device),
            "latlon": torch.tensor([normalize_latlon(lat, lon)], device=device),          # (1, 4)
            "time": torch.tensor([normalize_timestamp(*week_hour)], device=device),        # (1, 4)
        }

        unmsk_patch, *_ = model.model.encoder(datacube)
        # unmsk_patch: (1, 1 + n_patches, 1024) -> [CLS, patch_1, ..., patch_1024]
        patch_tokens = unmsk_patch[:, 1:, :]  # descartar CLS
        n_patches = patch_tokens.shape[1]
        grid = int(n_patches ** 0.5)  # deberia ser 32 para chip_size=256, patch_size=8
        assert grid * grid == n_patches, f"Grid no cuadrado: {n_patches} patches"

        grid_np = patch_tokens.reshape(grid, grid, CLAY_V15.embed_dim).cpu().numpy()
        outputs.append(grid_np)

    return outputs