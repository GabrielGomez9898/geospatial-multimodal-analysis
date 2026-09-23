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
import numpy as np
import torch

from preprocessing import Chip
from config import CLAY_V15


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

        # TODO: el nombre/forma exacta del "datacube" de entrada depende de la
        # version del repo -- normalmente es un dict con: pixels, gsd, waves,
        # latlon, time. Revisa ClayDataModule / el notebook de tutoriales
        # "Explore embeddings from Clay Encoder" para el formato exacto.
        datacube = {
            "pixels": pixels,
            "gsd": torch.tensor([gsd_m], device=device),
            "waves": torch.tensor(wavelengths_um, device=device),
            "latlon": torch.tensor([[lat, lon]], device=device),
            "time": torch.tensor([week_hour], device=device),
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
