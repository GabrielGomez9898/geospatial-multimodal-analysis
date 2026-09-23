"""
satmae_extractor.py

Requiere: git clone https://github.com/sustainlab-group/SatMAE
Checkpoint: pesos de fMoW-Sentinel, ViT-Large, variante "group_c" (grouped channels),
publicados por los autores (ver README del repo, seccion "Pretrained Models").

IMPORTANTE: el repo original define la arquitectura "group_c" en models_mae_group_channels.py
/ models_vit_group_channels.py (el nombre exacto de archivo/clase puede variar segun el
commit). Verifica contra tu clon del repo antes de correr esto.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch

from preprocessing import Chip
from config import SATMAE_FMOW_SENTINEL


def load_satmae_model(checkpoint_path: str, device: str = "cuda"):
    # TODO: import exacto segun el repo clonado, p.ej.:
    #   from models_vit_group_channels import vit_large_patch8_group_c
    #   model = vit_large_patch8_group_c(
    #       img_size=SATMAE_FMOW_SENTINEL.chip_size,
    #       patch_size=SATMAE_FMOW_SENTINEL.patch_size,
    #       channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),  # indices dentro de spec.bands
    #   )
    #   checkpoint = torch.load(checkpoint_path, map_location="cpu")
    #   model.load_state_dict(checkpoint["model"], strict=False)
    raise NotImplementedError(
        "Conecta aqui la clase real del repo clonado de SatMAE "
        "(ver comentario TODO arriba)."
    )


@torch.no_grad()
def extract_satmae_patch_tokens(
    model,
    chips: List[Chip],
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Devuelve, por cada chip, un array (n_grupos=3, 12, 12, 1024): los patch tokens
    de la ultima capa del encoder, organizados por grupo espectral y posicion
    espacial. El protocolo pide conservar "la organizacion de tokens" -- aqui se
    conserva TANTO la espacial como la de grupo, para que tu decidas en metrics.py
    si los combinas (p. ej. concatenando a lo largo de la dimension de canal, o
    promediando entre grupos antes de comparar t1 vs t2).
    """
    outputs = []
    for chip in chips:
        pixels = torch.from_numpy(chip.array).unsqueeze(0).float().to(device)  # (1, 10, 96, 96)

        # TODO: la firma exacta de forward() depende de la implementacion "group_c".
        # Normalmente devuelve la secuencia completa de tokens: [CLS, g1_p1...g1_p144,
        # g2_p1...g2_p144, g3_p1...g3_p26] -- confirma el orden contra el codigo fuente.
        tokens = model.forward_encoder_no_mask(pixels)  # (1, 1 + 3*144, 1024), por ejemplo
        patch_tokens = tokens[:, 1:, :]  # descartar CLS

        n_groups = len(SATMAE_FMOW_SENTINEL.band_groups)
        grid = SATMAE_FMOW_SENTINEL.chip_size // SATMAE_FMOW_SENTINEL.patch_size  # 12
        expected = n_groups * grid * grid
        assert patch_tokens.shape[1] == expected, (
            f"Se esperaban {expected} patch tokens ({n_groups} grupos x {grid}x{grid}), "
            f"se obtuvieron {patch_tokens.shape[1]}. Revisa el orden real de tokens del modelo."
        )

        grouped = patch_tokens.reshape(n_groups, grid, grid, SATMAE_FMOW_SENTINEL.embed_dim)
        outputs.append(grouped.cpu().numpy())

    return outputs
