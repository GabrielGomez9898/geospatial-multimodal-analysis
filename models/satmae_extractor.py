"""
satmae_extractor.py

Usa la libreria `rshf` (Remote Sensing HuggingFace-style loading), que publica
el checkpoint multiespectral de SatMAE ("SatMAE+Group", ViT-Large, fMoW-Sentinel)
en un formato mucho mas simple de cargar que el repo de investigacion original.

Requiere: pip install rshf
Checkpoint: se descarga automaticamente desde HuggingFace la primera vez que
corras esto: "MVRL/satmae-vitlarge-multispec-pretrain"

IMPORTANTE -- a diferencia de Clay, no pude confirmar el nombre exacto de la
clase para la variante MULTIESPECTRAL desde afuera (el unico ejemplo publico
que encontre es para la variante RGB simple, con otra clase/config). Antes de
correr nada mas, ejecuta esto para descubrir que expone el paquete:

    python -c "import rshf.satmae as m; print([x for x in dir(m) if not x.startswith('_')])"

y ajusta el import de abajo con el nombre real que aparezca (probablemente
algo como SatMAE o SatMAEGroupChannels). Este archivo asume, como primera
hipotesis, que es la misma clase `SatMAE` que la variante RGB, solo que el
checkpoint trae su propio config.json con channel_groups -- si el forward
falla, dime el error y lo ajustamos, igual que con Clay.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch

from preprocessing import Chip
from config import SATMAE_FMOW_SENTINEL

SATMAE_CHECKPOINT_ID = "MVRL/satmae-vitlarge-multispec-pretrain"


def load_satmae_model(device: str = "cuda"):
    from rshf.satmae import SatMAE_Pre_MS

    model = SatMAE_Pre_MS.from_pretrained(SATMAE_CHECKPOINT_ID)
    model.eval()
    model.to(device)
    return model


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

        # Confirmado por el ejemplo publico de rshf (variante RGB):
        # model.forward_encoder(input, mask_ratio=0.0) -> (tokens, mask, ids_restore)
        # IMPORTANTE (ver discusion): incluso con mask_ratio=0.0, el modelo
        # devuelve los patch tokens en un orden ALEATORIO distinto en cada
        # llamada (comportamiento estandar de la implementacion MAE). Hay que
        # usar ids_restore para revertir esa permutacion ANTES de reorganizar
        # en el grid espacial -- si no, se compara ruido, no posiciones reales.
        tokens, mask, ids_restore = model.forward_encoder(pixels, mask_ratio=0.0)
        patch_tokens = tokens[:, 1:, :]  # descartar CLS -> (1, 432, 1024)

        idx = ids_restore.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        patch_tokens = torch.gather(patch_tokens, dim=1, index=idx)  # orden real restaurado

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