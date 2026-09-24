"""
smoke_test_satmae.py

Igual que smoke_test_clay.py: NO usa OSCD, solo confirma que la carga del
modelo y el forward pass funcionan con tu instalacion de `rshf`, antes de
meter datos reales en la ecuacion.

Uso:
    pip install rshf
    python smoke_test_satmae.py
"""

from __future__ import annotations
import numpy as np
import torch

from config import SATMAE_FMOW_SENTINEL
from preprocessing import Chip


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    print(f"Usando device: {device}")

    from models.satmae_extractor import load_satmae_model, extract_satmae_patch_tokens

    print("Cargando SatMAE (variante multiespectral) desde HuggingFace...")
    model = load_satmae_model(device=device)
    print("Modelo cargado OK.\n")

    fake_pixels = np.random.uniform(0, 3000, size=(len(SATMAE_FMOW_SENTINEL.bands), 96, 96)).astype(np.float32)
    chip = Chip(array=fake_pixels, row_off=0, col_off=0)

    # --- Paso diagnostico: ver la forma CRUDA antes de asumir la estructura
    # de 3 grupos espectrales (ver discusion: el pos_embed del checkpoint
    # sugiere que podria ser 1 solo grupo, no 3).
    pixels = torch.from_numpy(fake_pixels).unsqueeze(0).float().to(device)
    tokens_raw, *_ = model.forward_encoder(pixels, mask_ratio=0.0)
    print(f"Forma CRUDA de salida del encoder: {tokens_raw.shape}")
    print(f"  (1 + N patches esperados: si N=144 -> 1 solo grupo; si N=432 -> 3 grupos)\n")

    print("Corriendo forward pass con un chip sintetico (asumiendo 3 grupos)...")
    try:
        emb = extract_satmae_patch_tokens(model, [chip], device=device)[0]

        n_groups = len(SATMAE_FMOW_SENTINEL.band_groups)
        grid = SATMAE_FMOW_SENTINEL.chip_size // SATMAE_FMOW_SENTINEL.patch_size
        print(f"\nForma del embedding: {emb.shape}  (esperado: ({n_groups}, {grid}, {grid}, {SATMAE_FMOW_SENTINEL.embed_dim}))")
        print(f"Contiene NaN: {np.isnan(emb).any()}")
        print(f"Rango de valores: [{emb.min():.4f}, {emb.max():.4f}]")
        print("\n✔ Smoke test OK: la hipotesis de 3 grupos SI calzo con la forma real.")
    except AssertionError as e:
        print(f"\n✗ La hipotesis de 3 grupos NO calzo con la forma real: {e}")
        print("Esto es esperable dado lo que vimos en tokens_raw.shape arriba -- ")
        print("dime ese numero (el de 'Forma CRUDA') y ajusto metrics.py/config.py ")
        print("para la arquitectura real de este checkpoint (probablemente 1 solo grupo).")


if __name__ == "__main__":
    main()