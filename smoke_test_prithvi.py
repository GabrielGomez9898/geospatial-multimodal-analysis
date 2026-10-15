"""
smoke_test_prithvi.py

Igual que los otros dos: NO usa OSCD, solo confirma que la carga del modelo y
el forward pass funcionan, y de paso verifica que no exista el mismo problema
de barajado de tokens que encontramos en SatMAE (llamando dos veces con el
mismo input y comparando).

Uso:
    pip install terratorch
    python smoke_test_prithvi.py
"""

from __future__ import annotations
import numpy as np
import torch

from config import PRITHVI_EO2_300M
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

    from models.prithvi_extractor import load_prithvi_model, extract_prithvi_patch_tokens

    print("Cargando Prithvi EO 2.0 (300M) desde HuggingFace via terratorch...")
    model = load_prithvi_model(device=device)
    print("Modelo cargado OK.\n")

    n_bands = len(PRITHVI_EO2_300M.bands)
    size = PRITHVI_EO2_300M.chip_size
    fake_t1 = np.random.uniform(0, 1, size=(n_bands, size, size)).astype(np.float32)
    fake_t2 = np.random.uniform(0, 1, size=(n_bands, size, size)).astype(np.float32)
    chip_t1 = Chip(array=fake_t1, row_off=0, col_off=0)
    chip_t2 = Chip(array=fake_t2, row_off=0, col_off=0)

    print("Corriendo forward pass con un par de chips sinteticos...")
    emb = extract_prithvi_patch_tokens(model, chip_t1, chip_t2, device=device)

    grid = PRITHVI_EO2_300M.chip_size // PRITHVI_EO2_300M.patch_size
    print(f"\nForma del embedding: {emb.shape}  (esperado: (2, {grid}, {grid}, {PRITHVI_EO2_300M.embed_dim}))")
    print(f"Contiene NaN: {np.isnan(emb).any()}")
    print(f"Rango de valores: [{emb.min():.4f}, {emb.max():.4f}]")

    assert emb.shape == (2, grid, grid, PRITHVI_EO2_300M.embed_dim)
    assert not np.isnan(emb).any()

    # --- Diagnostico de barajado (ver lo que paso con SatMAE) ---
    print("\nVerificando estabilidad entre dos llamadas con el MISMO input...")
    emb_again = extract_prithvi_patch_tokens(model, chip_t1, chip_t2, device=device)
    identical = np.allclose(emb, emb_again, atol=1e-4)
    diff = np.abs(emb - emb_again).mean()
    print(f"¿Resultado estable entre llamadas? {identical}  (diferencia promedio: {diff:.8f})")
    if not identical and diff > 0.01:
        print("✗ ALERTA: parece haber el mismo problema de barajado que tuvo SatMAE. Avisame.")
    else:
        print("✔ Estable (la diferencia, si existe, es ruido numerico normal, no barajado).")

    print("\n✔ Smoke test OK: Prithvi carga y corre correctamente con tu instalacion.")


if __name__ == "__main__":
    main()