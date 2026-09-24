"""
smoke_test_clay.py

Prueba minima: NO descarga OSCD, NO usa datos reales. Genera un chip
aleatorio del tamano correcto y lo pasa por Clay para confirmar que:
  1. el checkpoint carga sin errores
  2. el formato del "datacube" que armamos en clay_extractor.py es el que
     tu version instalada del repo realmente espera
  3. la forma de salida es la esperada: (32, 32, 1024), sin NaN/Inf

Esto aisla los problemas de "mi codigo no llama bien a la API de Clay" de
los problemas de "algo salio mal leyendo OSCD" -- son dos fuentes de error
distintas y conviene descartar una antes de meter la otra en la ecuacion.

Uso:
    python smoke_test_clay.py --checkpoint /ruta/clay-v1.5.ckpt --metadata /ruta/metadata.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from config import CLAY_V15
from preprocessing import Chip


def pick_device() -> str:
    """Detecta el mejor dispositivo disponible: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None,
                         help="cuda, mps o cpu. Si no se especifica, se detecta automaticamente.")
    args = parser.parse_args()
    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from models.clay_extractor import load_clay_model, extract_clay_patch_tokens

    print(f"Cargando Clay desde {args.checkpoint} ...")
    model = load_clay_model(str(args.checkpoint), str(args.metadata), device=device)
    print("Modelo cargado OK.\n")

    # Chip sintetico: valores en un rango parecido a reflectancia Sentinel-2
    # escalada (uint16, tipicamente 0-10000). No importa que sea ruido --
    # solo estamos probando que el forward pass corre y da la forma esperada.
    fake_pixels = np.random.uniform(0, 3000, size=(len(CLAY_V15.bands), 256, 256)).astype(np.float32)
    chip = Chip(array=fake_pixels, row_off=0, col_off=0)

    wavelengths_um = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]

    print("Corriendo forward pass con un chip sintetico...")
    emb = extract_clay_patch_tokens(
        model, [chip],
        gsd_m=10.0,
        wavelengths_um=wavelengths_um,
        lat=40.0, lon=-3.0,     # coordenadas cualquiera, no importan para esta prueba
        week_hour=(20, 10),      # semana 20, 10am, tampoco importa para esta prueba
        device=device,
    )[0]

    print(f"\nForma del embedding: {emb.shape}  (esperado: (32, 32, {CLAY_V15.embed_dim}))")
    print(f"Contiene NaN: {np.isnan(emb).any()}")
    print(f"Contiene Inf: {np.isinf(emb).any()}")
    print(f"Rango de valores: [{emb.min():.4f}, {emb.max():.4f}]")

    assert emb.shape == (32, 32, CLAY_V15.embed_dim), "Forma inesperada -- revisa el reshape en clay_extractor.py"
    assert not np.isnan(emb).any(), "El embedding tiene NaN -- revisa la normalizacion/datacube"

    print("\n✔ Smoke test OK: Clay carga y corre correctamente con tu instalacion del repo.")


if __name__ == "__main__":
    main()