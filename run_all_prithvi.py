"""
run_all_prithvi.py

Igual que run_all_clay.py / run_all_satmae.py, pero para Prithvi EO 2.0 (300M).
No necesita checkpoint local (se descarga via terratorch/HuggingFace la primera
vez) ni lat/lon/fechas reales (el modelo base, sin sufijo _tl, no acepta
metadatos -- ver prithvi_extractor.py).

Las escenas que no alcanzan una ventana de 672 px a 10 m (224 px a 30 m) se
listan en "excluded_pairs" del manifest.json: con la configuracion actual
Prithvi solo cubre 5 de las 14 escenas.

Uso:
    python run_all_prithvi.py \
        --oscd-root model_assets/clay/data/oscd
"""

from __future__ import annotations
import argparse
from pathlib import Path

from run_pipeline import pick_device, run_for_prithvi, add_align_args, extract_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    add_align_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("results/comparacion_protocolo"),
                         help="Los .npz quedan en <out-dir>/prithvi/<region>.npz")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from models.prithvi_extractor import load_prithvi_model
    prithvi_model = load_prithvi_model(device=device)
    print("Prithvi cargado. Empezando el loop sobre los pares...\n")

    def run_scene(img_t1, img_t2, mask, region, date_t1, date_t2, offsets):
        return run_for_prithvi(img_t1, img_t2, mask, prithvi_model, offsets, device=device)

    extract_all("prithvi", args.oscd_root, run_scene, args.align_with, args.out_dir)


if __name__ == "__main__":
    main()
