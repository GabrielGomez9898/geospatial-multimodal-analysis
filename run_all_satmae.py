"""
run_all_satmae.py

Igual que run_all_clay.py, pero para SatMAE. No necesita checkpoint local ni
metadata.yaml (se descarga solo de HuggingFace la primera vez), ni lat/lon/
fechas reales (SatMAE no usa metadatos geograficos).

Uso:
    python run_all_satmae.py \
        --oscd-root model_assets/clay/data/oscd \
        --combine-mode max
"""

from __future__ import annotations
import argparse
from pathlib import Path

from run_pipeline import pick_device, run_for_satmae, add_align_args, extract_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--combine-mode", type=str, choices=["mean", "max"], default="max")
    add_align_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("results/comparacion_protocolo"),
                         help="Los .npz quedan en <out-dir>/satmae/<region>.npz")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from models.satmae_extractor import load_satmae_model
    satmae_model = load_satmae_model(device=device)
    print("SatMAE cargado. Empezando el loop sobre los pares...\n")

    def run_scene(img_t1, img_t2, mask, region, date_t1, date_t2, offsets):
        return run_for_satmae(img_t1, img_t2, mask, satmae_model, offsets,
                              device=device, combine_mode=args.combine_mode)

    extract_all("satmae", args.oscd_root, run_scene, args.align_with, args.out_dir,
                extra_config={"combine_mode": args.combine_mode})


if __name__ == "__main__":
    main()
