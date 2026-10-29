"""
run_all_clay.py

Extrae Clay sobre los 14 pares de OSCD (split "train", los unicos con mascara
publica) y guarda, por escena, la distancia coseno, la fraccion de cambio p,
la etiqueta C/NC/mixta y |e1 - e2| de cada token (ver
run_pipeline.save_scene_tokens). Los embeddings se descartan chip a chip.
Las metricas se calculan despues con evaluate.py.

Uso:
    python run_all_clay.py \
        --oscd-root model_assets/clay/data/oscd \
        --clay-checkpoint model_assets/clay/clay-v1.5.ckpt \
        --clay-metadata model_assets/clay/metadata.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path

from run_pipeline import (
    pick_device,
    get_scene_metadata,
    run_for_clay,
    add_align_args,
    extract_all,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-root", type=Path, required=True)
    parser.add_argument("--clay-checkpoint", type=Path, required=True)
    parser.add_argument("--clay-metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    add_align_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("results/comparacion_protocolo"),
                         help="Los .npz quedan en <out-dir>/clay/<region>.npz")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from models.clay_extractor import load_clay_model
    clay_model = load_clay_model(
        checkpoint_path=str(args.clay_checkpoint),
        metadata_path=str(args.clay_metadata),
        device=device,
    )
    print("Clay cargado. Empezando el loop sobre los pares...\n")

    def run_scene(img_t1, img_t2, mask, region, date_t1, date_t2, offsets):
        meta_t1 = get_scene_metadata(region, date_t1)
        meta_t2 = get_scene_metadata(region, date_t2)
        return run_for_clay(img_t1, img_t2, mask, clay_model, meta_t1, meta_t2, offsets, device=device)

    extract_all("clay", args.oscd_root, run_scene, args.align_with, args.out_dir)


if __name__ == "__main__":
    main()
