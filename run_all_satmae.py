"""
run_all_satmae.py

Igual que run_all_clay.py, pero para SatMAE. No necesita checkpoint local ni
metadata.yaml (se descarga solo de HuggingFace la primera vez), ni lat/lon/
fechas reales (SatMAE no usa metadatos geograficos).

Uso:
    python run_all_satmae.py \
        --oscd-root model_assets/clay/data/oscd \
        --align-with clay satmae prithvi \
        --combine-mode max \
        --out results/satmae/satmae_all_pairs.json
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import SATMAE_FMOW_SENTINEL
from preprocessing import resample_20m_bands_to_10m, common_windows, aligned_chip_offsets, footprint_px_10m
from run_pipeline import (pick_device, load_oscd_pair, run_for_satmae,
                          add_align_args, align_specs_from_names, window_info)
from run_all_clay import aggregate  # misma funcion, no hace falta duplicarla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    add_align_args(parser)
    parser.add_argument("--combine-mode", type=str, choices=["mean", "max"], default="max")
    parser.add_argument("--out", type=Path, default=Path("results/satmae/satmae_all_pairs.json"))
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()
    if args.csv_out is None:
        args.csv_out = args.out.with_name(args.out.stem + "_chips.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from torchgeo.datasets import OSCD
    oscd = OSCD(root=str(args.oscd_root), split="train", download=True)
    print(f"Pares en el dataset: {len(oscd)}")

    from models.satmae_extractor import load_satmae_model
    satmae_model = load_satmae_model(device=device)
    print("SatMAE cargado. Empezando el loop sobre los pares...\n")

    align_specs = align_specs_from_names(args.align_with, "satmae")
    print(f"Ventanas alineadas con: {[s.name for s in align_specs]}")

    per_pair = {}
    all_chip_rows = []
    failed_pairs = []
    excluded_pairs = []  # escenas donde no cabe ni una ventana comun

    for index in range(len(oscd)):
        try:
            img_t1, img_t2, mask, region, date_t1_str, date_t2_str = load_oscd_pair(oscd, index)
            img_t1 = resample_20m_bands_to_10m(img_t1)
            img_t2 = resample_20m_bands_to_10m(img_t2)

            print(f"[{index+1}/{len(oscd)}] {region:15s} tamano={img_t1.shape[1]}x{img_t1.shape[2]}", end="  ")

            window, _ = common_windows(img_t1.shape[1], img_t1.shape[2], align_specs)
            offsets = aligned_chip_offsets(img_t1.shape[1], img_t1.shape[2], SATMAE_FMOW_SENTINEL, align_specs)
            if not offsets:
                print(f"-> EXCLUIDA (no cabe una ventana comun de {window}x{window} px a 10 m)")
                excluded_pairs.append({
                    "region": region,
                    "size_10m": [int(img_t1.shape[1]), int(img_t1.shape[2])],
                    "window_px_10m": window,
                    "reason": "La escena es mas chica que la ventana comun (huella mas grande de "
                              "--align-with). Se excluye para todos los modelos.",
                })
                continue

            chip_results = run_for_satmae(img_t1, img_t2, mask, satmae_model, offsets,
                                           device=device, combine_mode=args.combine_mode)
            chip_results = [{**info, **r} for info, r in
                            zip(window_info(offsets, SATMAE_FMOW_SENTINEL), chip_results)]

            auprc_vals = [r["auprc"] for r in chip_results]
            effect_vals = [r["effect_size_auc"] for r in chip_results]

            per_pair[region] = {
                "n_chips": len(chip_results),
                "auprc": aggregate(auprc_vals),
                "effect_size_auc": aggregate(effect_vals),
                "chips": chip_results,
            }

            for chip_idx, r in enumerate(chip_results):
                all_chip_rows.append({"region": region, "chip_idx": chip_idx, **r})

            print(f"-> {len(chip_results)} chips, "
                  f"effect_size medio={aggregate(effect_vals)['mean']:.3f}")

        except Exception as e:
            print(f"FALLO en el par {index} ({e.__class__.__name__}: {e})")
            failed_pairs.append({"index": index, "error": str(e)})

    all_auprc = [c["auprc"] for pair in per_pair.values() for c in pair["chips"]]
    all_effect = [c["effect_size_auc"] for pair in per_pair.values() for c in pair["chips"]]
    pair_level_auprc = [p["auprc"]["mean"] for p in per_pair.values() if p["auprc"]["n_valid"] > 0]
    pair_level_effect = [p["effect_size_auc"]["mean"] for p in per_pair.values() if p["effect_size_auc"]["n_valid"] > 0]

    summary = {
        "combine_mode": args.combine_mode,
        "align_with": [s.name for s in align_specs],
        "window_px_10m": max(footprint_px_10m(s) for s in align_specs),
        "chip_footprint_m": footprint_px_10m(SATMAE_FMOW_SENTINEL) * 10,
        "excluded_pairs": excluded_pairs,
        "per_pair": per_pair,
        "failed_pairs": failed_pairs,
        "aggregate_across_all_chips": {
            "auprc": aggregate(all_auprc),
            "effect_size_auc": aggregate(all_effect),
        },
        "aggregate_across_pairs": {
            "auprc": aggregate(pair_level_auprc),
            "effect_size_auc": aggregate(pair_level_effect),
        },
    }

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nResultados guardados en {args.out}")

    if all_chip_rows:
        fieldnames = []
        for row in all_chip_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(all_chip_rows)
        print(f"CSV por chip guardado en {args.csv_out}")

    print(f"\n=== RESUMEN (SatMAE, combine_mode={args.combine_mode}) ===")
    print(f"Pares evaluados: {len(per_pair)} / {len(oscd)}  "
          f"(excluidos por tamano: {len(excluded_pairs)}, fallidos: {len(failed_pairs)})")
    print(f"Effect size (por escena) -> media={summary['aggregate_across_pairs']['effect_size_auc']['mean']:.3f}, "
          f"std={summary['aggregate_across_pairs']['effect_size_auc']['std']:.3f}")
    print(f"AUPRC (por escena)       -> media={summary['aggregate_across_pairs']['auprc']['mean']:.3f}, "
          f"std={summary['aggregate_across_pairs']['auprc']['std']:.3f}")


if __name__ == "__main__":
    main()