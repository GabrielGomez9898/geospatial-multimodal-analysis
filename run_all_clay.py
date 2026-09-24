"""
run_all_clay.py

Corre Clay sobre los 14 pares de OSCD (split "train", los unicos con mascara
publica) y agrega los resultados. Reutiliza toda la logica de run_pipeline.py
en vez de duplicarla -- este script es solo el loop + la agregacion.

Uso:
    python run_all_clay.py \
        --oscd-root model_assets/clay/data/oscd \
        --clay-checkpoint model_assets/clay/clay-v1.5.ckpt \
        --clay-metadata model_assets/clay/metadata.yaml \
        --chip-overlap 0.5 \
        --out clay_all_pairs.json
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import CLAY_V15
from preprocessing import resample_20m_bands_to_10m
from run_pipeline import (
    pick_device,
    load_oscd_pair,
    parse_oscd_date,
    get_scene_metadata,
    run_for_clay,
)


def aggregate(values: list[float]) -> dict:
    """Media, desviacion estandar y n, ignorando NaN (chips sin ambas clases)."""
    arr = np.array(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    n_nan = int(np.isnan(arr).sum())
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n_valid": 0, "n_nan": n_nan}
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "n_valid": int(len(valid)),
        "n_nan": n_nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-root", type=Path, required=True)
    parser.add_argument("--clay-checkpoint", type=Path, required=True)
    parser.add_argument("--clay-metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chip-overlap", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/clay/clay_all_pairs.json"))
    parser.add_argument("--csv-out", type=Path, default=None,
                         help="CSV con una fila por chip. Por defecto, mismo nombre que --out "
                              "pero con sufijo _chips.csv, en la misma carpeta.")
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

    from models.clay_extractor import load_clay_model
    clay_model = load_clay_model(
        checkpoint_path=str(args.clay_checkpoint),
        metadata_path=str(args.clay_metadata),
        device=device,
    )
    print("Clay cargado. Empezando el loop sobre los pares...\n")

    clay_stride = int(CLAY_V15.chip_size * (1 - args.chip_overlap)) if args.chip_overlap > 0 else None

    per_pair = {}
    all_chip_rows = []  # para el CSV: una fila por chip
    failed_pairs = []

    for index in range(len(oscd)):
        try:
            img_t1, img_t2, mask, region, date_t1_str, date_t2_str = load_oscd_pair(oscd, index)
            img_t1 = resample_20m_bands_to_10m(img_t1)
            img_t2 = resample_20m_bands_to_10m(img_t2)

            date_t1, date_t2 = parse_oscd_date(date_t1_str), parse_oscd_date(date_t2_str)
            meta_t1 = get_scene_metadata(region, date_t1)
            meta_t2 = get_scene_metadata(region, date_t2)

            print(f"[{index+1}/{len(oscd)}] {region:15s} tamano={img_t1.shape[1]}x{img_t1.shape[2]}", end="  ")

            chip_results = run_for_clay(img_t1, img_t2, mask, clay_model, meta_t1, meta_t2,
                                         device=device, stride=clay_stride)

            auprc_vals = [r["auprc"] for r in chip_results]
            effect_vals = [r["effect_size_auc"] for r in chip_results]

            per_pair[region] = {
                "n_chips": len(chip_results),
                "auprc": aggregate(auprc_vals),
                "effect_size_auc": aggregate(effect_vals),
                "chips": chip_results,
            }

            for chip_idx, r in enumerate(chip_results):
                row = {"region": region, "chip_idx": chip_idx, **r}
                all_chip_rows.append(row)

            print(f"-> {len(chip_results)} chips, "
                  f"effect_size medio={aggregate(effect_vals)['mean']:.3f}")

        except Exception as e:
            print(f"FALLO en el par {index} ({e.__class__.__name__}: {e})")
            failed_pairs.append({"index": index, "error": str(e)})

    # --- agregacion global ---
    all_auprc = [c["auprc"] for pair in per_pair.values() for c in pair["chips"]]
    all_effect = [c["effect_size_auc"] for pair in per_pair.values() for c in pair["chips"]]
    pair_level_auprc = [pair["auprc"]["mean"] for pair in per_pair.values() if pair["auprc"]["n_valid"] > 0]
    pair_level_effect = [pair["effect_size_auc"]["mean"] for pair in per_pair.values() if pair["effect_size_auc"]["n_valid"] > 0]

    summary = {
        "per_pair": per_pair,
        "failed_pairs": failed_pairs,
        # Agregado sobre TODOS los chips de TODAS las escenas (ojo: los chips
        # dentro de una misma escena pueden estar solapados si --chip-overlap > 0,
        # asi que no son totalmente independientes entre si).
        "aggregate_across_all_chips": {
            "auprc": aggregate(all_auprc),
            "effect_size_auc": aggregate(all_effect),
        },
        # Agregado usando UN valor por escena (la media de sus chips) -- esta es
        # la version estadisticamente mas defendible, porque las 14 escenas si
        # son independientes entre si (ciudades distintas, fechas distintas).
        "aggregate_across_pairs": {
            "auprc": aggregate(pair_level_auprc),
            "effect_size_auc": aggregate(pair_level_effect),
        },
    }

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nResultados guardados en {args.out}")

    if all_chip_rows:
        # Las filas no todas tienen las mismas columnas (separability_metrics
        # agrega "note" solo quando un chip no tiene ambas clases) -- se arma
        # el encabezado con la union de todas las claves, no solo las de la
        # primera fila.
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

    print(f"\n=== RESUMEN ===")
    print(f"Pares procesados OK: {len(per_pair)} / {len(oscd)}  (fallidos: {len(failed_pairs)})")
    print(f"Effect size (por escena) -> media={summary['aggregate_across_pairs']['effect_size_auc']['mean']:.3f}, "
          f"std={summary['aggregate_across_pairs']['effect_size_auc']['std']:.3f}")
    print(f"AUPRC (por escena)       -> media={summary['aggregate_across_pairs']['auprc']['mean']:.3f}, "
          f"std={summary['aggregate_across_pairs']['auprc']['std']:.3f}")


if __name__ == "__main__":
    main()