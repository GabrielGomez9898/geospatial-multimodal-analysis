"""
json_to_csv.py

Reconstruye el CSV por chip a partir de un clay_all_pairs.json ya generado
(por si el CSV fallo al escribirse pero el JSON si se guardo -- no hace falta
volver a correr el modelo).

Uso:
    python json_to_csv.py --json clay_all_pairs.json --csv-out clay_all_pairs_chips.csv
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.json.read_text())

    rows = []
    for region, pair_data in data["per_pair"].items():
        for chip_idx, chip in enumerate(pair_data["chips"]):
            rows.append({"region": region, "chip_idx": chip_idx, **chip})

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV guardado en {args.csv_out}  ({len(rows)} filas)")


if __name__ == "__main__":
    main()