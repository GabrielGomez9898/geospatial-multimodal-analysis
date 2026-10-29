"""
evaluate.py

Secciones G.1, G.3 y G.4 del protocolo, calculadas a partir de los .npz que
dejan run_all_*.py (no carga ningun modelo):

  G.1  conteos de tokens C / NC / mixtas (excluidas) y % excluido, por modelo
  G.3  AUROC por par de imagenes y modelo (distancia coseno, C como positiva)
  G.4  indice de Fisher por par de imagenes y modelo

Todas las metricas se calculan POR PAR (juntando los tokens de todos los chips
de la escena) y se resumen con media y desviacion estandar entre pares.

La comparacion principal usa solo los pares validos para TODOS los modelos
evaluados (G.3). Como resultado complementario se reporta cada modelo sobre
todos sus pares validos (p. ej. Clay y SatMAE sobre las 14 escenas, Prithvi
sobre 5).

Uso:
    python evaluate.py
    python evaluate.py --results-dir results/comparacion_protocolo --models clay satmae prithvi

Salidas, en <results-dir>/evaluacion/:
    por_par.csv    una fila por (modelo, par)
    resumen.json   conteos, pares validos/invalidos, comparacion principal y complementaria
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics import LABEL_C, LABEL_NC, LABEL_MIXED, auroc, fisher_index


def evaluate_pair(npz_path: Path) -> dict:
    data = np.load(npz_path)
    labels, dist = data["label"], data["dist"]
    dist_c, dist_nc = dist[labels == LABEL_C], dist[labels == LABEL_NC]
    n_mixed = int((labels == LABEL_MIXED).sum())

    notes = []
    if len(dist_c) == 0:
        notes.append("sin tokens C")
    if len(dist_nc) == 0:
        notes.append("sin tokens NC")
    fisher = fisher_index(dist_c, dist_nc)
    if not notes and np.isnan(fisher):
        notes.append("denominador de Fisher = 0")

    return {
        "region": str(data["region"]),
        "n_chips": int(len(data["chip_offsets_10m"])),
        "token_size_m": int(data["token_size_m"]),
        "n_tokens": int(len(labels)),
        "n_C": int(len(dist_c)),
        "n_NC": int(len(dist_nc)),
        "n_mixed_excluded": n_mixed,
        "pct_excluded": 100.0 * n_mixed / len(labels),
        "valid_auroc": len(dist_c) > 0 and len(dist_nc) > 0,
        "auroc": auroc(dist_c, dist_nc),
        "fisher": fisher,
        "mu_C": float(dist_c.mean()) if len(dist_c) else float("nan"),
        "mu_NC": float(dist_nc.mean()) if len(dist_nc) else float("nan"),
        "var_C": float(dist_c.var()) if len(dist_c) else float("nan"),
        "var_NC": float(dist_nc.var()) if len(dist_nc) else float("nan"),
        "note": "; ".join(notes),
    }


def mean_std(values: list[float]) -> dict:
    """Media y desviacion estandar muestral (ddof=1) entre pares, ignorando NaN."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "n_pairs": int(len(arr)),
    }


def summarize(rows: list[dict], regions: list[str]) -> dict:
    sel = [r for r in rows if r["region"] in regions]
    return {
        "auroc": mean_std([r["auroc"] for r in sel]),
        "fisher": mean_std([r["fisher"] for r in sel]),
        "fisher_zero_denominator": [r["region"] for r in sel if r["note"] == "denominador de Fisher = 0"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/comparacion_protocolo"))
    parser.add_argument("--models", nargs="+", default=["clay", "satmae", "prithvi"])
    args = parser.parse_args()

    out_dir = args.results_dir / "evaluacion"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: dict[str, list[dict]] = {}
    model_info: dict[str, dict] = {}
    for model in args.models:
        model_dir = args.results_dir / model
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[{model}] no hay {manifest_path}: se omite (corre run_all_{model}.py primero)")
            continue
        manifest = json.loads(manifest_path.read_text())
        rows = [evaluate_pair(model_dir / f"{region}.npz") for region in manifest["scenes"]]
        rows_by_model[model] = rows

        totals = {k: sum(r[k] for r in rows) for k in ("n_tokens", "n_C", "n_NC", "n_mixed_excluded")}
        totals["pct_excluded"] = 100.0 * totals["n_mixed_excluded"] / totals["n_tokens"] if totals["n_tokens"] else float("nan")
        model_info[model] = {
            "token_size_m": manifest["token_size_m"],
            "chip_footprint_m": manifest["chip_footprint_m"],
            "token_counts": totals,
            "valid_pairs": [r["region"] for r in rows if r["valid_auroc"]],
            "invalid_pairs": (
                [{"region": r["region"], "reason": r["note"]} for r in rows if not r["valid_auroc"]]
                + [{"region": e["region"], "reason": "escena excluida por tamano (sin chips)"}
                   for e in manifest["excluded_pairs"]]
                + [{"index": f["index"], "reason": f"fallo en la extraccion: {f['error']}"}
                   for f in manifest["failed_pairs"]]
            ),
        }

    if not rows_by_model:
        raise SystemExit("No hay resultados para evaluar.")

    common_pairs = sorted(set.intersection(*(set(info["valid_pairs"]) for info in model_info.values())))
    for model, rows in rows_by_model.items():
        model_info[model]["principal"] = summarize(rows, common_pairs)
        model_info[model]["complementaria"] = summarize(rows, model_info[model]["valid_pairs"])

    summary = {
        "models": list(rows_by_model),
        "common_valid_pairs": common_pairs,
        "notes": {
            "principal": "Media y std (ddof=1) entre los pares validos para TODOS los modelos.",
            "complementaria": "Media y std (ddof=1) entre todos los pares validos de cada modelo.",
            "fisher": "Varianzas poblacionales dentro de cada par; pares con denominador 0 "
                      "se listan en fisher_zero_denominator y no entran en la media.",
        },
        "per_model": model_info,
    }
    (out_dir / "resumen.json").write_text(json.dumps(summary, indent=2))

    all_rows = [{"model": m, **r} for m, rows in rows_by_model.items() for r in rows]
    with open(out_dir / "por_par.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== G.1 Conteos de tokens ===")
    print(f"{'modelo':10s} {'token':>7s} {'C':>8s} {'NC':>9s} {'mixtas':>8s} {'% excl':>7s}")
    for m, info in model_info.items():
        t = info["token_counts"]
        print(f"{m:10s} {info['token_size_m']:>5d} m {t['n_C']:>8d} {t['n_NC']:>9d} "
              f"{t['n_mixed_excluded']:>8d} {t['pct_excluded']:>6.1f}%")

    print(f"\n=== Comparacion principal: {len(common_pairs)} pares comunes ===")
    print(f"Pares: {', '.join(common_pairs)}")
    for key in ("principal", "complementaria"):
        if key == "complementaria":
            print("\n=== Complementaria: todos los pares validos de cada modelo ===")
        print(f"{'modelo':10s} {'n':>3s} {'AUROC':>16s} {'Fisher':>16s}")
        for m, info in model_info.items():
            a, fi = info[key]["auroc"], info[key]["fisher"]
            print(f"{m:10s} {a['n_pairs']:>3d} {a['mean']:>7.3f} ± {a['std']:<6.3f} "
                  f"{fi['mean']:>7.3f} ± {fi['std']:<6.3f}")

    for m, info in model_info.items():
        if info["invalid_pairs"]:
            print(f"\n[{m}] pares sin AUROC: " +
                  ", ".join(f"{p.get('region', p.get('index'))} ({p['reason']})" for p in info["invalid_pairs"]))

    print(f"\nGuardado en {out_dir / 'por_par.csv'} y {out_dir / 'resumen.json'}")


if __name__ == "__main__":
    main()
