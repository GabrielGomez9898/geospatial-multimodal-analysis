"""
visualize.py

Seccion G.5 del protocolo, a partir de los .npz de la corrida principal (no
carga modelos; solo lee OSCD para las imagenes RGB de los mapas):

  Histogramas  Distribucion normalizada de la distancia coseno de los tokens
               C y NC, por modelo. Fila 1: todos los pares validos de cada
               modelo. Fila 2: solo los pares validos para los tres modelos.
  Mapas        Para tres pares de OSCD: T1 (RGB), T2 (RGB), mascara de cambio
               y mapa espacial de distancias de cada modelo, con su escala de
               color. Las escalas son propias de cada modelo porque sus
               distancias no son comparables en magnitud.
  UMAP         Proyeccion 2D de los vectores de diferencia |e1 - e2|, una por
               modelo, coloreada por etiqueta C/NC. Muestra balanceada y
               reproducible (semilla fija). Clay y SatMAE usan exactamente los
               mismos lugares del terreno (comparten la grilla de tokens de
               80 m); Prithvi (tokens de 480 m) usa su propia muestra.
               Exploratorio: no es una medida de desempeno y las proyecciones
               de modelos distintos no son comparables entre si.

Uso:
    python visualize.py
    python visualize.py --results-dir results/comparacion_protocolo --pairs cupertino beirut pisa

Salidas en <results-dir>/visualizacion/: fig_histogramas.png,
fig_mapas_<par>.png, fig_umap.png, umap_coordenadas.csv, muestra_umap.json
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics import LABEL_C, LABEL_NC, auroc

MODELS = ["clay", "satmae", "prithvi"]
NAMES = {"clay": "Clay v1.5", "satmae": "SatMAE", "prithvi": "Prithvi EO 2.0"}
COLORS = {"clay": "#2a78d6", "satmae": "#eb6834", "prithvi": "#1baf7a"}
NC_COLOR = "#9a9993"
SEED = 0
UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42}
PRITHVI_NC_SAMPLE = 200   # Prithvi tiene muy pocos tokens C: no se puede balancear


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlecolor": "#1b1b1a", "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    return plt


def clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="#e7e6e1", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def location_keys(d) -> np.ndarray:
    return d["center_row_10m"].astype(np.int64) * 1_000_000 + d["center_col_10m"].astype(np.int64)


def valid_pairs(results_dir: Path, model: str) -> list[str]:
    rows = csv.DictReader(open(results_dir / "evaluacion" / "por_par.csv"))
    return [r["region"] for r in rows if r["model"] == model and r["valid_auroc"] == "True"]


# ---------------------------------------------------------------------------
# Histogramas
# ---------------------------------------------------------------------------

def fig_histogramas(plt, res: Path, out: Path) -> list[dict]:
    pairs = {m: valid_pairs(res, m) for m in MODELS}
    common = sorted(set.intersection(*(set(v) for v in pairs.values())))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    stats = []
    for row, (title, sel) in enumerate((("todos los pares válidos", pairs),
                                        (f"pares comunes ({', '.join(common)})", {m: common for m in MODELS}))):
        for col, m in enumerate(MODELS):
            ax = axes[row, col]
            dc, dnc = [], []
            for region in sel[m]:
                d = np.load(res / m / f"{region}.npz")
                dc.append(d["dist"][d["label"] == LABEL_C]); dnc.append(d["dist"][d["label"] == LABEL_NC])
            dc, dnc = np.concatenate(dc), np.concatenate(dnc)
            hi = float(np.quantile(np.concatenate([dc, dnc]), 0.995))
            bins = np.linspace(0, hi, 50)
            ax.hist(dnc, bins=bins, density=True, histtype="stepfilled", color=NC_COLOR, alpha=0.35, lw=0)
            ax.hist(dnc, bins=bins, density=True, histtype="step", color=NC_COLOR, lw=1.6,
                    label=f"NC (n={len(dnc):,}, media {dnc.mean():.3f})".replace(",", " "))
            label_c = f"C (n={len(dc):,}, media {dc.mean():.3f})".replace(",", " ")
            if len(dc) >= 30:
                ax.hist(dc, bins=bins, density=True, histtype="step", color=COLORS[m], lw=2, label=label_c)
            else:
                # con tan pocos tokens C un histograma normalizado son barras sueltas: se marca cada token
                for i, x in enumerate(dc):
                    ax.axvline(x, ymax=0.35, color=COLORS[m], lw=2, label=label_c + " (cada línea, un token)" if i == 0 else None)
            ax.set_xlim(0, hi)
            ax.set_title(f"{NAMES[m]}: {len(sel[m])} pares", loc="left")
            if col == 0:
                ax.set_ylabel(f"Densidad\n({title})" if row == 0 else "Densidad\n(pares comunes)")
            if row == 1:
                ax.set_xlabel("Distancia coseno T1–T2")
            ax.legend(loc="upper right", fontsize=7.2)
            clean(ax)
            stats.append({"grupo": "todos" if row == 0 else "comunes", "model": m, "pares": len(sel[m]),
                          "n_C": len(dc), "n_NC": len(dnc), "media_C": float(dc.mean()),
                          "media_NC": float(dnc.mean()), "mediana_C": float(np.median(dc)),
                          "mediana_NC": float(np.median(dnc)), "auroc_tokens_agrupados": auroc(dc, dnc)})
    fig.text(0.01, -0.02, "Histogramas normalizados (área = 1) para comparar formas pese al desbalance C/NC. "
             "Cada modelo tiene su propio eje x: las distancias no son comparables en magnitud entre modelos.",
             fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Mapas de distancias
# ---------------------------------------------------------------------------

def rgb(img13: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """RGB (B04, B03, B02) con estiramiento lineal entre percentiles comunes a T1 y T2."""
    x = img13[[3, 2, 1]].astype(np.float32)
    x = (x - lo[:, None, None]) / (hi - lo)[:, None, None]
    return np.clip(x, 0, 1).transpose(1, 2, 0)


def token_raster(d) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Mapa de distancias a resolucion de token y su extent en pixeles de 10 m de la escena."""
    tok = int(d["token_size_m"]) // 10
    r0, c0 = d["center_row_10m"].min(), d["center_col_10m"].min()
    ri = ((d["center_row_10m"] - r0) / tok).astype(int)
    ci = ((d["center_col_10m"] - c0) / tok).astype(int)
    grid = np.full((ri.max() + 1, ci.max() + 1), np.nan)
    grid[ri, ci] = d["dist"]
    top, left = r0 - tok / 2, c0 - tok / 2
    return grid, (left, left + grid.shape[1] * tok, top + grid.shape[0] * tok, top)


def fig_mapa(plt, oscd, res: Path, region: str, out: Path) -> None:
    from run_pipeline import load_oscd_pair
    idx = next(i for i in range(len(oscd)) if oscd.files[i]["region"] == region)
    t1, t2, mask, *_ = load_oscd_pair(oscd, idx)
    h, w = mask.shape
    both = np.concatenate([t1[[3, 2, 1]].reshape(3, -1), t2[[3, 2, 1]].reshape(3, -1)], axis=1)
    lo, hi = np.percentile(both, 2, axis=1), np.percentile(both, 98, axis=1)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3.4))
    panels = [("T1 (RGB)", rgb(t1, lo, hi)), ("T2 (RGB)", rgb(t2, lo, hi)), ("Máscara OSCD", mask)]
    for ax, (title, img) in zip(axes[:3], panels):
        ax.imshow(img, cmap="Greys" if img.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, loc="left")

    rows = {r["model"]: r for r in csv.DictReader(open(res / "evaluacion" / "por_par.csv")) if r["region"] == region}
    for ax, m in zip(axes[3:], MODELS):
        ax.set_facecolor("#e7e6e1")
        path = res / m / f"{region}.npz"
        if path.exists():
            d = np.load(path)
            grid, extent = token_raster(d)
            vmax = float(np.nanquantile(grid, 0.99))
            im = ax.imshow(grid, extent=extent, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(mask, levels=[0.5], colors="#3fd0ff", linewidths=0.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7); cb.set_label("distancia coseno", fontsize=7)
            r = rows.get(m, {})
            a = r.get("auroc", "")
            sub = f"AUROC {float(a):.3f}" if a not in ("", "nan") else "sin tokens C"
            ax.set_title(f"{NAMES[m]} · {int(d['token_size_m'])} m · {sub}", loc="left", fontsize=8.5)
        else:
            ax.text(0.5, 0.5, "escena más pequeña\nque el chip", ha="center", va="center",
                    transform=ax.transAxes, color="#52514e")
            ax.set_title(NAMES[m], loc="left", fontsize=8.5)
    for ax in axes:
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(f"{region}  ({h} × {w} px a 10 m)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.04, "Mapas de distancia: cada celda es un token; gris = terreno no cubierto por chips. "
             "Contorno celeste: cambio según OSCD. Escala de color propia de cada modelo (0 al percentil 99).",
             fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def delta_for(d, token_idx: np.ndarray) -> np.ndarray:
    """|e1 - e2| (float32, aplanado) de los tokens dados; todos deben ser C o NC."""
    rows = np.full(len(d["dist"]), -1, dtype=np.int64)
    rows[d["delta_token_idx"]] = np.arange(len(d["delta_token_idx"]))
    r = rows[token_idx]
    assert (r >= 0).all()
    return d["delta"][r].astype(np.float32).reshape(len(r), -1)


def umap_samples(res: Path) -> dict:
    """Muestra balanceada C/NC. Clay y SatMAE: los mismos lugares. Prithvi: propia."""
    rng = np.random.default_rng(SEED)
    scenes = json.loads((res / "clay" / "manifest.json").read_text())["scenes"]
    cand = {LABEL_C: [], LABEL_NC: []}      # (region, idx_clay, idx_satmae)
    for region in scenes:
        a, b = np.load(res / "clay" / f"{region}.npz"), np.load(res / "satmae" / f"{region}.npz")
        _, ia, ib = np.intersect1d(location_keys(a), location_keys(b), return_indices=True)
        for lab in (LABEL_C, LABEL_NC):
            ok = (a["label"][ia] == lab) & (b["label"][ib] == lab)
            cand[lab] += [(region, i, j) for i, j in zip(ia[ok], ib[ok])]
    n = len(cand[LABEL_C])
    pick_nc = rng.choice(len(cand[LABEL_NC]), size=n, replace=False)
    shared = [(r, i, j, LABEL_C) for r, i, j in cand[LABEL_C]] + \
             [(*cand[LABEL_NC][k], LABEL_NC) for k in sorted(pick_nc)]

    pr_c, pr_nc = [], []
    for region in json.loads((res / "prithvi" / "manifest.json").read_text())["scenes"]:
        d = np.load(res / "prithvi" / f"{region}.npz")
        pr_c += [(region, i) for i in np.flatnonzero(d["label"] == LABEL_C)]
        pr_nc += [(region, i) for i in np.flatnonzero(d["label"] == LABEL_NC)]
    pick = rng.choice(len(pr_nc), size=min(PRITHVI_NC_SAMPLE, len(pr_nc)), replace=False)
    prithvi = [(r, i, LABEL_C) for r, i in pr_c] + [(*pr_nc[k], LABEL_NC) for k in sorted(pick)]
    return {"shared": shared, "prithvi": prithvi}


def fig_umap(plt, res: Path, out: Path, coords_csv: Path) -> dict:
    import umap

    samples = umap_samples(res)

    def gather(m, regions, token_idx):
        """|e1 - e2| de la muestra, leyendo cada .npz una sola vez (en el orden de la muestra)."""
        regions, token_idx = np.array(regions), np.array(token_idx)
        X = None
        for region in dict.fromkeys(regions):
            sel = regions == region
            block = delta_for(np.load(res / m / f"{region}.npz"), token_idx[sel])
            if X is None:
                X = np.empty((len(regions), block.shape[1]), dtype=np.float32)
            X[sel] = block
        return X

    shared, pr = samples["shared"], samples["prithvi"]
    data = {}
    for m, col in (("clay", 1), ("satmae", 2)):
        data[m] = (gather(m, [s[0] for s in shared], [s[col] for s in shared]),
                   np.array([s[3] for s in shared]), [s[0] for s in shared])
    data["prithvi"] = (gather("prithvi", [s[0] for s in pr], [s[1] for s in pr]),
                       np.array([s[2] for s in pr]), [s[0] for s in pr])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    rows, info = [], {}
    for ax, m in zip(axes, MODELS):
        X, y, regions = data[m]
        emb = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
        nc, c = y == LABEL_NC, y == LABEL_C
        ax.scatter(emb[nc, 0], emb[nc, 1], s=5, color=NC_COLOR, alpha=0.55, linewidths=0,
                   label=f"NC (n={int(nc.sum()):,})".replace(",", " "))
        ax.scatter(emb[c, 0], emb[c, 1], s=7 if c.sum() > 50 else 40, color=COLORS[m], alpha=0.8,
                   edgecolor="white" if c.sum() <= 50 else "none", linewidths=0.8,
                   label=f"C (n={int(c.sum()):,})".replace(",", " "))
        ax.set_title(NAMES[m], loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#d9d8d3")
        ax.legend(loc="best", fontsize=7.5, markerscale=1.8)
        for (u, v), lab, reg in zip(emb, y, regions):
            rows.append({"model": m, "region": reg, "label": "C" if lab == LABEL_C else "NC",
                         "umap_1": float(u), "umap_2": float(v)})
        info[m] = {"n_C": int(c.sum()), "n_NC": int(nc.sum()), "dim": int(X.shape[1])}
    fig.text(0.01, -0.05,
             "UMAP de |e1 − e2| (euclidiana, n_neighbors=15, min_dist=0.1, semilla 42); una proyección por modelo. "
             "Clay y SatMAE: mismos lugares, muestra balanceada C/NC.\nPrithvi: solo "
             f"{info['prithvi']['n_C']} tokens C en todo OSCD, no se puede balancear. Exploratorio: la separación "
             "visual depende de los parámetros y no mide desempeño.", fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

    with open(coords_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    return {"params": UMAP_PARAMS, "seed_muestra": SEED, "por_modelo": info,
            "nota": "Clay y SatMAE comparten exactamente los mismos lugares del terreno."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/comparacion_protocolo"))
    ap.add_argument("--oscd-root", type=Path, default=Path("model_assets/clay/data/oscd"))
    ap.add_argument("--pairs", nargs=3, default=["cupertino", "beirut", "pisa"],
                    help="Tres pares para los mapas (por defecto, uno facil, uno grande y uno dificil, "
                         "todos cubiertos por los tres modelos).")
    args = ap.parse_args()
    out = args.results_dir / "visualizacion"
    out.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()

    stats = fig_histogramas(plt, args.results_dir, out / "fig_histogramas.png")
    with open(out / "histogramas_resumen.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        wr.writeheader(); wr.writerows(stats)
    print("Histogramas listos")

    from torchgeo.datasets import OSCD
    oscd = OSCD(root=str(args.oscd_root), split="train", download=False)
    for region in args.pairs:
        fig_mapa(plt, oscd, args.results_dir, region, out / f"fig_mapas_{region}.png")
    print(f"Mapas listos: {', '.join(args.pairs)}")

    info = fig_umap(plt, args.results_dir, out / "fig_umap.png", out / "umap_coordenadas.csv")
    (out / "muestra_umap.json").write_text(json.dumps(info, indent=2))
    print(f"UMAP listo. Salidas en {out}")


if __name__ == "__main__":
    main()
