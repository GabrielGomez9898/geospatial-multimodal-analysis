"""
compare_overlap.py

Compara la corrida sin solape (results/comparacion_protocolo) con la de 50 %
de solape (results/comparacion_protocolo_50_solapamiento), a dos niveles:

  1. Metricas por par (AUROC y Fisher, seccion G): ¿cambia la conclusion?
     Se agrega una tercera variante, "50 % promediado por lugar": con solape,
     cada lugar del terreno aparece en varios chips; se promedia su distancia
     y se cuenta una sola vez, para separar el efecto de cubrir mas terreno
     del efecto de contar el mismo terreno varias veces.

  2. Embeddings, token a token. Las dos corridas comparten la grilla de tokens
     sobre el terreno (ver preprocessing._axis_offsets), asi que el mismo
     (center_row_10m, center_col_10m) es el mismo lugar en ambas. Para cada
     lugar se compara:
       - la distancia coseno T1-T2 sin solape vs. la de cada chip que lo
         cubre con solape (mismo lugar, distinto contexto alrededor);
       - el vector de diferencia |e1 - e2| (solo tokens C y NC): similitud
         coseno y razon de normas entre ambas corridas;
       - como varian esas diferencias segun la posicion del token en el chip
         (borde vs. centro).

No carga modelos: lee los .npz y evaluacion/por_par.csv de ambas carpetas.

Uso:
    python compare_overlap.py
    python compare_overlap.py --sin results/comparacion_protocolo \\
        --con results/comparacion_protocolo_50_solapamiento --out results/comparativa_solapamiento

Salidas en --out: tablas/*.csv, figuras/*.png, resumen.json, LEEME.md
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from metrics import LABEL_C, LABEL_NC, LABEL_MIXED, auroc, fisher_index

MODELS = ["clay", "satmae", "prithvi"]
NAMES = {"clay": "Clay v1.5", "satmae": "SatMAE", "prithvi": "Prithvi EO 2.0"}
# Paleta categorica validada (slots 1-3, pasan todos los pares en CVD), con
# marcador distinto por modelo como codificacion secundaria.
COLORS = {"clay": "#2a78d6", "satmae": "#eb6834", "prithvi": "#1baf7a"}
MARKERS = {"clay": "o", "satmae": "s", "prithvi": "D"}
CHUNK = 8192


# ---------------------------------------------------------------------------
# Cruce de tokens entre corridas
# ---------------------------------------------------------------------------

def location_keys(d) -> np.ndarray:
    """Clave entera unica por lugar del terreno (centro del token en la grilla de 10 m)."""
    return d["center_row_10m"].astype(np.int64) * 1_000_000 + d["center_col_10m"].astype(np.int64)


def delta_rows(d) -> np.ndarray:
    """Para cada token, fila de `delta` que le corresponde (-1 si es mixto y no se guardo)."""
    rows = np.full(len(d["dist"]), -1, dtype=np.int64)
    rows[d["delta_token_idx"]] = np.arange(len(d["delta_token_idx"]))
    return rows


def border_fraction(d) -> np.ndarray:
    """Posicion del token en su chip: 0 = fila/columna del borde, 1 = centro del chip."""
    g = int(d["grid_size"])
    r, c = d["token_row"].astype(np.int64), d["token_col"].astype(np.int64)
    b = np.minimum.reduce([r, c, g - 1 - r, g - 1 - c])
    return b / ((g - 1) // 2)


def compare_scene(sin_path: Path, con_path: Path) -> dict:
    s, c = np.load(sin_path), np.load(con_path)
    ks, kc = location_keys(s), location_keys(c)
    assert len(np.unique(ks)) == len(ks), f"{sin_path}: la corrida sin solape repite lugares"

    # --- lugares de la corrida con solape (cada uno aparece en 1 o mas chips)
    loc_keys, loc_inv, loc_count = np.unique(kc, return_inverse=True, return_counts=True)
    loc_mean = np.bincount(loc_inv, weights=c["dist"].astype(np.float64)) / loc_count
    first = np.zeros(len(loc_keys), dtype=np.int64)
    first[loc_inv[::-1]] = np.arange(len(kc))[::-1]
    loc_label = c["label"][first]

    # --- cruce con la corrida sin solape
    order = np.argsort(ks)
    pos = np.clip(np.searchsorted(ks[order], kc), 0, len(ks) - 1)
    matched = ks[order][pos] == kc            # token con solape cuyo lugar existe sin solape
    sin_idx = order[pos]
    assert np.allclose(s["p"][sin_idx[matched]], c["p"][matched], atol=1e-6), \
        f"{con_path}: p distinto para el mismo lugar (grillas desalineadas)"

    # distancia por lugar: sin solape vs. promedio con solape
    loc_pos = np.clip(np.searchsorted(ks[order], loc_keys), 0, len(ks) - 1)
    loc_matched = ks[order][loc_pos] == loc_keys
    loc_sin = order[loc_pos][loc_matched]

    # por ocurrencia: |d_con - d_sin| relativo, y desvio respecto del promedio del lugar
    d_sin_occ = s["dist"][sin_idx[matched]].astype(np.float64)
    d_con_occ = c["dist"][matched].astype(np.float64)
    rel_vs_sin = np.abs(d_con_occ - d_sin_occ) / ((d_con_occ + d_sin_occ) / 2 + 1e-12)
    # ocurrencia cuyo chip con solape es identico a un chip sin solape (misma entrada)
    same_chip = np.zeros(len(kc), dtype=bool)
    same_chip[matched] = rel_vs_sin < 1e-5
    loc_mean_occ = loc_mean[loc_inv]
    multi = loc_count[loc_inv] >= 2
    rel_vs_loc = np.abs(c["dist"] - loc_mean_occ) / (loc_mean_occ + 1e-12)

    # |e1 - e2|: similitud coseno y razon de normas, por ocurrencia
    rs, rc = delta_rows(s), delta_rows(c)
    both = matched & (rc >= 0)
    both &= rs[np.where(matched, sin_idx, 0)] >= 0
    occ = np.flatnonzero(both)
    ds_all, dc_all = s["delta"], c["delta"]
    cos, ratio = np.empty(len(occ)), np.empty(len(occ))
    for i in range(0, len(occ), CHUNK):
        o = occ[i:i + CHUNK]
        a = ds_all[rs[sin_idx[o]]].astype(np.float32).reshape(len(o), -1)
        b = dc_all[rc[o]].astype(np.float32).reshape(len(o), -1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cos[i:i + CHUNK] = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
        ratio[i:i + CHUNK] = nb / np.maximum(na, 1e-12)

    return {
        "region": str(s["region"]),
        "n_loc_sin": len(ks),
        "n_tok_con": len(kc),
        "n_loc_con": len(loc_keys),
        "n_loc_nuevos": int((~loc_matched).sum()),
        "loc_dist_sin": s["dist"][loc_sin].astype(np.float64),
        "loc_dist_con": loc_mean[loc_matched],
        "loc_count_con": loc_count[loc_matched],
        "loc_label": loc_label[loc_matched],
        "dedup_dist": loc_mean, "dedup_label": loc_label,
        "rel_vs_sin": rel_vs_sin, "border_con": border_fraction(c)[matched],
        "border_sin": border_fraction(s)[sin_idx[matched]],
        "rel_vs_loc": rel_vs_loc[multi], "border_multi": border_fraction(c)[multi],
        "delta_cos": cos, "delta_ratio": ratio, "delta_label": c["label"][occ],
        "delta_border": border_fraction(c)[occ],
        "same_chip": same_chip[matched], "delta_same_chip": same_chip[occ],
    }


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def read_por_par(path: Path) -> dict:
    return {(r["model"], r["region"]): r for r in csv.DictReader(open(path))}


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def mean_std(v) -> tuple[float, float, int]:
    v = np.array([x for x in v if not np.isnan(x)])
    return (float(v.mean()) if len(v) else float("nan"),
            float(v.std(ddof=1)) if len(v) > 1 else float("nan"), int(len(v)))


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    med = np.array([np.median(y[idx == i]) if (idx == i).any() else np.nan for i in range(len(edges) - 1)])
    return (edges[:-1] + edges[1:]) / 2, med


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

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


def clean(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="#e7e6e1", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_auroc(plt, metric_rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, key, title in ((axes[0], "auroc_con", "Con solape, todos los tokens"),
                           (axes[1], "auroc_con_por_lugar", "Con solape, promediado por lugar")):
        ax.plot([0.3, 1.0], [0.3, 1.0], color="#7a7974", lw=1, ls=(0, (3, 3)), zorder=1)
        for m in MODELS:
            pts = [(r["auroc_sin"], r[key], r["region"]) for r in metric_rows
                   if r["model"] == m and not np.isnan(r["auroc_sin"]) and not np.isnan(r[key])]
            if not pts:
                continue
            x, y, _ = zip(*pts)
            ax.scatter(x, y, s=34, color=COLORS[m], marker=MARKERS[m], edgecolor="white",
                       linewidth=1.1, zorder=3, label=f"{NAMES[m]} ({len(pts)} pares)")
            for xi, yi, reg in pts:
                if abs(yi - xi) >= 0.03:
                    ax.annotate(reg, (xi, yi), xytext=(4, -9), textcoords="offset points",
                                fontsize=7, color="#52514e")
        ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.01); ax.set_aspect("equal")
        ax.set_xlabel("AUROC sin solape"); ax.set_ylabel("AUROC con 50 % de solape")
        ax.set_title(title, loc="left")
        clean(ax)
    axes[0].legend(loc="lower right", fontsize=7.8)
    fig.text(0.01, -0.02, "Línea punteada: mismo valor en ambas corridas. Se rotulan los pares que se "
             "alejan 0.03 o más.", fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_distancias(plt, data, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, m in zip(axes, MODELS):
        x, y = data[m]["loc_dist_sin"], data[m]["loc_dist_con"]
        hi = float(np.quantile(np.concatenate([x, y]), 0.995))
        hb = ax.hexbin(x, y, gridsize=55, extent=(0, hi, 0, hi), bins="log", mincnt=1,
                       cmap="Blues", linewidths=0)
        ax.plot([0, hi], [0, hi], color="#1b1b1a", lw=0.8, ls=(0, (3, 3)))
        rho = spearmanr(x, y).statistic
        ax.text(0.04, 0.95, f"Spearman ρ = {rho:.3f}\n{len(x):,} lugares".replace(",", " "),
                transform=ax.transAxes, va="top", fontsize=7.8, color="#1b1b1a")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_title(NAMES[m], loc="left")
        ax.set_xlabel("Distancia coseno sin solape")
        if m == "clay":
            ax.set_ylabel("Distancia con solape (media del lugar)")
        clean(ax)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("lugares (escala log)", fontsize=7); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_delta(plt, data, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, m in zip(axes, MODELS):
        cos, lab = data[m]["delta_cos"], data[m]["delta_label"]
        for label, color, name in ((LABEL_NC, "#9a9993", "NC"), (LABEL_C, COLORS[m], "C")):
            v = cos[lab == label]
            if len(v):
                ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step",
                        lw=1.8, color=color, label=f"{name} (mediana {np.median(v):.2f}, n={len(v):,})".replace(",", " "))
        ax.set_title(NAMES[m], loc="left"); ax.set_xlabel("Similitud coseno de |e1 − e2| (sin vs. con solape)")
        ax.legend(loc="upper left", fontsize=7.3)
        clean(ax, "y")
    axes[0].set_ylabel("Fracción de ocurrencias")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_posicion(plt, data, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    edges = np.linspace(0, 1, 7)
    for m in MODELS:
        d = data[m]
        o, od = ~d["same_chip"], ~d["delta_same_chip"]
        n = int(o.sum())
        x, y = binned(d["border_con"][o], d["rel_vs_sin"][o], edges)
        axes[0].plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2,
                     label=f"{NAMES[m]} ({n:,} ocurrencias)".replace(",", " "))
        x, y = binned(d["delta_border"][od], d["delta_cos"][od], edges)
        axes[1].plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2, label=NAMES[m])
    axes[0].set_ylabel("Diferencia relativa de distancia (mediana)")
    axes[0].set_title("Distancia coseno: sin vs. con solape", loc="left")
    axes[1].set_ylabel("Similitud coseno de |e1 − e2| (mediana)")
    axes[1].set_title("Vector de diferencia: sin vs. con solape", loc="left")
    for ax in axes:
        ax.set_xlabel("Posición del token en el chip con solape (0 = borde, 1 = centro)")
        ax.set_xlim(0, 1)
        clean(ax)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7.8)
    fig.text(0.01, -0.03, "Solo ocurrencias cuyo chip con solape difiere del chip sin solape (las "
             "idénticas dan diferencia 0). Prithvi: solo beirut aporta chips distintos.",
             fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_mapa(plt, sin_dir: Path, con_dir: Path, region: str, out):
    """Mapa de la escena a resolucion de token: p de la mascara, distancia sin y con solape."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for row, m in enumerate(("clay", "satmae")):
        s, c = np.load(sin_dir / m / f"{region}.npz"), np.load(con_dir / m / f"{region}.npz")
        tok = int(s["token_size_m"]) // 10
        r0 = min(s["center_row_10m"].min(), c["center_row_10m"].min())
        c0 = min(s["center_col_10m"].min(), c["center_col_10m"].min())

        def cells(d):
            """Fila y columna de cada token en una grilla comun a ambas corridas."""
            return (((d["center_row_10m"] - r0) / tok).astype(int),
                    ((d["center_col_10m"] - c0) / tok).astype(int))

        ri_s, ci_s = cells(s)
        ri_c, ci_c = cells(c)
        H = max(ri_s.max(), ri_c.max()) + 1
        W = max(ci_s.max(), ci_c.max()) + 1
        p = np.full((H, W), np.nan); p[ri_c, ci_c] = c["p"]; p[ri_s, ci_s] = s["p"]
        ds = np.full((H, W), np.nan); ds[ri_s, ci_s] = s["dist"]
        acc = np.zeros((H, W)); cnt = np.zeros((H, W))
        np.add.at(acc, (ri_c, ci_c), c["dist"]); np.add.at(cnt, (ri_c, ci_c), 1)
        dc = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        vmax = float(np.nanquantile(np.concatenate([s["dist"], c["dist"]]), 0.99))
        panels = ((p, "Máscara OSCD (fracción p)", "Greys", 0, 1),
                  (ds, "Distancia sin solape", "magma", 0, vmax),
                  (dc, "Distancia con solape (media del lugar)", "magma", 0, vmax))
        for col, (img, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            ax.set_facecolor("#e7e6e1")
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(f"{NAMES[m]}: {title}", loc="left", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
    fig.text(0.01, -0.01, f"Escena: {region}. Cada celda es un token (80 m). Gris claro: terreno sin "
             "cubrir por ningún chip en esa corrida.", fontsize=7.5, color="#7a7974")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin", type=Path, default=Path("results/comparacion_protocolo"))
    ap.add_argument("--con", type=Path, default=Path("results/comparacion_protocolo_50_solapamiento"))
    ap.add_argument("--out", type=Path, default=Path("results/comparativa_solapamiento"))
    ap.add_argument("--map-region", default="beirut")
    args = ap.parse_args()
    tab, figd = args.out / "tablas", args.out / "figuras"
    tab.mkdir(parents=True, exist_ok=True); figd.mkdir(parents=True, exist_ok=True)

    pp_sin = read_por_par(args.sin / "evaluacion" / "por_par.csv")
    pp_con = read_por_par(args.con / "evaluacion" / "por_par.csv")

    data, metric_rows, emb_rows, scene_rows = {}, [], [], []
    for m in MODELS:
        scenes = json.loads((args.sin / m / "manifest.json").read_text())["scenes"]
        acc = {k: [] for k in ("loc_dist_sin", "loc_dist_con", "loc_count_con", "loc_label", "rel_vs_sin",
                               "border_con", "border_sin", "rel_vs_loc", "border_multi",
                               "delta_cos", "delta_ratio", "delta_label", "delta_border",
                               "same_chip", "delta_same_chip")}
        for region in scenes:
            r = compare_scene(args.sin / m / f"{region}.npz", args.con / m / f"{region}.npz")
            print(f"[{m}] {region:12s} lugares sin={r['n_loc_sin']:6d} con={r['n_loc_con']:6d} "
                  f"(nuevos {r['n_loc_nuevos']:5d}) tokens con={r['n_tok_con']:6d}")
            for k in acc:
                acc[k].append(r[k])

            # metricas por par, incluida la variante "promediado por lugar"
            lab, dd = r["dedup_label"], r["dedup_dist"]
            a_dedup = auroc(dd[lab == LABEL_C], dd[lab == LABEL_NC])
            f_dedup = fisher_index(dd[lab == LABEL_C], dd[lab == LABEL_NC])
            s_row, c_row = pp_sin[(m, region)], pp_con[(m, region)]
            metric_rows.append({
                "model": m, "region": region,
                "n_chips_sin": int(s_row["n_chips"]), "n_chips_con": int(c_row["n_chips"]),
                "n_C_sin": int(s_row["n_C"]), "n_C_con": int(c_row["n_C"]),
                "n_C_con_por_lugar": int((lab == LABEL_C).sum()),
                "auroc_sin": fnum(s_row["auroc"]), "auroc_con": fnum(c_row["auroc"]),
                "auroc_con_por_lugar": a_dedup,
                "delta_auroc": fnum(c_row["auroc"]) - fnum(s_row["auroc"]),
                "fisher_sin": fnum(s_row["fisher"]), "fisher_con": fnum(c_row["fisher"]),
                "fisher_con_por_lugar": f_dedup,
            })
            mask_c, mask_nc = r["delta_label"] == LABEL_C, r["delta_label"] == LABEL_NC
            scene_rows.append({
                "model": m, "region": region,
                "lugares_sin": r["n_loc_sin"], "lugares_con": r["n_loc_con"], "lugares_nuevos": r["n_loc_nuevos"],
                "tokens_con": r["n_tok_con"],
                "spearman_dist": float(spearmanr(r["loc_dist_sin"], r["loc_dist_con"]).statistic),
                "dif_rel_dist_mediana": float(np.median(r["rel_vs_sin"])),
                "cos_delta_e_mediana_C": float(np.median(r["delta_cos"][mask_c])) if mask_c.any() else float("nan"),
                "cos_delta_e_mediana_NC": float(np.median(r["delta_cos"][mask_nc])) if mask_nc.any() else float("nan"),
            })
        data[m] = {k: np.concatenate(v) for k, v in acc.items()}

        d = data[m]
        other = ~d["delta_same_chip"]   # medianas de |e1 - e2| solo con chips distintos
        lc, ln = (d["delta_label"] == LABEL_C) & other, (d["delta_label"] == LABEL_NC) & other
        identical = d["same_chip"]
        emb_rows.append({
            "model": m,
            "lugares_comparados": len(d["loc_dist_sin"]),
            "ocurrencias_con_solape": len(d["rel_vs_sin"]),
            "pct_ocurrencias_mismo_chip": 100 * float(identical.mean()),
            "pearson_dist": float(pearsonr(d["loc_dist_sin"], d["loc_dist_con"]).statistic),
            "spearman_dist": float(spearmanr(d["loc_dist_sin"], d["loc_dist_con"]).statistic),
            "dif_rel_dist_mediana": float(np.median(d["rel_vs_sin"])),
            "dif_rel_dist_p90": float(np.quantile(d["rel_vs_sin"], 0.9)),
            "dif_rel_dist_mediana_otro_chip": float(np.median(d["rel_vs_sin"][~identical])) if (~identical).any() else float("nan"),
            "desvio_rel_entre_chips_mediana": float(np.median(d["rel_vs_loc"])) if len(d["rel_vs_loc"]) else float("nan"),
            "cos_delta_e_mediana_C": float(np.median(d["delta_cos"][lc])) if lc.any() else float("nan"),
            "cos_delta_e_mediana_NC": float(np.median(d["delta_cos"][ln])) if ln.any() else float("nan"),
            "cos_delta_e_p10": float(np.quantile(d["delta_cos"][other], 0.1)) if other.any() else float("nan"),
            "razon_normas_delta_e_mediana": float(np.median(d["delta_ratio"][other])) if other.any() else float("nan"),
        })

    # --- resumen de metricas por modelo (pares validos en las tres variantes)
    summary_rows = []
    for m in MODELS:
        rows = [r for r in metric_rows if r["model"] == m
                and not np.isnan(r["auroc_sin"]) and not np.isnan(r["auroc_con"])]
        out = {"model": m, "pares": len(rows)}
        for k in ("auroc_sin", "auroc_con", "auroc_con_por_lugar", "fisher_sin", "fisher_con", "fisher_con_por_lugar"):
            mu, sd, _ = mean_std([r[k] for r in rows])
            out[f"{k}_media"], out[f"{k}_std"] = mu, sd
        deltas = np.array([r["delta_auroc"] for r in rows])
        out["delta_auroc_medio"] = float(deltas.mean()) if len(deltas) else float("nan")
        out["delta_auroc_abs_max"] = float(np.abs(deltas).max()) if len(deltas) else float("nan")
        out["wilcoxon_p_auroc"] = (float(wilcoxon([r["auroc_con"] for r in rows], [r["auroc_sin"] for r in rows]).pvalue)
                                   if len(rows) >= 6 else float("nan"))
        same_order = None
        if len(rows) >= 3:
            same_order = float(spearmanr([r["auroc_sin"] for r in rows], [r["auroc_con"] for r in rows]).statistic)
        out["spearman_orden_pares"] = same_order if same_order is not None else float("nan")
        summary_rows.append(out)

    write_csv(tab / "metricas_por_par.csv", metric_rows)
    write_csv(tab / "resumen_metricas.csv", summary_rows)
    write_csv(tab / "embeddings_por_modelo.csv", emb_rows)
    write_csv(tab / "embeddings_por_escena.csv", scene_rows)

    # --- posicion en el chip (tabla de la figura 4)
    edges = np.linspace(0, 1, 7)
    pos_rows = []
    for m in MODELS:
        o, od = ~data[m]["same_chip"], ~data[m]["delta_same_chip"]
        x, y1 = binned(data[m]["border_con"][o], data[m]["rel_vs_sin"][o], edges)
        _, y2 = binned(data[m]["delta_border"][od], data[m]["delta_cos"][od], edges)
        for xi, a, b in zip(x, y1, y2):
            pos_rows.append({"model": m, "posicion_en_chip": round(float(xi), 3),
                             "dif_rel_dist_mediana": a, "cos_delta_e_mediana": b})
    write_csv(tab / "efecto_posicion_en_chip.csv", pos_rows)

    plt = setup_mpl()
    fig_auroc(plt, metric_rows, figd / "fig1_auroc_sin_vs_con.png")
    fig_distancias(plt, data, figd / "fig2_distancia_por_lugar.png")
    fig_delta(plt, data, figd / "fig3_similitud_delta_e.png")
    fig_posicion(plt, data, figd / "fig4_efecto_posicion.png")
    fig_mapa(plt, args.sin, args.con, args.map_region, figd / f"fig5_mapa_{args.map_region}.png")

    (args.out / "resumen.json").write_text(json.dumps(
        {"sin_solape": str(args.sin), "con_solape": str(args.con),
         "metricas": summary_rows, "embeddings": emb_rows}, indent=2))
    write_readme(args, summary_rows, emb_rows, metric_rows)
    print(f"\nListo: {args.out}")


def write_readme(args, summary_rows, emb_rows, metric_rows):
    S = {r["model"]: r for r in summary_rows}
    E = {r["model"]: r for r in emb_rows}

    def f(x, n=3):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"

    lines = [
        "# Comparativa: sin solape vs. 50 % de solape",
        "",
        f"Fuentes: `{args.sin}` (sin solape) y `{args.con}` (50 % de solape). "
        "Generado por `compare_overlap.py`; no edites este archivo a mano.",
        "",
        "## 1. Métricas por par (sección G)",
        "",
        "| Modelo | Pares | AUROC sin | AUROC con | AUROC con, por lugar | Δ AUROC medio | máx \\|Δ\\| | Wilcoxon p | Fisher sin | Fisher con |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        r = S[m]
        lines.append(
            f"| {NAMES[m]} | {r['pares']} | {f(r['auroc_sin_media'])} ± {f(r['auroc_sin_std'])} | "
            f"{f(r['auroc_con_media'])} ± {f(r['auroc_con_std'])} | {f(r['auroc_con_por_lugar_media'])} | "
            f"{f(r['delta_auroc_medio'], 4)} | {f(r['delta_auroc_abs_max'])} | {f(r['wilcoxon_p_auroc'], 2)} | "
            f"{f(r['fisher_sin_media'], 2)} | {f(r['fisher_con_media'], 2)} |")
    lines += [
        "",
        "«Con, por lugar»: con solape, la distancia de cada lugar se promedia entre los chips que lo cubren "
        "y el lugar cuenta una vez. Wilcoxon: prueba pareada sobre el AUROC por par (solo con ≥ 6 pares).",
        "",
        "## 2. Embeddings, token a token",
        "",
        "| Modelo | Lugares | Ocurrencias con solape | % mismo chip | Spearman ρ distancia | Dif. relativa mediana (otro chip) | cos(\\|e1−e2\\|) mediana C | cos(\\|e1−e2\\|) mediana NC | Razón de normas mediana |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        r = E[m]
        lines.append(
            f"| {NAMES[m]} | {r['lugares_comparados']:,} | {r['ocurrencias_con_solape']:,} | "
            f"{f(r['pct_ocurrencias_mismo_chip'], 1)} % | {f(r['spearman_dist'])} | "
            f"{f(100 * r['dif_rel_dist_mediana_otro_chip'], 1)} % | {f(r['cos_delta_e_mediana_C'])} | "
            f"{f(r['cos_delta_e_mediana_NC'])} | {f(r['razon_normas_delta_e_mediana'])} |".replace(",", " "))
    lines += [
        "",
        "- **% mismo chip**: ocurrencias cuyo chip con solape coincide exactamente con un chip sin solape "
        "(misma entrada, distancia idéntica). Las demás columnas excluyen esos casos y miden solo el efecto "
        "del contexto. Diferencia relativa: |d_con − d_sin| / promedio de ambas.",
        "- **cos(|e1−e2|)**: similitud coseno entre el vector de diferencia del mismo lugar en ambas corridas "
        "(1 = misma dirección). Solo tokens C y NC (los únicos guardados).",
        "",
        "## Archivos",
        "",
        "- `tablas/metricas_por_par.csv`: AUROC y Fisher por modelo y par en las tres variantes.",
        "- `tablas/resumen_metricas.csv`: resumen por modelo de la tabla anterior.",
        "- `tablas/embeddings_por_modelo.csv` y `embeddings_por_escena.csv`: comparación token a token.",
        "- `tablas/efecto_posicion_en_chip.csv`: diferencias según la posición del token en el chip.",
        "- `figuras/fig1_auroc_sin_vs_con.png`: AUROC por par, sin vs. con solape.",
        "- `figuras/fig2_distancia_por_lugar.png`: distancia coseno de cada lugar en ambas corridas.",
        "- `figuras/fig3_similitud_delta_e.png`: similitud de |e1 − e2| entre corridas, C vs. NC.",
        "- `figuras/fig4_efecto_posicion.png`: cómo cambian las diferencias del borde al centro del chip.",
        f"- `figuras/fig5_mapa_{args.map_region}.png`: mapa de la escena (máscara, distancia sin y con solape).",
    ]
    (args.out / "LEEME.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
