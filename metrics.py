"""
metrics.py

Implementa lo que acordamos en la discusion conceptual:
  - distancia coseno entre t1 y t2, token a token
  - remuestreo de la MASCARA de cambio (no de los embeddings) a la grilla nativa
    de cada modelo
  - separabilidad zero-shot: AUPRC + tamano de efecto (AUC de Mann-Whitney),
    en vez de una clasificacion con umbral entrenado
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Distancia entre representaciones
# ---------------------------------------------------------------------------

def cosine_distance_map(emb_t1: np.ndarray, emb_t2: np.ndarray) -> np.ndarray:
    """
    emb_t1, emb_t2: (H, W, D) -- misma grilla espacial, mismo modelo, dos fechas.
    Devuelve (H, W): 1 - similitud_coseno en cada posicion.
    """
    assert emb_t1.shape == emb_t2.shape
    num = np.sum(emb_t1 * emb_t2, axis=-1)
    denom = np.linalg.norm(emb_t1, axis=-1) * np.linalg.norm(emb_t2, axis=-1)
    denom = np.clip(denom, a_min=1e-8, a_max=None)
    cos_sim = num / denom
    return 1.0 - cos_sim


def combine_satmae_groups(distance_per_group: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    SatMAE entrega 3 mapas de distancia (uno por grupo espectral, mismo grid
    espacial 12x12). Los combina en un unico mapa (12, 12).

    distance_per_group: (n_groups, 12, 12)
    mode: "mean" (default) o "max" (mas sensible a cambios visibles en un solo
          grupo espectral, a costa de mas falsos positivos).
    """
    if mode == "mean":
        return distance_per_group.mean(axis=0)
    elif mode == "max":
        return distance_per_group.max(axis=0)
    raise ValueError(f"mode desconocido: {mode}")


# ---------------------------------------------------------------------------
# Remuestreo de la mascara de cambio a la grilla nativa de cada modelo
# ---------------------------------------------------------------------------

def resample_mask_to_grid(
    mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    mask: (H_px, W_px) binaria (0/1) a resolucion de pixel original de OSCD,
          YA RECORTADA al mismo extent espacial que el chip que se paso al modelo
          (fundamental: si hiciste center-crop o remuestreo de GSD en preprocessing.py,
          debes aplicar EXACTAMENTE la misma transformacion geometrica a la mascara
          antes de llamar a esta funcion).

    Devuelve (grid_h, grid_w) binaria: un token se marca como "cambio" si la
    fraccion de pixeles de cambio dentro de su area cubierta supera `threshold`.
    """
    h_px, w_px = mask.shape
    assert h_px % grid_h == 0 and w_px % grid_w == 0, (
        "La mascara debe ser exactamente divisible por el grid del modelo; "
        "revisa que se le aplico el mismo crop/resample que a la imagen."
    )
    block_h, block_w = h_px // grid_h, w_px // grid_w
    reshaped = mask.reshape(grid_h, block_h, grid_w, block_w)
    fraction_changed = reshaped.mean(axis=(1, 3))
    return (fraction_changed > threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# Separabilidad zero-shot
# ---------------------------------------------------------------------------

def separability_metrics(distance_map: np.ndarray, mask_grid: np.ndarray) -> dict:
    """
    distance_map, mask_grid: (grid_h, grid_w), ya alineados uno a uno.
    Devuelve AUPRC y el tamano de efecto no parametrico (AUC de Mann-Whitney,
    equivalente al "Probability of Superiority": la probabilidad de que un
    token de cambio elegido al azar tenga mayor distancia que uno sin cambio).
    """
    y_true = mask_grid.flatten()
    scores = distance_map.flatten()

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {
            "auprc": float("nan"),
            "effect_size_auc": float("nan"),
            "n_change_tokens": int(y_true.sum()),
            "n_total_tokens": int(len(y_true)),
            "note": "No hay ambas clases presentes en este chip; metricas no definidas.",
        }

    auprc = average_precision_score(y_true, scores)

    changed_scores = scores[y_true == 1]
    unchanged_scores = scores[y_true == 0]
    u_stat, _ = mannwhitneyu(changed_scores, unchanged_scores, alternative="greater")
    effect_size_auc = u_stat / (len(changed_scores) * len(unchanged_scores))

    return {
        "auprc": float(auprc),
        "effect_size_auc": float(effect_size_auc),
        "n_change_tokens": int(y_true.sum()),
        "n_total_tokens": int(len(y_true)),
    }
