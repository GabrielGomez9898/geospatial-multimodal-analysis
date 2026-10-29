"""
metrics.py

Implementa la seccion G del protocolo:
  - distancia coseno entre t1 y t2, token a token (G.2)
  - fraccion de cambio p por token, calculada sobre la MASCARA llevada a la
    grilla nativa de cada modelo (no se remuestrean los embeddings), y
    etiquetas C / NC / mixtas (G.1)
  - AUROC (G.3) e indice de Fisher (G.4)
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


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
# Seccion G.1 del protocolo: regiones con cambio (C), sin cambio (NC) y mixtas
# ---------------------------------------------------------------------------

# Umbrales del protocolo (seccion G.1), sobre la fraccion p de pixeles con
# cambio dentro de la region que cubre cada token.
CHANGE_THRESHOLD = 0.50      # C:  p > 0.50
NO_CHANGE_THRESHOLD = 0.05   # NC: p < 0.05
LABEL_C, LABEL_NC, LABEL_MIXED = 1, 0, -1   # mixtas: 0.05 <= p <= 0.50 (excluidas)


def token_change_fraction(mask: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """
    mask: (H_px, W_px) con valores en [0, 1] (binaria a 10 m, o fracciones de
          cambio ya promediadas, como la de Prithvi a 30 m), YA RECORTADA al
          mismo extent espacial que el chip que se paso al modelo.

    Devuelve (grid_h, grid_w): p = pixeles con cambio / pixeles de la region
    que cubre cada token.
    """
    h_px, w_px = mask.shape
    assert h_px % grid_h == 0 and w_px % grid_w == 0, (
        "La mascara debe ser exactamente divisible por el grid del modelo; "
        "revisa que se le aplico el mismo crop/resample que a la imagen."
    )
    block_h, block_w = h_px // grid_h, w_px // grid_w
    reshaped = mask.astype(np.float32).reshape(grid_h, block_h, grid_w, block_w)
    return reshaped.mean(axis=(1, 3))


def label_tokens(p: np.ndarray) -> np.ndarray:
    """Etiqueta cada token como LABEL_C, LABEL_NC o LABEL_MIXED segun G.1."""
    labels = np.full(p.shape, LABEL_MIXED, dtype=np.int8)
    labels[p > CHANGE_THRESHOLD] = LABEL_C
    labels[p < NO_CHANGE_THRESHOLD] = LABEL_NC
    return labels


# ---------------------------------------------------------------------------
# Secciones G.3 y G.4: discriminacion (AUROC) y separabilidad (Fisher)
# ---------------------------------------------------------------------------

def auroc(dist_c: np.ndarray, dist_nc: np.ndarray) -> float:
    """AUROC de la distancia coseno con C como clase positiva. NaN si falta una clase."""
    if len(dist_c) == 0 or len(dist_nc) == 0:
        return float("nan")
    y_true = np.concatenate([np.ones(len(dist_c)), np.zeros(len(dist_nc))])
    return float(roc_auc_score(y_true, np.concatenate([dist_c, dist_nc])))


def fisher_index(dist_c: np.ndarray, dist_nc: np.ndarray) -> float:
    """
    F = (mu_C - mu_NC)^2 / (var_C + var_NC). Varianzas poblacionales (ddof=0).
    NaN si falta una clase o si el denominador es cero (G.4 pide reportar ese
    caso por separado).
    """
    if len(dist_c) == 0 or len(dist_nc) == 0:
        return float("nan")
    denom = dist_c.var() + dist_nc.var()
    if denom == 0:
        return float("nan")
    return float((dist_c.mean() - dist_nc.mean()) ** 2 / denom)
