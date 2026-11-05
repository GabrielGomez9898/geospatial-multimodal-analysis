"""
run_pipeline.py

Orquesta, para UN par de imagenes OSCD (t1, t2 + mascara de cambio), el flujo
completo para los tres modelos:

    leer imagen OSCD (13 bandas, alineadas a 10 m)
        -> preprocesar segun el modelo (bandas, GSD, chips)
        -> extraer patch tokens de t1 y t2
        -> distancia coseno token a token (G.2) y |e1 - e2| (para UMAP, G.5)
        -> fraccion de cambio p por token y etiqueta C / NC / mixta (G.1)
        -> descartar e1 y e2 y guardar todo lo anterior en un .npz por escena

Las metricas (AUROC, Fisher, G.3-G.4) NO se calculan aqui sino en
evaluate.py, a partir de los .npz -- asi se pueden recalcular o graficar sin
volver a correr los modelos.

Uso (un solo par, prueba rapida):
    python run_pipeline.py --model satmae --oscd-root model_assets/clay/data/oscd --pair-index 3
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from config import (
    ALL_MODEL_SPECS, CLAY_V15, SATMAE_FMOW_SENTINEL, PRITHVI_EO2_300M, PRITHVI_MEAN, PRITHVI_STD,
)
from preprocessing import (
    resample_20m_bands_to_10m,
    prepare_for_clay,
    prepare_for_satmae,
    prepare_prithvi_pairs,
    aligned_chip_offsets,
    common_windows,
    window_stride,
    token_px_10m,
    footprint_px_10m,
)
from metrics import (
    cosine_distance_map,
    combine_satmae_groups,
    token_change_fraction,
    label_tokens,
    LABEL_C,
    LABEL_NC,
    LABEL_MIXED,
    CHANGE_THRESHOLD,
    NO_CHANGE_THRESHOLD,
)


def load_oscd_pair(dataset, index: int):
    """
    dataset: instancia de torchgeo.datasets.OSCD(root=..., split="train", download=True)
    index: indice del par dentro del dataset (0 a 13, ya que "train" tiene los
           14 pares con mascara publica).

    Devuelve tambien `region` (el nombre de la ciudad, p.ej. "paris") en vez de
    rutas a .tif de referencia -- los .tif de OSCD no traen georreferenciacion
    real (ver get_scene_metadata), asi que usamos el nombre de la ciudad para
    ubicar la escena.
    """
    import numpy as np

    sample = dataset[index]
    image = sample["image"].numpy()  # (2, 13, H, W)
    mask = sample["mask"].numpy()[0]  # (H, W), ya en 0/1

    img_t1, img_t2 = image[0].astype(np.float32), image[1].astype(np.float32)

    files_meta = dataset.files[index]
    region = files_meta["region"]
    date_t1_str, date_t2_str = files_meta["dates"]

    return img_t1, img_t2, mask, region, date_t1_str, date_t2_str


def parse_oscd_date(date_str: str):
    """
    dates.txt no tiene un formato unico documentado entre versiones del dataset.
    Intenta los formatos mas comunes; si ninguno calza, imprime el valor crudo
    para que lo ajustes a mano (es un dato de una sola linea, no vale la pena
    una libreria de parseo de fechas para esto).
    """
    from datetime import datetime

    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"No pude parsear la fecha '{date_str}' con los formatos conocidos. "
        f"Imprime dataset.files[0]['dates'] y dime que formato exacto trae."
    )


def get_scene_metadata(region: str, acquisition_date) -> dict:
    """
    Los .tif de OSCD (via torchgeo) NO traen georreferenciacion real (torchgeo
    los expone como NonGeoDataset a proposito, solo para benchmarking de ML).
    Por eso usamos una tabla de coordenadas aproximadas por ciudad en vez de
    leer el CRS del archivo -- cada region de OSCD es una ciudad real conocida
    (ver Daudt et al. 2018 / rcdaudt.github.io/oscd). La precision es a nivel
    de ciudad, no de pixel, lo cual es razonable para el rol que cumple lat/lon
    en Clay (contexto geografico aproximado, no georreferenciacion exacta).
    """
    region_coords = {
        "abudhabi": (24.4539, 54.3773),
        "aguasclaras": (-15.7168, -48.0247),
        "beihai": (21.4733, 109.1011),
        "beirut": (33.8938, 35.5018),
        "bercy": (48.8384, 2.3781),
        "bordeaux": (44.8378, -0.5792),
        "cupertino": (37.3230, -122.0322),
        "mumbai": (19.0760, 72.8777),
        "nantes": (47.2184, -1.5536),
        "paris": (48.8566, 2.3522),
        "pisa": (43.7228, 10.4017),
        "rennes": (48.1173, -1.6778),
        "hongkong": (22.3193, 114.1694),
        "saclay_e": (48.7167, 2.1667),
    }
    if region not in region_coords:
        raise KeyError(
            f"Region '{region}' no esta en la tabla de coordenadas. "
            f"Regiones conocidas: {list(region_coords.keys())}"
        )
    lat, lon = region_coords[region]

    week = acquisition_date.isocalendar()[1]  # semana ISO del ano (1-53)
    hour = 10  # supuesto documentado: hora aprox. de paso de Sentinel-2

    return {"lat": lat, "lon": lon, "week_hour": (week, hour)}


def add_align_args(parser: argparse.ArgumentParser) -> None:
    """Argumento comun a todos los scripts: con que modelos se alinean las ventanas."""
    parser.add_argument("--align-with", nargs="+", choices=list(ALL_MODEL_SPECS),
                         default=[],
                         help="Modelos cuyas huellas definen la grilla comun de ventanas (sin solape). "
                              "La ventana mide lo que la huella mas grande del conjunto; todos los "
                              "modelos evaluan las mismas ventanas, con su chip centrado en cada una. "
                              "Por defecto, solo el modelo que se corre: grilla propia sobre toda la "
                              "escena. Para la comparacion alineada: --align-with clay satmae prithvi.")
    parser.add_argument("--chip-overlap", type=float, default=0.0,
                         help="Fraccion de solape entre ventanas vecinas (0 = sin solape, 0.5 = 50%%). "
                              "Con solape, el mismo terreno aparece en varios chips; usar un --out-dir "
                              "distinto, p. ej. results/comparacion_protocolo_50_solapamiento.")


def align_specs_from_names(names: list[str], model_name: str) -> list:
    """Specs de --align-with; el modelo que se esta corriendo siempre se incluye."""
    names = list(dict.fromkeys(names + [model_name]))
    return [ALL_MODEL_SPECS[n] for n in names]


# Longitudes de onda centrales (micrometros) de las 10 bandas que usa Clay,
# en el mismo orden que CLAY_V15.bands. Son valores estandar publicados por
# ESA para Sentinel-2 -- de todas formas, confirma contra metadata.yaml del
# repo de Clay antes de un resultado final, por si su version difiere.
CLAY_WAVELENGTHS_UM = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


def token_record(row_off: int, col_off: int, token_px: int, dist: np.ndarray,
                 p: np.ndarray, emb1: np.ndarray, emb2: np.ndarray, **extra) -> dict:
    """
    Resume UN chip a nivel de token, para poder descartar los embeddings.

    row_off, col_off: esquina del chip en la grilla de 10 m de la escena.
    token_px: lado de la region que cubre cada token, en pixeles de 10 m.
    dist, p: (gh, gw) -- distancia coseno t1-t2 y fraccion de cambio por token.
    emb1, emb2: (gh, gw, D), o (G, gh, gw, D) en SatMAE (un mapa por grupo espectral).
    extra: arrays adicionales por token con forma (gh, gw, ...) (p. ej. la
           distancia por grupo de SatMAE).

    Solo se guarda |e1 - e2| (float16) de los tokens C y NC: las mixtas se
    excluyen de toda la evaluacion (G.1) y no hacen falta para el UMAP (G.5).
    """
    gh, gw = p.shape
    labels = label_tokens(p).ravel()
    keep = labels != LABEL_MIXED
    rows, cols = np.divmod(np.arange(gh * gw), gw)

    delta = np.abs(emb1 - emb2)
    if delta.ndim == 4:  # SatMAE: (G, gh, gw, D) -> (gh, gw, G, D)
        delta = np.moveaxis(delta, 0, 2)
    delta = delta.reshape(gh * gw, *delta.shape[2:])[keep].astype(np.float16)

    record = {
        "dist": dist.ravel().astype(np.float32),
        "p": p.ravel().astype(np.float32),
        "label": labels,
        "token_row": rows.astype(np.int16),
        "token_col": cols.astype(np.int16),
        # centro de la region del token en la grilla de 10 m de la escena
        "center_row_10m": (row_off + (rows + 0.5) * token_px).astype(np.float32),
        "center_col_10m": (col_off + (cols + 0.5) * token_px).astype(np.float32),
        "delta_keep": keep,
        "delta": delta,
    }
    for key, arr in extra.items():
        record[key] = arr.reshape(gh * gw, *arr.shape[2:]).astype(np.float32)
    return record


def run_for_clay(img_t1, img_t2, mask, model, meta_t1: dict, meta_t2: dict,
                  offsets: list[tuple[int, int]], device: str = "cpu", max_chips: int | None = None):
    """
    meta_t1, meta_t2: salidas de get_scene_metadata() para cada fecha
    (contienen lat/lon reales y week_hour real de esa fecha).
    offsets: esquinas de los chips en la grilla de 10 m (ver
    preprocessing.aligned_chip_offsets).
    max_chips: si se pasa, solo procesa los primeros N chips (util para
    probar rapido antes de correr la escena completa).
    Devuelve un token_record() por chip.
    """
    from models.clay_extractor import extract_clay_patch_tokens

    chips_t1 = prepare_for_clay(img_t1, CLAY_V15, offsets)
    chips_t2 = prepare_for_clay(img_t2, CLAY_V15, offsets)

    if max_chips is not None:
        chips_t1 = chips_t1[:max_chips]
        chips_t2 = chips_t2[:max_chips]

    results = []
    for chip1, chip2 in zip(chips_t1, chips_t2):
        emb1 = extract_clay_patch_tokens(model, [chip1], gsd_m=10.0,
                                          wavelengths_um=CLAY_WAVELENGTHS_UM,
                                          lat=meta_t1["lat"], lon=meta_t1["lon"],
                                          week_hour=meta_t1["week_hour"], device=device)[0]
        emb2 = extract_clay_patch_tokens(model, [chip2], gsd_m=10.0,
                                          wavelengths_um=CLAY_WAVELENGTHS_UM,
                                          lat=meta_t2["lat"], lon=meta_t2["lon"],
                                          week_hour=meta_t2["week_hour"], device=device)[0]
        dist_map = cosine_distance_map(emb1, emb2)

        grid_h, grid_w = dist_map.shape
        mask_chip = mask[chip1.row_off:chip1.row_off + CLAY_V15.chip_size,
                          chip1.col_off:chip1.col_off + CLAY_V15.chip_size]
        p = token_change_fraction(mask_chip, grid_h, grid_w)

        results.append(token_record(chip1.row_off, chip1.col_off, token_px_10m(CLAY_V15),
                                    dist_map, p, emb1, emb2))
    return results


def run_for_satmae(img_t1, img_t2, mask, model, offsets: list[tuple[int, int]],
                    device: str = "cpu", max_chips: int | None = None,
                    combine_mode: str = "max"):
    """
    combine_mode="max" por defecto (no "mean"): ver discusion del protocolo --
    promediar los 3 grupos espectrales puede diluir un cambio que solo es
    visible en uno de ellos. La distancia por grupo tambien se guarda
    ("dist_groups"), por si se quiere recombinar despues.
    """
    from models.satmae_extractor import extract_satmae_patch_tokens

    chips_t1 = prepare_for_satmae(img_t1, SATMAE_FMOW_SENTINEL, offsets)
    chips_t2 = prepare_for_satmae(img_t2, SATMAE_FMOW_SENTINEL, offsets)

    if max_chips is not None:
        chips_t1 = chips_t1[:max_chips]
        chips_t2 = chips_t2[:max_chips]

    results = []
    for chip1, chip2 in zip(chips_t1, chips_t2):
        emb1 = extract_satmae_patch_tokens(model, [chip1], device=device)[0]  # (3, 12, 12, 1024)
        emb2 = extract_satmae_patch_tokens(model, [chip2], device=device)[0]

        dist_per_group = np.stack([
            cosine_distance_map(emb1[g], emb2[g]) for g in range(emb1.shape[0])
        ])  # (3, 12, 12)
        dist_map = combine_satmae_groups(dist_per_group, mode=combine_mode)

        grid_h, grid_w = dist_map.shape
        mask_chip = mask[chip1.row_off:chip1.row_off + SATMAE_FMOW_SENTINEL.chip_size,
                          chip1.col_off:chip1.col_off + SATMAE_FMOW_SENTINEL.chip_size]
        p = token_change_fraction(mask_chip, grid_h, grid_w)

        results.append(token_record(chip1.row_off, chip1.col_off,
                                    token_px_10m(SATMAE_FMOW_SENTINEL),
                                    dist_map, p, emb1, emb2,
                                    dist_groups=np.moveaxis(dist_per_group, 0, -1)))  # (12, 12, 3)
    return results


def run_for_prithvi(img_t1, img_t2, mask, model, offsets: list[tuple[int, int]],
                     device: str = "cpu", max_chips: int | None = None):
    """
    A diferencia de Clay/SatMAE, Prithvi procesa t1 y t2 JUNTOS en una sola
    pasada (patch embedding 3D con T=2) -- ver prithvi_extractor.py. Por eso
    aqui se llama extract_prithvi_patch_tokens una vez por PAR de chips.

    La mascara viene ya remuestreada a 30 m y alineada con los chips (ver
    preprocessing.prepare_prithvi_pairs). offsets: esquinas de las ventanas de
    672 px en la grilla de 10 m (ver preprocessing.aligned_chip_offsets).
    """
    from models.prithvi_extractor import extract_prithvi_patch_tokens

    pairs = prepare_prithvi_pairs(img_t1, img_t2, mask, PRITHVI_EO2_300M,
                                   PRITHVI_MEAN, PRITHVI_STD, offsets)
    if max_chips is not None:
        pairs = pairs[:max_chips]

    results = []
    for pr in pairs:
        joint = extract_prithvi_patch_tokens(model, pr["chip_t1"], pr["chip_t2"], device=device)  # (2, 14, 14, 1024)
        dist_map = cosine_distance_map(joint[0], joint[1])

        grid_h, grid_w = dist_map.shape
        p = token_change_fraction(pr["mask_chip"], grid_h, grid_w)

        results.append(token_record(pr["chip_t1"].row_off, pr["chip_t1"].col_off,
                                    token_px_10m(PRITHVI_EO2_300M),
                                    dist_map, p, joint[0], joint[1]))
    return results


# ---------------------------------------------------------------------------
# Guardado por escena y loop sobre todo OSCD (comun a los tres run_all_*.py)
# ---------------------------------------------------------------------------

def label_counts(labels: np.ndarray) -> dict:
    n = len(labels)
    n_mixed = int((labels == LABEL_MIXED).sum())
    return {
        "n_tokens": n,
        "n_C": int((labels == LABEL_C).sum()),
        "n_NC": int((labels == LABEL_NC).sum()),
        "n_mixed_excluded": n_mixed,
        "pct_excluded": 100.0 * n_mixed / n if n else float("nan"),
    }


def save_scene_tokens(path: Path, records: list[dict], chip_offsets: list[tuple[int, int]],
                      spec, region: str) -> dict:
    """
    Concatena los token_record() de todos los chips de una escena y los guarda
    en un .npz. Arrays por token (todos con la misma longitud N): dist, p,
    label, chip_idx, token_row, token_col, center_row_10m, center_col_10m (y
    dist_groups en SatMAE). delta (N_CNC, ...) trae |e1 - e2| solo de los
    tokens C/NC, y delta_token_idx dice a que posicion de los arrays por
    token corresponde cada fila.
    Devuelve los conteos C / NC / mixtas de la escena.
    """
    arrays = {key: np.concatenate([r[key] for r in records]) for key in records[0] if key != "delta_keep"}
    keep = np.concatenate([r["delta_keep"] for r in records])
    arrays["delta_token_idx"] = np.flatnonzero(keep).astype(np.int32)
    arrays["chip_idx"] = np.concatenate([np.full(len(r["dist"]), i, dtype=np.int16)
                                         for i, r in enumerate(records)])
    arrays["chip_offsets_10m"] = np.array(chip_offsets[:len(records)], dtype=np.int32).reshape(-1, 2)
    grid = int(round(np.sqrt(len(records[0]["dist"]))))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, region=region, model=spec.name, grid_size=grid,
             token_size_m=token_px_10m(spec) * 10, chip_footprint_m=footprint_px_10m(spec) * 10,
             **arrays)
    return label_counts(arrays["label"])


SceneRunner = Callable[..., list]


def extract_all(model_name: str, oscd_root: Path, run_scene: SceneRunner,
                align_names: list[str], out_dir: Path, extra_config: dict | None = None,
                overlap: float = 0.0) -> None:
    """
    Loop sobre los 14 pares de OSCD (split "train") para un modelo: extrae,
    guarda <out_dir>/<model_name>/<region>.npz por escena y un manifest.json
    con la configuracion, los conteos C/NC/mixtas y las escenas excluidas.

    run_scene(img_t1, img_t2, mask, region, date_t1, date_t2, offsets) debe
    devolver la lista de token_record() de la escena (ver run_for_*).
    """
    from torchgeo.datasets import OSCD
    oscd = OSCD(root=str(oscd_root), split="train", download=True)
    print(f"Pares en el dataset: {len(oscd)}")

    spec = ALL_MODEL_SPECS[model_name]
    align_specs = align_specs_from_names(align_names, model_name)
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ventanas alineadas con: {[s.name for s in align_specs]} | solape: {overlap:.0%}")
    print(f"Guardando en: {model_dir}\n")

    scenes, excluded_pairs, failed_pairs = {}, [], []
    for index in range(len(oscd)):
        try:
            img_t1, img_t2, mask, region, date_t1_str, date_t2_str = load_oscd_pair(oscd, index)
            img_t1 = resample_20m_bands_to_10m(img_t1)
            img_t2 = resample_20m_bands_to_10m(img_t2)
            h, w = img_t1.shape[1], img_t1.shape[2]
            print(f"[{index+1}/{len(oscd)}] {region:15s} tamano={h}x{w}", end="  ")

            window, _ = common_windows(h, w, align_specs, overlap)
            offsets = aligned_chip_offsets(h, w, spec, align_specs, overlap)
            if not offsets:
                print(f"-> EXCLUIDA (no cabe una ventana de {window}x{window} px a 10 m)")
                excluded_pairs.append({
                    "region": region,
                    "size_10m": [int(h), int(w)],
                    "window_px_10m": window,
                    "reason": "La escena es mas chica que la ventana (huella mas grande de "
                              "--align-with); no sale ningun chip sin padding ni reescalado.",
                })
                continue

            date_t1, date_t2 = parse_oscd_date(date_t1_str), parse_oscd_date(date_t2_str)
            records = run_scene(img_t1, img_t2, mask, region, date_t1, date_t2, offsets)
            counts = save_scene_tokens(model_dir / f"{region}.npz", records, offsets, spec, region)
            scenes[region] = {"n_chips": len(records), **counts}
            print(f"-> {len(records)} chips, {counts['n_tokens']} tokens "
                  f"(C={counts['n_C']}, NC={counts['n_NC']}, mixtas={counts['n_mixed_excluded']})")

        except Exception as e:
            print(f"FALLO en el par {index} ({e.__class__.__name__}: {e})")
            failed_pairs.append({"index": index, "error": str(e)})

    grid_h = spec.chip_size // spec.patch_size
    window = max(footprint_px_10m(s) for s in align_specs)
    manifest = {
        "model": spec.name,
        "align_with": [s.name for s in align_specs],
        "window_px_10m": window,
        "chip_overlap": overlap,
        "stride_px_10m": window_stride(window, overlap),
        "chip_footprint_m": footprint_px_10m(spec) * 10,
        "token_grid": [grid_h, grid_h],
        "token_size_m": token_px_10m(spec) * 10,
        "thresholds": {"C": f"p > {CHANGE_THRESHOLD}", "NC": f"p < {NO_CHANGE_THRESHOLD}",
                       "mixed_excluded": f"{NO_CHANGE_THRESHOLD} <= p <= {CHANGE_THRESHOLD}"},
        **(extra_config or {}),
        "scenes": scenes,
        "excluded_pairs": excluded_pairs,
        "failed_pairs": failed_pairs,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== {spec.name} ===")
    print(f"Escenas extraidas: {len(scenes)} / {len(oscd)}  "
          f"(excluidas por tamano: {len(excluded_pairs)}, fallidas: {len(failed_pairs)})")
    print(f"Manifest: {model_dir / 'manifest.json'}")
    print("Siguiente paso: python evaluate.py")


def pick_device() -> str:
    """Detecta el mejor dispositivo disponible: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["clay", "satmae", "prithvi"], default="clay")
    parser.add_argument("--oscd-root", type=Path, required=True,
                         help="Carpeta donde torchgeo descargara/ya descargo OSCD")
    parser.add_argument("--pair-index", type=int, default=0,
                         help="Indice del par dentro del split 'train' (0-13)")
    parser.add_argument("--clay-checkpoint", type=Path, default=None,
                         help="Requerido si --model clay")
    parser.add_argument("--clay-metadata", type=Path, default=None,
                         help="Requerido si --model clay")
    parser.add_argument("--satmae-combine-mode", type=str, choices=["mean", "max"], default="max")
    parser.add_argument("--device", type=str, default=None,
                         help="cuda, mps o cpu. Si no se especifica, se detecta automaticamente.")
    parser.add_argument("--out", type=Path, default=None,
                         help="Por defecto: results/prueba/<model>/<region>.npz")
    parser.add_argument("--max-chips", type=int, default=None,
                         help="Limitar a los primeros N chips de la escena (prueba rapida)")
    add_align_args(parser)
    args = parser.parse_args()
    if args.model == "clay" and (args.clay_checkpoint is None or args.clay_metadata is None):
        parser.error("--clay-checkpoint y --clay-metadata son requeridos con --model clay")
    device = args.device or pick_device()
    print(f"Usando device: {device}")

    from torchgeo.datasets import OSCD
    oscd = OSCD(root=str(args.oscd_root), split="train", download=True)

    img_t1, img_t2, mask, region, date_t1_str, date_t2_str = load_oscd_pair(
        oscd, args.pair_index
    )
    img_t1 = resample_20m_bands_to_10m(img_t1)
    img_t2 = resample_20m_bands_to_10m(img_t2)
    print(f"Tamano de la escena (bandas, H, W): {img_t1.shape}")
    date_t1, date_t2 = parse_oscd_date(date_t1_str), parse_oscd_date(date_t2_str)
    print(f"Region: {region}  |  Fechas: t1={date_t1}  t2={date_t2}")

    spec = ALL_MODEL_SPECS[args.model]
    align_specs = align_specs_from_names(args.align_with, args.model)
    h, w = img_t1.shape[1], img_t1.shape[2]
    window, _ = common_windows(h, w, align_specs, args.chip_overlap)
    offsets = aligned_chip_offsets(h, w, spec, align_specs, args.chip_overlap)
    print(f"Ventana comun: {window} px a 10 m ({window * 10} m) -> {len(offsets)} ventanas")

    if not offsets:
        print("La escena no alcanza para una sola ventana comun: queda fuera de la evaluacion "
              "para todos los modelos. Elige otro --pair-index o cambia --align-with.")
        return
    if args.model == "clay":
        meta_t1 = get_scene_metadata(region, date_t1)
        meta_t2 = get_scene_metadata(region, date_t2)

        from models.clay_extractor import load_clay_model
        clay_model = load_clay_model(
            checkpoint_path=str(args.clay_checkpoint),
            metadata_path=str(args.clay_metadata),
            device=device,
        )
        results = run_for_clay(img_t1, img_t2, mask, clay_model, meta_t1, meta_t2, offsets,
                                device=device, max_chips=args.max_chips)
    elif args.model == "satmae":
        from models.satmae_extractor import load_satmae_model
        satmae_model = load_satmae_model(device=device)
        results = run_for_satmae(img_t1, img_t2, mask, satmae_model, offsets,
                                  device=device, max_chips=args.max_chips,
                                  combine_mode=args.satmae_combine_mode)
    else:  # prithvi
        from models.prithvi_extractor import load_prithvi_model
        prithvi_model = load_prithvi_model(device=device)
        results = run_for_prithvi(img_t1, img_t2, mask, prithvi_model, offsets,
                                   device=device, max_chips=args.max_chips)

    out = args.out or Path(f"results/prueba/{args.model}/{region}.npz")
    counts = save_scene_tokens(out, results, offsets, spec, region)
    print(f"Tokens guardados en {out}")
    print(f"Chips: {len(results)} | tokens: {counts['n_tokens']} | C={counts['n_C']} "
          f"NC={counts['n_NC']} mixtas={counts['n_mixed_excluded']} ({counts['pct_excluded']:.1f}% excluido)")


if __name__ == "__main__":
    main()