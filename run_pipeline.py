"""
run_pipeline.py

Orquesta, para UN par de imagenes OSCD (t1, t2 + mascara de cambio), el flujo
completo para los tres modelos:

    leer imagen OSCD (13 bandas, alineadas a 10 m)
        -> preprocesar segun el modelo (bandas, GSD, chips)
        -> extraer patch tokens de t1 y t2
        -> distancia coseno token a token
        -> remuestrear mascara a la grilla del modelo
        -> AUPRC + tamano de efecto

Este script asume que ya implementaste los TODO marcados en models/*.py contra
las APIs reales de cada repo. No descarga checkpoints ni instala dependencias.

Uso:
    python run_pipeline.py --oscd-pair-dir /ruta/a/par_de_imagen --out results.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import CLAY_V15, SATMAE_FMOW_SENTINEL, PRITHVI_EO2_300M, PRITHVI_MEAN, PRITHVI_STD
from preprocessing import (
    resample_20m_bands_to_10m,
    prepare_for_clay,
    prepare_for_satmae,
    prepare_prithvi_pairs,
    prithvi_scene_size_30m,
)
from metrics import (
    cosine_distance_map,
    combine_satmae_groups,
    resample_mask_to_grid,
    separability_metrics,
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


# Longitudes de onda centrales (micrometros) de las 10 bandas que usa Clay,
# en el mismo orden que CLAY_V15.bands. Son valores estandar publicados por
# ESA para Sentinel-2 -- de todas formas, confirma contra metadata.yaml del
# repo de Clay antes de un resultado final, por si su version difiere.
CLAY_WAVELENGTHS_UM = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


def run_for_clay(img_t1, img_t2, mask, model, meta_t1: dict, meta_t2: dict,
                  device: str = "cpu", max_chips: int | None = None, stride: int | None = None):
    """
    meta_t1, meta_t2: salidas de get_scene_metadata() para cada fecha
    (contienen lat/lon reales y week_hour real de esa fecha).
    max_chips: si se pasa, solo procesa los primeros N chips (util para
    probar rapido antes de correr la escena completa).
    stride: paso entre chips en pixeles (ver preprocessing.prepare_for_clay).
    """
    from models.clay_extractor import extract_clay_patch_tokens

    chips_t1 = prepare_for_clay(img_t1, CLAY_V15, stride=stride)
    chips_t2 = prepare_for_clay(img_t2, CLAY_V15, stride=stride)

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
        block = CLAY_V15.chip_size // grid_h  # pixeles de mascara por token
        mask_chip = mask[chip1.row_off:chip1.row_off + CLAY_V15.chip_size,
                          chip1.col_off:chip1.col_off + CLAY_V15.chip_size]
        mask_grid = resample_mask_to_grid(mask_chip, grid_h, grid_w)

        results.append(separability_metrics(dist_map, mask_grid))
    return results


def run_for_satmae(img_t1, img_t2, mask, model, device: str = "cpu",
                    max_chips: int | None = None, stride: int | None = None,
                    combine_mode: str = "max"):
    """
    combine_mode="max" por defecto (no "mean"): ver discusion del protocolo --
    promediar los 3 grupos espectrales puede diluir un cambio que solo es
    visible en uno de ellos.
    """
    from models.satmae_extractor import extract_satmae_patch_tokens

    chips_t1 = prepare_for_satmae(img_t1, SATMAE_FMOW_SENTINEL, stride=stride)
    chips_t2 = prepare_for_satmae(img_t2, SATMAE_FMOW_SENTINEL, stride=stride)

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
        mask_grid = resample_mask_to_grid(mask_chip, grid_h, grid_w)

        results.append(separability_metrics(dist_map, mask_grid))
    return results


def run_for_prithvi(img_t1, img_t2, mask, model, device: str = "cpu",
                     max_chips: int | None = None, stride: int | None = None):
    """
    A diferencia de Clay/SatMAE, Prithvi procesa t1 y t2 JUNTOS en una sola
    pasada (patch embedding 3D con T=2) -- ver prithvi_extractor.py. Por eso
    aqui se llama extract_prithvi_patch_tokens una vez por PAR de chips.

    La mascara viene ya remuestreada a 30 m y alineada con los chips (ver
    preprocessing.prepare_prithvi_pairs). Si la escena es mas chica que el chip
    a 30 m, devuelve una lista vacia: la escena queda fuera de la evaluacion.
    """
    from models.prithvi_extractor import extract_prithvi_patch_tokens

    pairs = prepare_prithvi_pairs(img_t1, img_t2, mask, PRITHVI_EO2_300M,
                                   PRITHVI_MEAN, PRITHVI_STD, stride=stride)
    if max_chips is not None:
        pairs = pairs[:max_chips]

    results = []
    for p in pairs:
        joint = extract_prithvi_patch_tokens(model, p["chip_t1"], p["chip_t2"], device=device)  # (2, 14, 14, 1024)
        dist_map = cosine_distance_map(joint[0], joint[1])

        grid_h, grid_w = dist_map.shape
        mask_grid = resample_mask_to_grid(p["mask_chip"], grid_h, grid_w)

        results.append(separability_metrics(dist_map, mask_grid))
    return results


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
                         help="Por defecto: results/<model>/<model>_results.json")
    parser.add_argument("--max-chips", type=int, default=None,
                         help="Limitar a los primeros N chips de la escena (prueba rapida)")
    parser.add_argument("--chip-overlap", type=float, default=0.0,
                         help="Fraccion de solape entre chips (0.0 = sin solape, 0.5 = 50%% de solape). "
                              "Util para escenas mas chicas que el chip_size del modelo.")
    args = parser.parse_args()
    if args.model == "clay" and (args.clay_checkpoint is None or args.clay_metadata is None):
        parser.error("--clay-checkpoint y --clay-metadata son requeridos con --model clay")
    if args.out is None:
        args.out = Path(f"results/{args.model}/{args.model}_results.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
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

    if args.model == "clay":
        meta_t1 = get_scene_metadata(region, date_t1)
        meta_t2 = get_scene_metadata(region, date_t2)

        from models.clay_extractor import load_clay_model
        clay_model = load_clay_model(
            checkpoint_path=str(args.clay_checkpoint),
            metadata_path=str(args.clay_metadata),
            device=device,
        )
        clay_stride = int(CLAY_V15.chip_size * (1 - args.chip_overlap)) if args.chip_overlap > 0 else None
        results = run_for_clay(img_t1, img_t2, mask, clay_model, meta_t1, meta_t2,
                                device=device, max_chips=args.max_chips, stride=clay_stride)
    elif args.model == "satmae":
        from models.satmae_extractor import load_satmae_model
        satmae_model = load_satmae_model(device=device)
        satmae_stride = int(SATMAE_FMOW_SENTINEL.chip_size * (1 - args.chip_overlap)) if args.chip_overlap > 0 else None
        results = run_for_satmae(img_t1, img_t2, mask, satmae_model,
                                  device=device, max_chips=args.max_chips, stride=satmae_stride,
                                  combine_mode=args.satmae_combine_mode)
    else:  # prithvi
        h30, w30 = prithvi_scene_size_30m(img_t1.shape[1], img_t1.shape[2])
        chip = PRITHVI_EO2_300M.chip_size
        print(f"Escena a 30 m: {h30}x{w30} px  (chip de Prithvi: {chip}x{chip})")
        if h30 < chip or w30 < chip:
            print("Esta escena NO alcanza el tamano minimo de Prithvi a 30 m: queda fuera "
                  "de la evaluacion (sin padding ni reescalado). Elige otro --pair-index.")
            results = []
        else:
            from models.prithvi_extractor import load_prithvi_model
            prithvi_model = load_prithvi_model(device=device)
            prithvi_stride = int(chip * (1 - args.chip_overlap)) if args.chip_overlap > 0 else None
            results = run_for_prithvi(img_t1, img_t2, mask, prithvi_model,
                                       device=device, max_chips=args.max_chips, stride=prithvi_stride)

    args.out.write_text(json.dumps({args.model: results}, indent=2))
    print(f"Resultados guardados en {args.out}")
    print(f"Chips procesados: {len(results)}")


if __name__ == "__main__":
    main()