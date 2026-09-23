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
import rasterio
from rasterio.warp import transform as rio_transform

from config import CLAY_V15, SATMAE_FMOW_SENTINEL, PRITHVI_EO2_300M, PRITHVI_MEAN, PRITHVI_STD
from preprocessing import (
    resample_20m_bands_to_10m,
    prepare_for_clay,
    prepare_for_satmae,
    prepare_for_prithvi,
)
from metrics import (
    cosine_distance_map,
    combine_satmae_groups,
    resample_mask_to_grid,
    separability_metrics,
)


def load_oscd_pair(pair_dir: Path):
    """
    TODO: PENDIENTE -- falta confirmar la estructura exacta de carpetas de tu
    descarga de OSCD (nombres de archivo de bandas, formato de la mascara,
    donde vienen las fechas de adquisicion) antes de terminar esta funcion.
    Debe devolver:
        img_t1, img_t2: (13, H, W) float32, reflectancia, alineadas a 10 m
        mask: (H, W) int, 0/1
        ref_tif_t1, ref_tif_t2: rutas a un .tif de referencia de cada fecha,
            usadas por get_scene_metadata() para leer lat/lon reales.
    """
    raise NotImplementedError(
        "Conecta aqui tu lector de OSCD. Dile a Claude como esta organizada tu "
        "carpeta de OSCD y te completa esta funcion."
    )


def get_scene_metadata(ref_tif_path: Path, acquisition_date) -> dict:
    """
    Extrae lat/lon reales (centroide de la escena) desde un GeoTIFF, sin
    depender de como esta organizada tu carpeta de OSCD -- solo necesita la
    ruta a UN archivo .tif de esa fecha (cualquier banda sirve, todas
    comparten el mismo grid espacial).

    acquisition_date: objeto datetime.date con la fecha real de adquisicion
    de esa imagen (la necesitas para el campo "time" de Clay: semana del ano
    y hora del dia). Sentinel-2 no trae hora exacta de captura en el nombre
    de archivo tipico de OSCD, asi que si no la tienes, usa una hora fija
    razonable (p. ej. la hora de paso del satelite, ~10:30 local) y documenta
    el supuesto.
    """
    with rasterio.open(ref_tif_path) as src:
        cx_px, cy_px = src.width / 2, src.height / 2
        cx, cy = src.xy(cy_px, cx_px)  # coordenadas en el CRS nativo del raster
        lon, lat = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
        lon, lat = lon[0], lat[0]

    week = acquisition_date.isocalendar()[1]  # semana ISO del ano (1-53)
    hour = 10  # supuesto documentado: hora aprox. de paso de Sentinel-2

    return {"lat": lat, "lon": lon, "week_hour": (week, hour)}


# Longitudes de onda centrales (micrometros) de las 10 bandas que usa Clay,
# en el mismo orden que CLAY_V15.bands. Son valores estandar publicados por
# ESA para Sentinel-2 -- de todas formas, confirma contra metadata.yaml del
# repo de Clay antes de un resultado final, por si su version difiere.
CLAY_WAVELENGTHS_UM = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


def run_for_clay(img_t1, img_t2, mask, model, meta_t1: dict, meta_t2: dict):
    """
    meta_t1, meta_t2: salidas de get_scene_metadata() para cada fecha
    (contienen lat/lon reales y week_hour real de esa fecha).
    """
    from models.clay_extractor import extract_clay_patch_tokens

    chips_t1 = prepare_for_clay(img_t1, CLAY_V15)
    chips_t2 = prepare_for_clay(img_t2, CLAY_V15)

    results = []
    for chip1, chip2 in zip(chips_t1, chips_t2):
        emb1 = extract_clay_patch_tokens(model, [chip1], gsd_m=10.0,
                                          wavelengths_um=CLAY_WAVELENGTHS_UM,
                                          lat=meta_t1["lat"], lon=meta_t1["lon"],
                                          week_hour=meta_t1["week_hour"])[0]
        emb2 = extract_clay_patch_tokens(model, [chip2], gsd_m=10.0,
                                          wavelengths_um=CLAY_WAVELENGTHS_UM,
                                          lat=meta_t2["lat"], lon=meta_t2["lon"],
                                          week_hour=meta_t2["week_hour"])[0]
        dist_map = cosine_distance_map(emb1, emb2)

        grid_h, grid_w = dist_map.shape
        block = CLAY_V15.chip_size // grid_h  # pixeles de mascara por token
        mask_chip = mask[chip1.row_off:chip1.row_off + CLAY_V15.chip_size,
                          chip1.col_off:chip1.col_off + CLAY_V15.chip_size]
        mask_grid = resample_mask_to_grid(mask_chip, grid_h, grid_w)

        results.append(separability_metrics(dist_map, mask_grid))
    return results


def run_for_satmae(img_t1, img_t2, mask, model, combine_mode="mean"):
    from models.satmae_extractor import extract_satmae_patch_tokens

    chips_t1 = prepare_for_satmae(img_t1, SATMAE_FMOW_SENTINEL)
    chips_t2 = prepare_for_satmae(img_t2, SATMAE_FMOW_SENTINEL)

    results = []
    for chip1, chip2 in zip(chips_t1, chips_t2):
        emb1 = extract_satmae_patch_tokens(model, [chip1])[0]  # (3, 12, 12, 1024)
        emb2 = extract_satmae_patch_tokens(model, [chip2])[0]

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


def run_for_prithvi(img_t1, img_t2, mask, model):
    from models.prithvi_extractor import extract_prithvi_patch_tokens

    chips_t1 = prepare_for_prithvi(img_t1, PRITHVI_EO2_300M, PRITHVI_MEAN, PRITHVI_STD)
    chips_t2 = prepare_for_prithvi(img_t2, PRITHVI_EO2_300M, PRITHVI_MEAN, PRITHVI_STD)
    # Con una escena OSCD tipica, esto normalmente da 1 solo chip (ver discusion).

    results = []
    for chip1, chip2 in zip(chips_t1, chips_t2):
        joint = extract_prithvi_patch_tokens(model, chip1, chip2)  # (2, 12, 12, 1024)
        emb1, emb2 = joint[0], joint[1]
        dist_map = cosine_distance_map(emb1, emb2)

        grid_h, grid_w = dist_map.shape
        mask_chip = mask[chip1.row_off:chip1.row_off + PRITHVI_EO2_300M.chip_size,
                          chip1.col_off:chip1.col_off + PRITHVI_EO2_300M.chip_size]
        mask_grid = resample_mask_to_grid(mask_chip, grid_h, grid_w)

        results.append(separability_metrics(dist_map, mask_grid))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscd-pair-dir", type=Path, required=True)
    parser.add_argument("--clay-checkpoint", type=Path, required=True)
    parser.add_argument("--clay-metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=Path("clay_results.json"))
    args = parser.parse_args()

    # load_oscd_pair() debe devolver tambien las rutas a un .tif de referencia
    # por fecha (ver docstring de get_scene_metadata) para poder leer lat/lon reales.
    img_t1, img_t2, mask, ref_tif_t1, ref_tif_t2, date_t1, date_t2 = load_oscd_pair(
        args.oscd_pair_dir
    )
    img_t1 = resample_20m_bands_to_10m(img_t1)
    img_t2 = resample_20m_bands_to_10m(img_t2)

    meta_t1 = get_scene_metadata(ref_tif_t1, date_t1)
    meta_t2 = get_scene_metadata(ref_tif_t2, date_t2)

    from models.clay_extractor import load_clay_model
    clay_model = load_clay_model(
        checkpoint_path=str(args.clay_checkpoint),
        metadata_path=str(args.clay_metadata),
        device=args.device,
    )

    clay_results = run_for_clay(img_t1, img_t2, mask, clay_model, meta_t1, meta_t2)

    args.out.write_text(json.dumps({"clay": clay_results}, indent=2))
    print(f"Resultados guardados en {args.out}")
    print(f"Chips procesados: {len(clay_results)}")


if __name__ == "__main__":
    main()