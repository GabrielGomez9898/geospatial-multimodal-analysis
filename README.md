# geospatial-multimodal-analysis
Diseño y evaluación de una arquitectura multimodal basada en modelos fundacionales especializados para la detección y descripción de cambios en imágenes satelitales multiespectrales

# Pipeline de extracción y comparación de embeddings — Clay v1.5 / SatMAE / Prithvi EO 2.0

Implementa las secciones D, E y F del protocolo de tu asesora sobre OSCD, más la
métrica de separabilidad zero-shot que definimos en la discusión conceptual.

## Estructura

```
config.py              Ficha técnica de los 3 modelos (bandas, GSD, patch size, etc.)
preprocessing.py        Selección de bandas, remuestreo de GSD, recorte en chips, normalización
metrics.py               Distancia coseno, fracción p y etiquetas C/NC/mixtas, AUROC, Fisher
models/
  clay_extractor.py      Carga y forward pass de Clay v1.5
  satmae_extractor.py    Carga y forward pass de SatMAE (fMoW-Sentinel, group_c)
  prithvi_extractor.py   Carga y forward pass de Prithvi EO 2.0 (300M)
run_pipeline.py          Extracción por chip + guardado por escena (.npz); prueba con un par
run_all_{clay,satmae,prithvi}.py   Extracción sobre los 14 pares de OSCD, un modelo por script
evaluate.py              Sección G del protocolo (G.1, G.3, G.4) a partir de los .npz
build_report.py          Informe PDF de la comparación a partir de la salida de evaluate.py
```

## Flujo (sección G del protocolo)

1. **Extracción** (una vez por modelo, el orden no importa):
   ```bash
   python run_all_clay.py --oscd-root model_assets/clay/data/oscd \
       --clay-checkpoint model_assets/clay/clay-v1.5.ckpt --clay-metadata model_assets/clay/metadata.yaml
   python run_all_satmae.py --oscd-root model_assets/clay/data/oscd
   python run_all_prithvi.py --oscd-root model_assets/clay/data/oscd
   ```
   Para cada chip se calculan e1 y e2, se guarda por token la distancia
   coseno, la fracción de cambio `p`, la etiqueta C/NC/mixta, su posición en
   la escena y `|e1 − e2|` en float16 (solo tokens C y NC, para el UMAP), y
   se descartan los embeddings. Resultado:
   `results/comparacion_protocolo/<modelo>/<region>.npz` + `manifest.json`.
2. **Evaluación** (sin cargar modelos): `python evaluate.py` →
   `results/comparacion_protocolo/evaluacion/por_par.csv` y `resumen.json`.
3. **Informe** (opcional): `python build_report.py` →
   `results/comparacion_protocolo/informe_comparacion_protocolo.pdf`. Las
   tablas y la figura se regeneran desde los resultados; los párrafos de
   configuración, limitaciones y decisiones son texto fijo y hay que
   actualizarlos si cambia la configuración.

**Variante con 50 % de solape** (para comparar contra la corrida sin solape):
```bash
O=results/comparacion_protocolo_50_solapamiento
python run_all_clay.py ... --chip-overlap 0.5 --out-dir $O     # mismos argumentos que arriba
python run_all_satmae.py --oscd-root model_assets/clay/data/oscd --chip-overlap 0.5 --out-dir $O
python run_all_prithvi.py --oscd-root model_assets/clay/data/oscd --chip-overlap 0.5 --out-dir $O
python evaluate.py --results-dir $O
```
Con solape, la grilla de chips (paso = medio chip) se centra en la escena y se
desplaza lo necesario para que sus tokens caigan sobre la misma grilla del
terreno que la corrida sin solape: el mismo `center_row_10m`/`center_col_10m`
identifica la misma región en ambas corridas. Un mismo lugar aparece en varios
chips (con distinto contexto), así que en el AUROC por par cuenta varias veces.
`build_report.py` describe la corrida sin solape; no usarlo sobre esta carpeta.

## Qué falta que conectar (marcado como `TODO` en el código)

Este pipeline define correctamente la interfaz, los tamaños esperados y el
manejo de metadatos/GSD según la documentación oficial de cada modelo, pero
**no puede instalar ni ejecutar los tres repositorios de terceros desde este
entorno** (sin acceso de red a GitHub/HuggingFace de terceros). Antes de correr
`run_pipeline.py` se necesita:

1. **Clay**: `pip install git+https://github.com/Clay-foundation/model.git`,
   descargar `clay-v1.5.ckpt`, y confirmar en `models/clay_extractor.py` el
   nombre exacto de la clase (`ClayMAEModule` en la doc actual) y el formato
   del "datacube" de entrada contra el notebook oficial
   `docs/tutorials/embeddings.ipynb`.
2. **SatMAE**: `git clone https://github.com/sustainlab-group/SatMAE`, descargar
   el checkpoint ViT-Large `group_c` de fMoW-Sentinel, y completar
   `load_satmae_model()` en `models/satmae_extractor.py` con la clase real del
   repo (`models_vit_group_channels.py` o equivalente en tu commit).
3. **Prithvi**: `pip install terratorch`, y confirmar en
   `models/prithvi_extractor.py` la firma exacta de `forward_features()` y si
   el backbone incluye o no un token CLS en tu versión instalada.
4. **Lector de OSCD**: implementar `load_oscd_pair()` en `run_pipeline.py` con
   `rasterio` o `tifffile`, devolviendo las 13 bandas alineadas a 10 m y la
   máscara binaria.
5. **Normalización de Clay/SatMAE**: completar los `TODO` de
   `preprocessing.py` (`prepare_for_clay`, `prepare_for_satmae`) con las
   estadísticas oficiales de cada uno (`configs/metadata.yaml` de Clay;
   estadísticas de fMoW-Sentinel del repo de SatMAE). Los de Prithvi ya están
   cargados en `config.py` (vienen del `config.json` oficial).

## Decisiones metodológicas ya incorporadas (de nuestra discusión)

- **Prithvi se remuestrea de 10 m a 30 m** por average pooling 3×3 y se recorta
  a 192×192 px (múltiplo de 16), en vez de forzar 224×224 con padding.
- **Chips sin solape, dos modos de corrida.** La huella de cada chip en terreno
  es distinta (Clay 256 px × 10 m = 2560 m, SatMAE 96 px × 10 m = 960 m,
  Prithvi 224 px × 30 m = 6720 m). La grilla de ventanas se arma con la huella
  más grande del conjunto `--align-with`, centrada en la escena y sin solape;
  cada modelo recorta su chip centrado en cada ventana.
  - **Escena completa (por defecto):** cada modelo usa su propia grilla, así que
    el número de chips por escena varía entre modelos. Salida en
    `results/comparacion_protocolo/` (la corrida anterior, con AUPRC y umbral
    0.15 por chip, quedó en `results/sin_solape_escena_completa/`).
  - **Alineada (`--align-with clay satmae prithvi`):** ventana = huella de
    Prithvi (672 px a 10 m); los tres modelos tienen el mismo número de chips
    por escena y cada chip mira el mismo centro. Las escenas donde no cabe
    ninguna ventana se excluyen para todos. Pasar `--out results/sin_solape/...`.

  Cada token del .npz lleva `center_row_10m` y `center_col_10m` (centro de su
  región en la grilla de 10 m de la escena) para cruzar modelos.
- **La máscara de cambio se remuestrea a la grilla nativa de cada modelo**
  (no al revés) — se evita interpolar vectores de 1024 dimensiones.
- **Distancia coseno** como métrica principal (invariante a norma).
- **Etiquetas de tokens (G.1):** C si `p` > 0.50, NC si `p` < 0.05, mixtas
  (0.05 ≤ `p` ≤ 0.50) excluidas.
- **AUROC (G.3) e índice de Fisher (G.4) por par de imágenes**, juntando los
  tokens de todos los chips de la escena; media y std (ddof=1) entre pares.
  La comparación principal usa los pares válidos para todos los modelos
  (con escena completa, las 5 escenas donde cabe Prithvi); la complementaria,
  todos los pares válidos de cada modelo.
- SatMAE devuelve 3 mapas (uno por grupo espectral); se combinan con
  `combine_satmae_groups()` (media por defecto, máximo como alternativa).
- Prithvi funde t1/t2 dentro del encoder (atención cruzada) — la distancia
  que se calcula sobre sus dos bloques temporales no es directamente
  comparable en interpretación a la de Clay/SatMAE, aunque el cálculo
  numérico es el mismo. Documentar esto en la sección de resultados.

