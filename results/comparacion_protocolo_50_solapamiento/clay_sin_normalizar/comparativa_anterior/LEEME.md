# Comparativa: sin solape vs. 50 % de solape

Fuentes: `results/comparacion_protocolo` (sin solape) y `results/comparacion_protocolo_50_solapamiento` (50 % de solape). Generado por `compare_overlap.py`; no edites este archivo a mano.

## 1. Métricas por par (sección G)

| Modelo | Pares | AUROC sin | AUROC con | AUROC con, por lugar | Δ AUROC medio | máx \|Δ\| | Wilcoxon p | Fisher sin | Fisher con |
|---|---|---|---|---|---|---|---|---|---|
| Clay v1.5 | 14 | 0.811 ± 0.170 | 0.814 ± 0.157 | 0.814 | 0.0030 | 0.050 | 0.67 | 0.81 | 0.72 |
| SatMAE | 14 | 0.843 ± 0.160 | 0.838 ± 0.166 | 0.842 | -0.0044 | 0.031 | 0.22 | 1.72 | 1.80 |
| Prithvi EO 2.0 | 2 | 0.991 ± 0.004 | 0.991 ± 0.004 | 0.991 | 0.0000 | 0.000 | — | 7.55 | 7.55 |

«Con, por lugar»: con solape, la distancia de cada lugar se promedia entre los chips que lo cubren y el lugar cuenta una vez. Wilcoxon: prueba pareada sobre el AUROC por par (solo con ≥ 6 pares).

## 2. Embeddings, token a token

| Modelo | Lugares | Ocurrencias con solape | % mismo chip | Spearman ρ distancia | Dif. relativa mediana (otro chip) | cos(\|e1−e2\|) mediana C | cos(\|e1−e2\|) mediana NC | Razón de normas mediana |
|---|---|---|---|---|---|---|---|---|
| Clay v1.5 | 76 800 | 215 808 | 22.7 % | 1.000 | 0.0 % | 1.000 | 1.000 | 1.000 |
| SatMAE | 91 296 | 323 208 | 16.5 % | 0.998 | 3.3 % | 0.979 | 0.974 | 1.000 |
| Prithvi EO 2.0 | 980 | 1 225 | 64.0 % | 0.999 | 2.9 % | 1.000 | 1.000 | 1.000 |

- **% mismo chip**: ocurrencias cuyo chip con solape coincide exactamente con un chip sin solape (misma entrada, distancia idéntica). La «diferencia relativa (otro chip)» excluye esos casos y mide solo el efecto del contexto: |d_con − d_sin| / promedio de ambas.
- **cos(|e1−e2|)**: similitud coseno entre el vector de diferencia del mismo lugar en ambas corridas (1 = misma dirección). Solo tokens C y NC (los únicos guardados).

## Archivos

- `tablas/metricas_por_par.csv`: AUROC y Fisher por modelo y par en las tres variantes.
- `tablas/resumen_metricas.csv`: resumen por modelo de la tabla anterior.
- `tablas/embeddings_por_modelo.csv` y `embeddings_por_escena.csv`: comparación token a token.
- `tablas/efecto_posicion_en_chip.csv`: diferencias según la posición del token en el chip.
- `figuras/fig1_auroc_sin_vs_con.png`: AUROC por par, sin vs. con solape.
- `figuras/fig2_distancia_por_lugar.png`: distancia coseno de cada lugar en ambas corridas.
- `figuras/fig3_similitud_delta_e.png`: similitud de |e1 − e2| entre corridas, C vs. NC.
- `figuras/fig4_efecto_posicion.png`: cómo cambian las diferencias del borde al centro del chip.
- `figuras/fig5_mapa_beirut.png`: mapa de la escena (máscara, distancia sin y con solape).
