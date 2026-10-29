"""
build_report.py

Genera el informe PDF de la comparacion del protocolo (seccion G) a partir de
los resultados de evaluate.py: evaluacion/resumen.json, evaluacion/por_par.csv
y el manifest.json de cada modelo. No carga modelos ni lee los .npz.

Uso (despues de run_all_*.py y evaluate.py):
    python build_report.py
    python build_report.py --results-dir results/comparacion_protocolo --out informe.pdf

Requiere reportlab y matplotlib. Usa la fuente Arial de macOS
(/System/Library/Fonts/Supplemental/); en otro sistema operativo hay que
cambiar FONT_DIR.

OJO: varios parrafos (configuracion, limitaciones, decisiones pendientes)
son texto fijo que describe la ejecucion actual. Si cambia la configuracion
(p. ej. al normalizar Clay o cambiar bandas), hay que actualizarlos a mano;
las tablas, conteos y la figura si se regeneran desde los resultados.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, default=Path("results/comparacion_protocolo"))
parser.add_argument("--out", type=Path, default=None,
                    help="Por defecto: <results-dir>/informe_comparacion_protocolo.pdf")
args = parser.parse_args()
RES = args.results_dir
OUT = args.out or RES / "informe_comparacion_protocolo.pdf"
FIG = RES / "evaluacion" / "fig_auroc_por_par.png"

FONT_DIR = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial", FONT_DIR + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_DIR + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FONT_DIR + "Arial Italic.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

summary = json.loads((RES / "evaluacion/resumen.json").read_text())
rows = list(csv.DictReader(open(RES / "evaluacion/por_par.csv")))
manifests = {m: json.loads((RES / m / "manifest.json").read_text()) for m in ("clay", "satmae", "prithvi")}
PM = summary["per_model"]
MODELS = ["clay", "satmae", "prithvi"]
NAMES = {"clay": "Clay v1.5", "satmae": "SatMAE", "prithvi": "Prithvi EO 2.0"}
SCENES = list(manifests["clay"]["scenes"])

by = {(r["model"], r["region"]): r for r in rows}

# ---------------------------------------------------------------- estilos
INK, INK2, MUTED, RULE, HEAD = (colors.HexColor(c) for c in ("#1b1b1a", "#52514e", "#7a7974", "#d9d8d3", "#f1f0ec"))
ss = getSampleStyleSheet()
base = ParagraphStyle("base", parent=ss["Normal"], fontName="Arial", fontSize=9.6, leading=13.6,
                      textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=base, fontSize=8.2, leading=11, textColor=INK2)
cell = ParagraphStyle("cell", parent=base, fontSize=8.2, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-Bold")
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=13, leading=17,
                    spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=10.5, leading=14,
                    spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=base, fontName="Arial-Bold", fontSize=18, leading=23, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=base, fontSize=10.5, leading=14, textColor=INK2, spaceAfter=2)
bullet = ParagraphStyle("bullet", parent=base, leftIndent=12, bulletIndent=2, spaceAfter=3)
code = ParagraphStyle("code", parent=base, fontName="Courier", fontSize=8, leading=10.5,
                      backColor=HEAD, borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=8)
note = ParagraphStyle("note", parent=base, backColor=colors.HexColor("#fbf3e4"),
                      borderColor=colors.HexColor("#e2c48a"), borderWidth=0.6, borderPadding=7,
                      spaceBefore=6, spaceAfter=10, leftIndent=6, rightIndent=6)


def P(text, style=base):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(data, widths, header_rows=1):
    data = [[c if not isinstance(c, str) else Paragraph(c, cellb if i < header_rows else cell)
             for c in row] for i, row in enumerate(data)]
    t = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Arial", 8.2),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK2),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def f3(x):
    x = float(x)
    return "—" if x != x else f"{x:.3f}"


def f2(x):
    x = float(x)
    if x != x:
        return "—"
    return f"{x:.3f}" if abs(x) < 0.01 else f"{x:.2f}"


def ms(d, fmt=f3):
    return f"{fmt(d['mean'])} ± {fmt(d['std'])}"


# ---------------------------------------------------------------- figura
font_manager.fontManager.addfont(FONT_DIR + "Arial.ttf")
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#1b1b1a"})
SERIES = {"clay": ("#2a78d6", "o"), "satmae": ("#eb6834", "s"), "prithvi": ("#1baf7a", "D")}
order = sorted(SCENES, key=lambda s: -float(by[("satmae", s)]["auroc"]))
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=220)
offsets = {"clay": -0.2, "satmae": 0.0, "prithvi": 0.2}
for i, _ in enumerate(order):
    ax.axhspan(i - 0.5, i + 0.5, color="#f6f5f2" if i % 2 == 0 else "white", zorder=0, lw=0)
ax.axvline(0.5, color="#7a7974", lw=1, ls=(0, (3, 3)), zorder=1)
ax.text(0.505, -0.55, "azar (0.5)", color="#7a7974", fontsize=7.5, va="bottom")
for m in MODELS:
    xs, ys = [], []
    for i, s in enumerate(order):
        r = by.get((m, s))
        if r and r["valid_auroc"] == "True":
            xs.append(float(r["auroc"])); ys.append(i + offsets[m])
    c, mk = SERIES[m]
    ax.scatter(xs, ys, s=34, color=c, marker=mk, edgecolor="white", linewidth=1.2, zorder=3,
               label=f"{NAMES[m]} ({len(xs)} pares)")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlim(0.3, 1.02)
ax.set_xlabel("AUROC por par (distancia coseno, C como clase positiva)")
ax.grid(axis="x", color="#e7e6e1", lw=0.6, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, fontsize=7.8,
          handletextpad=0.3, columnspacing=1.6, borderaxespad=0)
fig.tight_layout()
fig.savefig(FIG)
plt.close(fig)

# ---------------------------------------------------------------- contenido
story = []
story += [
    P("Comparación de representaciones para detección de cambios", title),
    P("Ejecución de la sección G del protocolo sobre OSCD: Clay v1.5, SatMAE y Prithvi EO 2.0", subtitle),
    P("Gabriel Gómez · Maestría MISIS · Repositorio <i>geospatial-multimodal-analysis</i> · "
      "Resultados en <font face='Courier'>results/comparacion_protocolo/</font>", small),
    Spacer(1, 8),
]

common = summary["common_valid_pairs"]
story += [
    P("Resumen", h1),
    P("Se extrajeron los patch tokens de la última capa del encoder de los tres modelos, congelados, "
      "sobre los 14 pares de OSCD con máscara pública. Para cada token se calculó la distancia coseno "
      "entre T1 y T2 y se etiquetó como cambio (C), sin cambio (NC) o mixto según la fracción de "
      "píxeles con cambio de su región. Con esas distancias se calcularon el AUROC y el índice de Fisher "
      "por par de imágenes (secciones G.1 a G.4)."),
]
story += bullets([
    f"<b>Comparación principal ({len(common)} pares comunes: {', '.join(common)}).</b> "
    f"AUROC: Clay {ms(PM['clay']['principal']['auroc'])}, SatMAE {ms(PM['satmae']['principal']['auroc'])}, "
    f"Prithvi {ms(PM['prithvi']['principal']['auroc'])}. Con dos pares esta comparación no tiene peso "
    "estadístico.",
    f"<b>Clay y SatMAE sobre los 14 pares.</b> AUROC {ms(PM['clay']['complementaria']['auroc'])} y "
    f"{ms(PM['satmae']['complementaria']['auroc'])}; Fisher {ms(PM['clay']['complementaria']['fisher'], f2)} y "
    f"{ms(PM['satmae']['complementaria']['fisher'], f2)}. SatMAE separa algo mejor, con alta variabilidad "
    "entre pares en ambos.",
    f"<b>Prithvi tiene solo {PM['prithvi']['token_counts']['n_C']} tokens C en total.</b> Sus tokens cubren "
    "480 × 480 m y en OSCD casi ningún token de ese tamaño supera el 50 % de cambio. Solo 2 de sus "
    "5 escenas permiten calcular AUROC, y por eso la comparación principal queda en 2 pares.",
    "<b>Clay se evaluó con píxeles sin normalizar</b> (sección 8). Sus resultados deben "
    "considerarse preliminares hasta repetir la extracción con la normalización oficial.",
])

# ---------------------------------------------------------------- 1
story += [
    P("1. Objetivo y alcance", h1),
    P("El protocolo pregunta qué tan diferentes son, en el espacio de representación de cada modelo, las "
      "zonas que cambiaron frente a las que no, en un escenario zero-shot: los modelos no se entrenan y "
      "las etiquetas de OSCD se usan solo para evaluar. Esta ejecución cubre:"),
]
story += bullets([
    "<b>G.1</b> Identificación de regiones con cambio, sin cambio y mixtas.",
    "<b>G.2</b> Distancia coseno entre los embeddings de T1 y T2 en cada posición espacial.",
    "<b>G.3</b> AUROC por par de imágenes y modelo (capacidad de discriminación).",
    "<b>G.4</b> Índice de Fisher por par de imágenes y modelo (separabilidad).",
])
story.append(P("La visualización (G.5: histogramas, mapas de distancias y UMAP) queda pendiente. Los "
               "datos necesarios ya están guardados, de modo que no hace falta volver a ejecutar los modelos."))

# ---------------------------------------------------------------- 2
story += [P("2. Configuración de la ejecución", h1), P("2.1 Modelos y recortes", h2)]
cfg = [["", "Clay v1.5", "SatMAE", "Prithvi EO 2.0"],
       ["Checkpoint", "clay-v1.5.ckpt", "MVRL/satmae-vitlarge-multispec-pretrain (group_c)", "prithvi_eo_v2_300 (terratorch)"],
       ["Bandas usadas", "B02–B08, B8A, B11, B12 (10)", "B02–B08, B8A, B11, B12 (10), en 3 grupos espectrales",
        "B02, B03, B04, B8A, B11, B12 (6)"],
       ["Resolución de entrada", "10 m", "10 m", "30 m (promedio de bloques 3×3)"],
       ["Chip", "256 × 256 px", "96 × 96 px", "224 × 224 px"],
       ["Huella del chip", "2560 m", "960 m", "6720 m"],
       ["Parche / grilla de tokens", "8 px / 32 × 32", "8 px / 12 × 12 (× 3 grupos)", "16 px / 14 × 14 (× 2 fechas)"],
       ["Región por token", "80 m", "80 m", "480 m"],
       ["Metadatos", "GSD, longitudes de onda, lat/lon (por ciudad), semana del año", "No", "No (versión base)"],
       ["Normalización", "Ninguna (pendiente)", "SentinelNormalize oficial (μ ± 2σ → [0, 1])", "z-score oficial (config.json)"],
       ["T1 y T2", "Pasadas separadas", "Pasadas separadas", "Una pasada conjunta (T = 2)"],
       ["Chips / escenas", f"{sum(s['n_chips'] for s in manifests['clay']['scenes'].values())} / 14",
        f"{sum(s['n_chips'] for s in manifests['satmae']['scenes'].values())} / 14",
        f"{sum(s['n_chips'] for s in manifests['prithvi']['scenes'].values())} / 5"]]
story.append(table(cfg, [3.3 * cm, 4.1 * cm, 4.9 * cm, 4.6 * cm]))
story.append(Spacer(1, 6))
story += [
    P("2.2 Recorte de la escena", h2),
    P("Cada modelo divide la escena con su propia grilla de chips, <b>sin solape</b>, centrada en la "
      "escena (el borde sobrante se reparte por igual a ambos lados). El número de chips depende del "
      "tamaño del chip y de la escena, y por eso varía entre modelos. No se usa padding ni reescalado: una "
      "escena más pequeña que la huella del chip no produce chips."),
    P("En Prithvi se recorta primero una ventana de 672 × 672 px a 10 m y luego se reduce a 224 × 224 px "
      "a 30 m. La máscara recibe la misma reducción. Así la imagen y la máscara cubren exactamente el mismo "
      "terreno. Nueve escenas miden menos de 672 px en algún lado y quedan fuera de la evaluación de "
      "Prithvi: " + ", ".join(e["region"] for e in manifests["prithvi"]["excluded_pairs"]) + "."),
    P("En los tres modelos se usa la salida de la última capa del encoder y se descarta el token CLS. En "
      "SatMAE la distancia de cada token se combina entre los 3 grupos espectrales con el máximo "
      "(<font face='Courier'>combine_mode = max</font>); también se guarda la distancia de cada grupo."),
]

# ---------------------------------------------------------------- 3
story += [
    P("3. Flujo de procesamiento", h1),
    P("La ejecución separa la <b>extracción</b>, que es costosa y requiere los modelos, de la "
      "<b>evaluación</b>, que es rápida y trabaja sobre archivos. Así se pueden recalcular métricas, "
      "cambiar umbrales o generar figuras sin volver a pasar las imágenes por los modelos."),
    P("3.1 Extracción (<font face='Courier'>run_all_clay.py</font>, "
      "<font face='Courier'>run_all_satmae.py</font>, <font face='Courier'>run_all_prithvi.py</font>)", h2),
    P("Para cada chip se calculan los embeddings e1 (T1) y e2 (T2), se resumen a nivel de token y se "
      "descartan antes de pasar al chip siguiente. Por escena se guarda un archivo "
      "<font face='Courier'>&lt;modelo&gt;/&lt;region&gt;.npz</font> con:"),
]
store = [["Campo", "Contenido", "Uso"],
         ["dist", "Distancia coseno 1 − cos(e1, e2) por token", "G.2–G.4, histogramas, mapas"],
         ["p", "Fracción de píxeles con cambio en la región del token", "G.1"],
         ["label", "1 = C, 0 = NC, −1 = mixta", "G.1–G.4"],
         ["center_row_10m, center_col_10m", "Centro del token en la grilla de 10 m de la escena", "Mapas, cruce entre modelos"],
         ["chip_idx, token_row, token_col", "Chip y posición del token dentro del chip", "Mapas"],
         ["delta, delta_token_idx", "|e1 − e2| en float16, solo tokens C y NC", "UMAP (G.5)"],
         ["dist_groups (solo SatMAE)", "Distancia por grupo espectral", "Recombinar grupos"]]
story.append(table(store, [4.6 * cm, 7.4 * cm, 4.9 * cm]))
story.append(Spacer(1, 6))
story.append(P("Guardar e1 y e2 completos ocuparía unos 3 GB. Con este esquema la carpeta de resultados "
               "ocupa unos 670 MB (Clay 147 MB, SatMAE 521 MB, Prithvi 2 MB). Casi todo corresponde a "
               "|e1 − e2| de SatMAE. Los archivos .npz están excluidos de git porque superan el límite de "
               "GitHub y se regeneran con los scripts de extracción. Cada modelo escribe además un "
               "<font face='Courier'>manifest.json</font> con su configuración, los conteos por escena y "
               "las escenas excluidas."))
story += [
    P("3.2 Evaluación (<font face='Courier'>evaluate.py</font>)", h2),
    P("Lee los .npz de los tres modelos y escribe <font face='Courier'>evaluacion/por_par.csv</font> "
      "(una fila por modelo y par) y <font face='Courier'>evaluacion/resumen.json</font> (conteos, pares "
      "válidos e inválidos, y las comparaciones principal y complementaria)."),
]

# ---------------------------------------------------------------- 4
story += [
    P("4. Etiquetado de tokens (G.1)", h1),
    P("Para cada token, p = píxeles con cambio / píxeles de la región que cubre el token, medido sobre la "
      "máscara de OSCD llevada a la grilla nativa del modelo (se remuestrea la máscara, nunca los "
      "embeddings). Umbrales del protocolo: <b>C</b> si p &gt; 0.50, <b>NC</b> si p &lt; 0.05 y "
      "<b>mixta</b> (excluida) si 0.05 ≤ p ≤ 0.50."),
]
cnt = [["Modelo", "Región por token", "Tokens", "C", "NC", "Mixtas", "% excluido"]]
for m in MODELS:
    t = PM[m]["token_counts"]
    cnt.append([NAMES[m], f"{PM[m]['token_size_m']} m", f"{t['n_tokens']:,}".replace(",", " "),
                f"{t['n_C']:,}".replace(",", " "), f"{t['n_NC']:,}".replace(",", " "),
                f"{t['n_mixed_excluded']:,}".replace(",", " "), f"{t['pct_excluded']:.1f} %"])
story.append(table(cnt, [3.2 * cm, 2.6 * cm, 2.2 * cm, 1.9 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]))
story.append(Spacer(1, 6))
story.append(P("El desbalance es fuerte en los tres modelos: en Clay y SatMAE alrededor del 2 % de los "
               "tokens evaluados son C. En Prithvi la resolución del token domina: con regiones de "
               "480 × 480 m, el cambio urbano de OSCD rara vez ocupa más de la mitad de un token, y "
               "aumentan las mixtas (11.8 %)."))

# ---------------------------------------------------------------- 5
story += [
    P("5. Métricas (G.3 y G.4)", h1),
    P("Las métricas se calculan <b>por par de imágenes</b>, con todos los tokens C y NC de todos los chips "
      "de la escena, y se resumen con la media y la desviación estándar muestral (ddof = 1) entre pares."),
]
story += bullets([
    "<b>AUROC</b>: área bajo la curva ROC, con la distancia coseno como puntaje y C como clase positiva. "
    "0.5 equivale al azar y 1 a una separación perfecta. Un par solo es válido si tiene tokens C y NC.",
    "<b>Índice de Fisher</b>: F = (μ<sub>C</sub> − μ<sub>NC</sub>)<super>2</super> / "
    "(σ<sub>C</sub><super>2</super> + σ<sub>NC</sub><super>2</super>), con varianzas poblacionales dentro "
    "del par. Si el denominador es cero, el caso se reporta aparte; en esta ejecución no ocurrió.",
    "<b>Comparación principal</b>: solo los pares válidos para los tres modelos, como pide G.3. "
    "<b>Complementaria</b>: todos los pares válidos de cada modelo.",
])

# ---------------------------------------------------------------- 6
story += [P("6. Resultados", h1), P(f"6.1 Comparación principal: {len(common)} pares comunes ({', '.join(common)})", h2)]
res = [["Modelo", "Pares", "AUROC (media ± std)", "Fisher (media ± std)"]]
for m in MODELS:
    pr = PM[m]["principal"]
    res.append([NAMES[m], str(pr["auroc"]["n_pairs"]), ms(pr["auroc"]), ms(pr["fisher"], f2)])
story.append(table(res, [4 * cm, 2 * cm, 4.5 * cm, 4.5 * cm]))
story += [Spacer(1, 6), P("6.2 Complementaria: todos los pares válidos de cada modelo", h2)]
res = [["Modelo", "Pares", "AUROC (media ± std)", "Fisher (media ± std)"]]
for m in MODELS:
    pr = PM[m]["complementaria"]
    res.append([NAMES[m], str(pr["auroc"]["n_pairs"]), ms(pr["auroc"]), ms(pr["fisher"], f2)])
story.append(table(res, [4 * cm, 2 * cm, 4.5 * cm, 4.5 * cm]))
story.append(Spacer(1, 8))
story.append(KeepTogether([
    Image(str(FIG), width=16.5 * cm, height=16.5 * cm * 4.4 / 6.6),
    P("Figura 1. AUROC por par y modelo, con las escenas ordenadas por el AUROC de SatMAE. Prithvi solo "
      "aparece en abudhabi y cupertino: en las demás escenas no tiene chips o no tiene tokens C.", small),
]))
story += [P("6.3 Resultados por par", h2)]
pp = [["Par", "AUROC Clay", "AUROC SatMAE", "AUROC Prithvi", "Fisher Clay", "Fisher SatMAE", "Fisher Prithvi", "C (Clay / SatMAE / Prithvi)"]]
for s in SCENES:
    def g(m, k):
        r = by.get((m, s))
        if r is None:
            return "sin chips"
        if r["valid_auroc"] != "True":
            return "sin C" if r["n_C"] == "0" else "sin NC"
        return f3(r[k]) if k == "auroc" else f2(r[k])
    nc = " / ".join(by[(m, s)]["n_C"] if (m, s) in by else "—" for m in MODELS)
    pp.append([s, g("clay", "auroc"), g("satmae", "auroc"), g("prithvi", "auroc"),
               g("clay", "fisher"), g("satmae", "fisher"), g("prithvi", "fisher"), nc])
story.append(table(pp, [2.1 * cm, 1.75 * cm, 1.9 * cm, 1.9 * cm, 1.75 * cm, 1.9 * cm, 1.9 * cm, 3.8 * cm]))
story.append(P("«sin chips»: la escena es más pequeña que la huella del chip de Prithvi. "
               "«sin C»: la escena no tiene ningún token con p &gt; 0.50.", small))

# ---------------------------------------------------------------- 7
above = {m: sum(float(by[(m, s)]["auroc"]) >= 0.6 for s in SCENES) for m in ("clay", "satmae")}
story += [P("7. Interpretación", h1)]
story += bullets([
    "<b>Clay y SatMAE discriminan el cambio por encima del azar en la mayoría de los pares</b> "
    f"(AUROC ≥ 0.6 en {above['clay']} de 14 pares con Clay y en {above['satmae']} de 14 con SatMAE). SatMAE tiene un AUROC medio algo mayor "
    "y un índice de Fisher del doble, pero las desviaciones estándar entre pares (≈ 0.16 en AUROC) "
    "son mayores que la diferencia entre ambos, por lo que con 14 pares la diferencia no es concluyente.",
    "<b>Las escenas difíciles son las mismas para ambos modelos</b>: pisa, saclay_e y bercy, con "
    "AUROC cercano al azar, y Clay por debajo del azar en bercy (0.39). Esa coincidencia sugiere que la "
    "dificultad depende de la escena (tipo de cambio, poca cantidad de tokens C, diferencias "
    "radiométricas entre fechas) más que del modelo.",
    "<b>El índice de Fisher de SatMAE está dominado por pocas escenas</b>: va de 0.01 en pisa a 6.45 "
    "en cupertino. Por eso su desviación estándar supera a la media, y conviene leerlo junto con el AUROC.",
    "<b>El AUROC de Prithvi (0.958) no permite compararlo con los demás</b>: sale de 6 tokens C repartidos "
    "en 2 pares. Es coherente con los otros modelos en esos mismos pares, pero no es una estimación "
    "fiable.",
    "<b>Las distancias no son comparables en escala entre modelos.</b> Las distancias medias de Clay son "
    "del orden de 0.002–0.04, las de SatMAE de 0.07–0.43 y las de Prithvi de 0.02–0.06. Por eso la "
    "comparación se apoya en AUROC y Fisher, que no dependen de la escala absoluta.",
])

# ---------------------------------------------------------------- 8
story += [P("8. Limitaciones y diferencias con el protocolo", h1)]
story += bullets([
    "<b>Clay sin normalización de píxeles.</b> La función <font face='Courier'>prepare_for_clay</font> "
    "conserva un TODO: no aplica la media y desviación estándar oficiales de "
    "<font face='Courier'>metadata.yaml</font> (sensor sentinel-2-l2a), y el extractor tampoco lo hace. "
    "El punto D pide usar el preprocesamiento recomendado por los autores, así que los resultados de Clay "
    "deben repetirse con la normalización oficial antes de usarlos en la selección de modelo.",
    "<b>Bandas.</b> El protocolo (B) indica B02, B03, B04, B08, B11 y B12 para SatMAE y Clay. La ejecución "
    "usó 10 bandas en ambos (además B05, B06, B07 y B8A). En SatMAE responde a que el checkpoint group_c "
    "se entrenó con esos 3 grupos espectrales; en Clay es una decisión de implementación que el protocolo "
    "no contempla.",
    "<b>Metadatos de Prithvi.</b> El protocolo indica que Prithvi usa fechas y coordenadas. El checkpoint "
    "base <font face='Courier'>prithvi_eo_v2_300</font> no los acepta; la variante que sí lo hace es "
    "<font face='Courier'>prithvi_eo_v2_300_tl</font>.",
    "<b>Parche de Prithvi.</b> La tabla del punto E indica un parche interno de 14 × 14 px. El parche es "
    "de 16 × 16 px; 14 × 14 es la grilla de tokens (224 / 16).",
    "<b>Resolución del token.</b> Los modelos conservan su resolución nativa, como pide G.1, pero eso "
    "deja a Prithvi con tokens de 480 m frente a 80 m en los otros dos, y con muy pocos tokens C.",
    "<b>Prithvi procesa T1 y T2 juntos.</b> Sus dos bloques temporales pasan por la misma atención, así "
    "que su distancia T1–T2 no es independiente como en Clay y SatMAE, aunque el cálculo sea el mismo.",
    "<b>Coordenadas de Clay a nivel de ciudad.</b> Los archivos de OSCD en torchgeo no traen "
    "georreferenciación, así que lat/lon son aproximadas por ciudad y la hora se fija en 10:00.",
    "<b>Muestra pequeña.</b> Son 14 pares en total (5 para Prithvi). Las conclusiones aplican a los "
    "pares evaluados y no a la generalización a imágenes nuevas, como indica la sección 7 del protocolo.",
])

# ---------------------------------------------------------------- 9
story += [P("9. Decisiones pendientes", h1)]
story += bullets([
    "<b>Normalización de Clay.</b> Aplicar las estadísticas oficiales y repetir la extracción de Clay. "
    "Es prioritario, porque afecta los resultados de uno de los tres modelos.",
    "<b>Tratamiento de Prithvi.</b> Opciones: (a) mantener la configuración y reportar Prithvi como "
    "exploratorio, dejando Clay frente a SatMAE sobre 14 pares como comparación sólida; (b) pasarle la "
    "imagen a 10 m (tokens de 160 m), que se aparta de su resolución de preentrenamiento; (c) un umbral "
    "de C distinto para Prithvi, que se aparta de G.1. La opción (a) es la más fiel al protocolo.",
    "<b>Bandas y metadatos.</b> Ajustar el código al protocolo o el protocolo al código en los puntos "
    "de la sección 8. Cualquier cambio de bandas o de checkpoint obliga a repetir la extracción.",
    "<b>Visualización (G.5).</b> Histogramas, mapas de distancias de tres pares y UMAP, a partir de los "
    ".npz ya guardados.",
])

# ---------------------------------------------------------------- 10
story += [
    P("10. Reproducibilidad", h1),
    P("Desde la raíz del repositorio, con el entorno <font face='Courier'>.venv</font> activo. Los "
      "scripts de extracción son independientes y pueden ejecutarse en cualquier orden:"),
    P("python run_all_clay.py --oscd-root model_assets/clay/data/oscd \\<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;--clay-checkpoint model_assets/clay/clay-v1.5.ckpt \\<br/>"
      "&nbsp;&nbsp;&nbsp;&nbsp;--clay-metadata model_assets/clay/metadata.yaml<br/>"
      "python run_all_satmae.py --oscd-root model_assets/clay/data/oscd<br/>"
      "python run_all_prithvi.py --oscd-root model_assets/clay/data/oscd<br/>"
      "python evaluate.py", code),
    P("Estructura de resultados:"),
    P("results/comparacion_protocolo/<br/>"
      "&nbsp;&nbsp;clay/ satmae/ prithvi/&nbsp;&nbsp;&nbsp;&lt;region&gt;.npz + manifest.json<br/>"
      "&nbsp;&nbsp;evaluacion/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;por_par.csv + resumen.json", code),
    P("Archivos de código involucrados: <font face='Courier'>config.py</font> (ficha de cada modelo), "
      "<font face='Courier'>preprocessing.py</font> (bandas, remuestreo, recortes, normalización), "
      "<font face='Courier'>metrics.py</font> (distancia, p, etiquetas, AUROC, Fisher), "
      "<font face='Courier'>run_pipeline.py</font> (extracción por chip y guardado por escena), "
      "<font face='Courier'>models/*_extractor.py</font> (carga y forward de cada modelo) y "
      "<font face='Courier'>evaluate.py</font>."),
]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Comparación de representaciones · sección G del protocolo")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Comparación de representaciones: sección G del protocolo",
                        author="Gabriel Gómez")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("ok", OUT)
