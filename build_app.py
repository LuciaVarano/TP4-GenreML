"""build_app.py — genera docs/index.html, la página del TP4 (GenreML).

    python build_app.py

Toma el CSV que produce el pipeline de la Entrega 1 (plata), carga los artefactos del modelo de la Entrega 3
(si no están, corre entrenar.py con la misma semilla) y escribe una página única con gráficos de Altair.
Si los datos cambian, se corre de nuevo y la página se actualiza. «Probá el modelo» vive en Streamlit
(app_streamlit.py) y la página lo incrusta desde STREAMLIT_URL.
"""
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from string import Template

import altair as alt
import numpy as np
import pandas as pd

from modelo_app import FAMILIAS, MAPA_K10, etiqueta_familia

RAIZ = Path(__file__).parent
CSV = RAIZ / "genreML_tracks_latest.csv"            # la plata de la Entrega 1
ART = RAIZ / "artefactos"                            # lo que deja entrenar.py
SALIDA = RAIZ / "docs" / "index.html"                # GitHub Pages publica la carpeta /docs
STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "https://tp4-genreml-grupo2.streamlit.app/")   # <- poner la URL real
FECHA_DATOS = "9 de Septiembre de 2026"   # <- poner la fecha del corte que bajó el pipeline
N_PUNTOS = 20_000                                    # canciones que se dibujan en «Explorá» (sorteo con semilla 42)
SEMILLA = 42

# Colores con un solo significado en toda la página (y en Streamlit):
AZUL = "#2a78d6"      # lo que el modelo usa, elige o acierta
NARANJA = "#d95926"   # lo que queda afuera, el piso o el error
GRIS = "#7a7974"

NUM = ["duration_ms", "danceability", "energy", "loudness", "speechiness", "acousticness",
       "instrumentalness", "liveness", "valence", "tempo"]
CAT = ["key", "mode", "time_signature"]
ES_COL = {"duration_ms": "la duración", "danceability": "la bailabilidad", "energy": "la energía",
          "loudness": "el volumen", "speechiness": "la presencia de habla", "acousticness": "la acústica",
          "instrumentalness": "lo instrumental que es", "liveness": "el «en vivo»", "valence": "la positividad",
          "tempo": "el tempo", "key": "el tono", "mode": "el modo", "time_signature": "el compás"}

# ---- Números del notebook de la Entrega 3 que no se recalculan acá (salidas de las celdas 43, 51, 53, 59, 69, 102)
PISO_F1, PISO_ACC = 0.0002, 0.0104
COMPARACION = [("Random Forest", .2200, .0048, .2738), ("Gradient boosting", .2128, .0044, .2507),
               ("k vecinos", .1652, .0021, .1992), ("Regresión logística", .1458, .0031, .1961),
               ("Árbol de decisión", .1411, .0029, .1668), ("Piso (el género más frecuente)", PISO_F1, 0.0, PISO_ACC)]
ABLACION = [("Todas las columnas", 13, .2200, .0048), ("Sin time_signature (la elegida)", 12, .2182, .0038),
            ("Sin loudness", 12, .2123, .0039), ("Sin ambas", 11, .2083, .0044),
            ("Sin time_signature + 21 restas entre descriptores", 12, .2020, .0042)]
GRILLA = [("balanceados por submuestra", "sin límite", 2, .2245, .0046), ("iguales", "sin límite", 2, .2205, .0040),
          ("balanceados por submuestra", "sin límite", 6, .2133, .0017), ("iguales", "sin límite", 6, .2067, .0030),
          ("balanceados por submuestra", "16", 2, .2059, .0017)]
PERMUTACION = [("instrumentalness", .0722), ("acousticness", .0712), ("duration_ms", .0627), ("danceability", .0544),
               ("valence", .0503), ("energy", .0483), ("speechiness", .0420), ("loudness", .0386)]
TOPK_CV = {1: .272, 2: .378, 3: .443, 5: .529, 10: .647}     # validación cruzada, celda 102


def es(x, d=3):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def zona_eta2(v):
    """Zona del semáforo para η² (cortes 0,05 y 0,25, los de la Entrega 2)."""
    return "rojo" if v < 0.05 else "verde" if v > 0.25 else "amarillo"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def lista(xs):
    return ", ".join(xs[:-1]) + " y " + xs[-1]


def tabla(df):
    return '<div class="tabla">' + df.to_html(index=False, border=0, escape=True) + "</div>"


# ---------------------------------------------------------------- datos y artefactos
if not CSV.exists():
    raise FileNotFoundError(f"Falta {CSV.name}: es el CSV que produce el pipeline de la Entrega 1.")
if not (ART / "rangos.json").exists():
    subprocess.run([sys.executable, str(RAIZ / "entrenar.py")], check=True, cwd=RAIZ)   # misma semilla (42)

df = pd.read_csv(CSV)
meta = json.loads((ART / "metadatos.json").read_text(encoding="utf-8"))
pred = pd.read_csv(ART / "predicciones.csv.gz")
reporte = pd.read_csv(ART / "reporte_clases.csv", index_col=0)
confusiones = pd.read_csv(ART / "confusiones.csv")
FAM = json.loads((FAMILIAS / "genreml_mapas_familias.json").read_text(encoding="utf-8"))["mapas"][MAPA_K10]   # género -> familia (k = 10)
fam_cv = pd.read_csv(FAMILIAS / "genreml_familias_cv.csv")                                                 # validación cruzada de las familias

N_GEN = df.track_genre.nunique()
conteo = df.track_genre.value_counts()
acc, top3, f1 = meta["accuracy_test"], meta["top3_test"], meta["f1_macro_test"]
veces = round(acc * N_GEN)


def eta2(col):
    """Fracción de la varianza de la columna que explica el género (la medida de la Entrega 2)."""
    g = df.groupby("track_genre")[col]
    media = df[col].mean()
    return float((g.count() * (g.mean() - media) ** 2).sum() / ((df[col] - media) ** 2).sum())


eta = pd.DataFrame({"columna": NUM + CAT, "eta2": [eta2(c) for c in NUM + CAT]})
eta["uso"] = np.where(eta.columna == "time_signature", "Queda afuera", "Entra al modelo")
eta_orden = eta.sort_values("eta2", ascending=False)
mejores = [ES_COL[c] for c in eta_orden.columna[:3]]
eta_ts = float(eta.loc[eta.columna == "time_signature", "eta2"].iloc[0])

ex = df[["track_id", "acousticness", "energy", "valence"]].merge(pred, on="track_id")
ex["resultado"] = np.where(ex.genero == ex.pred1, "Acertó", "Se equivocó")
ex["familia_real"] = ex.genero.map(FAM).map(etiqueta_familia)       # familia del género verdadero
ex["familia_pred"] = ex.pred1.map(FAM).map(etiqueta_familia)        # familia del género que dijo el modelo
ex = ex.sample(N_PUNTOS, random_state=SEMILLA)[
    ["genero", "familia_real", "familia_pred", "acousticness", "energy", "valence", "resultado", "pred1", "pred2", "pred3", "p1", "particion"]]

# ---------------------------------------------------------------- gráficos (Altair)
alt.data_transformers.disable_max_rows()


def _tema():
    return {"config": {
        "view": {"stroke": None}, "background": "#ffffff", "padding": 8,
        "font": "Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif",
        "axis": {"domainColor": "#bbbbbb", "gridColor": "#ececec", "labelColor": "#555555", "labelFontSize": 13,
                 "tickColor": "#bbbbbb", "titleColor": "#555555", "titleFontSize": 13, "titleFontWeight": "normal"},
        "legend": {"labelColor": "#1a1a1a", "labelFontSize": 13, "titleColor": "#555555", "titleFontSize": 13},
        "title": {"anchor": "start", "fontSize": 16, "subtitleColor": "#555555", "subtitleFontSize": 13, "offset": 10}}}


alt.themes.register("tp4", _tema)
alt.themes.enable("tp4")


def g_datos():
    """Datos · una barra por género; la línea es la mediana. Lo que se mira: que casi no hay desbalance."""
    d = conteo.rename_axis("genero").reset_index(name="canciones")
    mediana = float(conteo.median())
    base = alt.Chart(d).encode(
        x=alt.X("genero:N", sort=conteo.index.tolist(), title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=8)),
        y=alt.Y("canciones:Q", title="Canciones", scale=alt.Scale(domain=[0, 1100])))
    barras = base.mark_bar(color=AZUL).encode(
        tooltip=[alt.Tooltip("genero:N", title="Género"), alt.Tooltip("canciones:Q", title="Canciones", format=",")])
    linea = alt.Chart(pd.DataFrame({"m": [mediana]})).mark_rule(color=NARANJA, strokeDash=[5, 4], strokeWidth=2).encode(y=alt.Y("m:Q", title="Canciones"))
    rotulo = alt.Chart(pd.DataFrame({"m": [mediana], "t": [f"mediana: {mediana:.0f} canciones por género"]})).mark_text(
        align="left", baseline="bottom", dy=-5, color=NARANJA, fontSize=13).encode(y=alt.Y("m:Q", title="Canciones"), x=alt.value(6), text="t:N")
    return alt.layer(barras, linea, rotulo).properties(
        width=820, height=260,
        title=alt.TitleParams(f"{miles(len(df))} canciones en {N_GEN} géneros; el más chico tiene {conteo.min()} "
                              f"({conteo.idxmin()}) y el más grande {conteo.max()} ({conteo.idxmax()})"))


def g_eta():
    """Lo que encontramos · η² de cada columna contra el género. Azul entra al modelo, naranja queda afuera."""
    base = alt.Chart(eta).encode(y=alt.Y("columna:N", sort=eta_orden.columna.tolist(), title=None),
                                 x=alt.X("eta2:Q", title="η²: parte de la variación de la columna que explica el género",
                                         scale=alt.Scale(domain=[0, 0.6])))
    barras = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color("uso:N", scale=alt.Scale(domain=["Entra al modelo", "Queda afuera"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("columna:N", title="Columna"), alt.Tooltip("eta2:Q", title="η²", format=".3f"),
                 alt.Tooltip("uso:N", title="En el modelo")])
    numeros = base.mark_text(align="left", dx=5, fontSize=13).encode(text=alt.Text("eta2:Q", format=".3f"),
                                                                      color=alt.value("#1a1a1a"))
    return alt.layer(barras, numeros).properties(
        width=640, height=330,
        title=alt.TitleParams("Cuánto separa cada columna a los géneros (0 = nada, 1 = lo determina)"))


def g_comparacion():
    """Modelo · media y ± un desvío de cada familia en validación cruzada; línea naranja = piso."""
    d = pd.DataFrame(COMPARACION, columns=["modelo", "media", "desvio", "accuracy"])
    d["lo"], d["hi"] = d.media - d.desvio, d.media + d.desvio
    escala = alt.Scale(domain=[0, 0.3])
    TITULO_X = "F1 macro en validación cruzada (media ± 1 desvío)"
    base = alt.Chart(d).encode(y=alt.Y("modelo:N", sort=d.modelo.tolist(), title=None))
    barra = base.mark_rule(strokeWidth=3, color=AZUL).encode(
        x=alt.X("lo:Q", title=TITULO_X, scale=escala), x2="hi:Q")
    punto = base.mark_point(filled=True, size=110, color=AZUL, opacity=1).encode(
        x=alt.X("media:Q", title=TITULO_X, scale=escala),
        tooltip=[alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("media:Q", title="F1 macro", format=".4f"),
                 alt.Tooltip("desvio:Q", title="Desvío", format=".4f"), alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f")])
    piso = alt.Chart(pd.DataFrame({"f": [PISO_F1]})).mark_rule(color=NARANJA, strokeDash=[5, 4], strokeWidth=2).encode(
        x=alt.X("f:Q", title=TITULO_X, scale=escala))
    return alt.layer(piso, barra, punto).properties(
        width=640, height=250, title=alt.TitleParams("El Random Forest gana por más que el desvío; la línea naranja es el piso"))


def g_topk():
    """Modelo · con k respuestas permitidas, cuántas veces está el género correcto (azul) contra el azar (naranja)."""
    d = pd.DataFrame({"k": list(TOPK_CV), "Modelo": list(TOPK_CV.values()), "Al azar": [k / N_GEN for k in TOPK_CV]})
    d = d.melt("k", var_name="serie", value_name="acierto")
    return alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=3).encode(
        x=alt.X("k:O", title="Respuestas permitidas (k)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("acierto:Q", title="El género correcto está entre las k primeras", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.7])),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Modelo", "Al azar"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("serie:N", title=""), alt.Tooltip("k:O", title="k"),
                 alt.Tooltip("acierto:Q", title="Acierto", format=".1%")]
    ).properties(width=640, height=250, title=alt.TitleParams("Dar tres intentos sube el acierto de 27 % a 44 % (validación cruzada)"))


def g_explora():
    """Copa · panorama de canciones (acústica × energía), coloreadas por si el modelo acertó.
    Cuatro controles tocan el filtro: género real, familia real, resultado del modelo y positividad mínima."""
    generos = ["Todos"] + sorted(ex.genero.unique())
    p_gen = alt.param(name="genero_sel", value="Todos", bind=alt.binding_select(
        options=generos, name="Género real: ", element="#c-explora"))
    p_fam = alt.param(name="familia_sel", value="Todas", bind=alt.binding_select(
        options=["Todas"] + sorted(ex.familia_real.unique()), name="Familia real: ", element="#c-explora"))
    p_res = alt.param(name="resultado_sel", value="Todas", bind=alt.binding_radio(
        options=["Todas", "Acertó", "Se equivocó"], name="Modelo: ", element="#c-explora"))
    p_val = alt.param(name="valencia_min", value=0, bind=alt.binding_range(
        min=0, max=1, step=0.05, name="Positividad mínima: ", element="#c-explora"))
    return alt.Chart(ex).mark_circle(size=26, opacity=0.4).encode(
        x=alt.X("acousticness:Q", title="Acústica", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("energy:Q", title="Energía", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("resultado:N", scale=alt.Scale(domain=["Acertó", "Se equivocó"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(title="El modelo", orient="bottom")),
        tooltip=[alt.Tooltip("genero:N", title="Género real"), alt.Tooltip("pred1:N", title="Modelo dice (1.º)"),
                 alt.Tooltip("p1:Q", title="Probabilidad", format=".0%"), alt.Tooltip("pred2:N", title="2.º"),
                 alt.Tooltip("pred3:N", title="3.º"), alt.Tooltip("familia_real:N", title="Familia real"), alt.Tooltip("familia_pred:N", title="Familia del 1.º"),
                 alt.Tooltip("resultado:N", title="Resultado"),
                 alt.Tooltip("particion:N", title="Partición"), alt.Tooltip("acousticness:Q", title="Acústica", format=".2f"),
                 alt.Tooltip("energy:Q", title="Energía", format=".2f"), alt.Tooltip("valence:Q", title="Positividad", format=".2f")]
    ).add_params(p_gen, p_fam, p_res, p_val).transform_filter(
        f"({p_gen.name} == 'Todos' || datum.genero == {p_gen.name}) && "
        f"({p_fam.name} == 'Todas' || datum.familia_real == {p_fam.name}) && "
        f"({p_res.name} == 'Todas' || datum.resultado == {p_res.name}) && datum.valence >= {p_val.name}"
    ).properties(width=760, height=460)


GRAFICOS = {"g-datos": g_datos(), "g-eta": g_eta(), "g-comp": g_comparacion(), "g-topk": g_topk(), "g-explora": g_explora()}
SPECS = json.dumps({k: c.to_dict() for k, c in GRAFICOS.items()}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

# ---------------------------------------------------------------- textos y tablas de «Cómo lo sabemos»
oracion = (f"Con sólo su audio se acierta el género exacto de {acc * 100:.0f} de cada 100 canciones, unas {veces} veces más "
           f"que al azar, y en {top3 * 100:.0f} de cada 100 el género correcto está entre las tres primeras respuestas del modelo.")

t_filtros = tabla(pd.DataFrame([
    ["Fuente original (bronce)", "114.000", "Una fila por par canción–género. 16.299 track_id aparecían con más de un género."],
    ["Plata (Entrega 1)", miles(len(df)), "Una fila por track_id: de cada canción multi-género se conservó el primero. 15 columnas, sin nulos."],
    ["Entrenamiento (80 %)", miles(meta["n_train"]), "Todo lo que se elige —familia, columnas, hiperparámetros— se elige acá, con validación cruzada."],
    ["Test (20 %)", miles(meta["n_test"]), "Se mira una vez. La misma grabación nunca queda de los dos lados (partición por grupos de audio idéntico)."],
    ["Audio repetido", "9.284 filas", "1.020 filas (260 grupos) tienen el mismo audio con géneros distintos: ningún modelo de audio puede acertar a la vez las dos etiquetas."],
], columns=["Paso", "Canciones", "Qué se hizo"]))

t_comp = tabla(pd.DataFrame([[m, es(f, 4) + " ± " + es(s, 4), es(a, 4)] for m, f, s, a in COMPARACION],
                            columns=["Modelo", "F1 macro CV (media ± desvío)", "Accuracy CV"]))
t_abl = tabla(pd.DataFrame([[n, c, es(f, 4) + " ± " + es(s, 4)] for n, c, f, s in ABLACION],
                           columns=["Columnas", "Cantidad", "F1 macro CV"]))
t_grilla = tabla(pd.DataFrame([[w, d, h, es(f, 4) + " ± " + es(s, 4)] for w, d, h, f, s in GRILLA],
                              columns=["Pesos de clase", "Profundidad", "Mín. por hoja", "F1 macro CV"]))
t_conf = tabla(confusiones.head(10).rename(columns={"real": "Género real", "predicho": "El modelo dijo", "errores": "Errores en el test"}))
rep = reporte.sort_values("f1-score", ascending=False)
t_gen = tabla(pd.DataFrame({"Mejores (F1)": [f"{g} ({es(v, 2)})" for g, v in rep["f1-score"].head(8).items()],
                            "Peores (F1)": [f"{g} ({es(v, 2)})" for g, v in rep["f1-score"].tail(8)[::-1].items()]}))
t_perm = tabla(pd.DataFrame([[c, es(v, 4)] for c, v in PERMUTACION], columns=["Columna", "Caída del F1 macro al mezclarla (test)"]))
imp = pd.Series(meta["importancia_impureza"]).sort_values(ascending=False).head(6)
def _num(x):
    return "—" if pd.isna(x) else es(float(x), 3)


def _fila_cv(objetivo, modo, nombre):
    r = fam_cv[(fam_cv.Objetivo == objetivo) & fam_cv.Modo.str.startswith(modo)].iloc[0]
    return [nombre, int(r.Clases), _num(r.Accuracy), _num(r["Accuracy piso"]), _num(r.Kappa), _num(r["Top-2"]), _num(r["Top-3"])]


t_fam_cv = tabla(pd.DataFrame([
    _fila_cv("Géneros (113)", "directo", "113 géneros (el objetivo)"), _fila_cv("k-means k=2 (silhouette)", "jer", "2 familias"),
    _fila_cv("k-means k=3", "jer", "3 familias"), _fila_cv(MAPA_K10, "jer", "10 familias (las de la página)"),
    _fila_cv("k-means k=20", "jer", "20 familias"), _fila_cv("Manual 20", "jer", "20 familias armadas a mano")],
    columns=["Agrupamiento", "Clases", "Accuracy", "Piso", "κ", "Top-2", "Top-3"]))
_por_fam = {}
for _g, _c in FAM.items():
    _por_fam.setdefault(_c, []).append(_g)
t_fam_lista = tabla(pd.DataFrame(
    [[etiqueta_familia(c), c, len(gs), ", ".join(sorted(gs)[:6]) + ("…" if len(gs) > 6 else "")]
     for c, gs in sorted(_por_fam.items(), key=lambda kv: -len(kv[1]))], columns=["Familia", "Código", "Géneros", "Algunos géneros"]))
_k10 = fam_cv[fam_cv.Objetivo == MAPA_K10]
F_JER, F_DIR = _k10[_k10.Modo.str.startswith("jer")].iloc[0], _k10[_k10.Modo == "directo"].iloc[0]
t_imp = tabla(pd.DataFrame({"Columna (tras el preprocesamiento)": imp.index, "Importancia por impureza": [es(v, 3) for v in imp]}))

COMO_DATOS = f"""
<h4>De dónde salen</h4>
<p>De la capa plata del pipeline de la Entrega 1: un conjunto de canciones de Spotify con sus descriptores de audio
(<code>{html.escape(CSV.name)}</code>). Fecha de los datos: <strong>{html.escape(FECHA_DATOS)}</strong>.
La página se regenera con <code>build_app.py</code> a partir de ese archivo.</p>
<h4>Qué es una fila</h4>
<p>Una canción: <code>track_id</code> es único, así que no hay duplicados de la clave. Pero la misma grabación puede estar
publicada en dos lanzamientos con otro <code>track_id</code>; por eso la partición agrupa el audio idéntico.</p>
<h4>Qué se filtró y qué quedó</h4>{t_filtros}
<h4>Qué columnas no entran</h4>
<p><code>track_genre</code> es lo que se predice. <code>track_id</code> es un identificador. Las columnas derivadas del género que se
crearon en el análisis (como <code>is_electronic</code>) quedan afuera porque contienen la respuesta. <code>time_signature</code> sale
por una decisión de modelado, no por fuga: se mide en «El modelo».</p>"""

COMO_HALLAZGOS = f"""
<h4>Una hipótesis que se sostuvo: <code>energy</code> y <code>loudness</code> dicen casi lo mismo (H3)</h4>
<p>Pearson 0,759 y Spearman 0,753. Cada una explica el género por separado con η² de 0,468 y 0,458; combinadas, 0,483: una mejora de
0,015, por debajo de la tolerancia de 0,02 fijada de antemano. Se puede prescindir de una sin perder información.</p>
<h4>Una que no: <code>time_signature</code> ayuda a predecir el género (H4)</h4>
<p>Partió en η² = 0,073 (amarillo). Al separar poblaciones bajó a 0,054 (amarillo) con los 19 géneros que tienen al menos 20 % de
canciones fuera de 4/4, y después a 0,045 (rojo) con los cinco de mayor proporción. El compás 4/4 domina en casi todos los géneros, y
la hipótesis quedó refutada. En el gráfico de arriba, recalculado sobre las {miles(len(df))} canciones, da {es(eta_ts)} ({zona_eta2(eta_ts)}).</p>
<p>El semáforo de η² se fijó antes de medir: rojo por debajo de 0,05, verde por encima de 0,25 y amarillo entre los dos.</p>
<h4>Cómo se construyó el gráfico</h4>
<p>η² es la suma de cuadrados entre géneros dividida por la suma de cuadrados total de cada columna. Para poder compararlas en un solo
eje, <code>key</code>, <code>mode</code> y <code>time_signature</code> se tratan como números sólo en esta medida (en el modelo son categóricas).
El color marca si la columna entra al modelo (azul) o queda afuera (naranja).</p>"""

COMO_MODELO = f"""
<h4>Métrica y piso</h4>
<p>F1 macro, declarada antes de experimentar: promedia el F1 de los {N_GEN} géneros por igual, así que no se puede esconder detrás de los
grandes. La accuracy y el top-k van de referencia. El piso es contestar siempre el género más frecuente: F1 macro {es(PISO_F1, 4)} y
accuracy {es(PISO_ACC, 3)} en validación cruzada (acertar al azar es 1/{N_GEN} = {es(1 / N_GEN, 3)}).</p>
<h4>Validación</h4>
<p>Se reserva un 20 % como test externo y el resto se valida con 5 pedazos (<code>StratifiedGroupKFold</code>, semilla {meta['semilla']}),
agrupando el audio idéntico para que ninguna copia caiga de los dos lados. El preprocesamiento (mediana, escala, one-hot,
<code>tempo</code> = 0 como faltante) se ajusta adentro de cada pedazo.</p>
<h4>La comparación completa</h4>{t_comp}
<h4>Qué columnas se usan</h4>
<p>Sacar <code>time_signature</code> cambia el F1 de 0,2200 a 0,2182, dentro del desvío, y deja una columna menos: se saca. Sacar
<code>loudness</code> sí baja más que el desvío (aunque H3 la daba por redundante): se queda. Las 21 restas entre descriptores
empeoran el resultado y no se usan.</p>{t_abl}
<h4>El ajuste del elegido</h4>
<p>Grilla de 100 árboles con profundidad y mínimo por hoja variables y dos formas de pesar las clases. Gana la de profundidad libre,
mínimo 2 por hoja y pesos balanceados por submuestra, a menos de un desvío de la segunda.</p>{t_grilla}
<h4>El test, medido una vez</h4>
<p>F1 macro <strong>{es(f1)}</strong>, accuracy <strong>{es(acc)}</strong>, top-3 <strong>{es(top3)}</strong> y top-5
<strong>{es(meta['top5_test'])}</strong>, sobre {miles(meta['n_test'])} canciones. En validación cruzada dio {es(meta['f1_macro_cv'], 4)}
± {es(meta['f1_cv_desvio'], 4)}: la búsqueda fue honesta. El modelo memoriza el entrenamiento (F1 de entrenamiento ≈ 0,99), y aun así
generaliza mejor que las otras cinco familias.</p>
<h4>Dónde se equivoca</h4>
<p>Los errores más grandes son entre géneros que suenan igual o que son el mismo con otro nombre. Los mejores géneros tienen una firma
de audio extrema; los peores son etiquetas que no son sonoras.</p>{t_conf}{t_gen}
<h4>En qué se apoya</h4>
<p>Mezclando una columna a la vez en el test, la que más hace caer el F1 es <code>instrumentalness</code>. Las columnas que parecen
redundantes (energía, volumen) se reparten el aporte: una importancia baja no prueba que la columna no sirva.</p>{t_perm}{t_imp}
<h4>Las familias: una capa para leer mejor el resultado</h4>
<p>El objetivo no cambia: se predice el género y se mide con F1 macro y top-k. Pero muchos errores son entre géneros que suenan igual, así que la
página agrupa los {N_GEN} géneros en 10 familias, las de la Parte 10 del notebook: k-means sobre el perfil medio de audio de cada género, calculado
sólo con el entrenamiento. Se agrupan <strong>géneros</strong> y no canciones: agrupar canciones da un objetivo que es función de las mismas columnas, y
cualquier clasificador lo aprende casi perfecto sin decir nada sobre géneros.</p>
<p>La app no entrena un segundo modelo: la probabilidad de cada familia es la suma de las probabilidades de sus géneros. En validación cruzada la
familia más probable acierta el {F_JER.Accuracy * 100:.0f} % de las canciones (piso: {F_JER["Accuracy piso"] * 100:.0f} %), con κ de {es(F_JER.Kappa, 2)}; la familia
correcta está entre las dos primeras en el {F_JER["Top-2"] * 100:.0f} % y entre las tres primeras en el {F_JER["Top-3"] * 100:.0f} %. Entrenar directo con la
familia como objetivo da prácticamente lo mismo (accuracy {es(F_DIR.Accuracy, 3)} contra {es(F_JER.Accuracy, 3)}), así que no hace falta un modelo nuevo.</p>{t_fam_cv}
<p>Las familias no son grupos separados sino un degradé: el silhouette elige k = 2 (Parte 10.3) y se usa k = 10 porque es la granularidad más útil
para leer, no porque el silhouette la elija. Agrupan por cómo suena (energía, acústica, bailabilidad), no por tradición o mercado, así que mezclan
géneros que una persona separaría. DBSCAN (<code>min_samples</code> = 4, <code>eps</code> = percentil 90 de la distancia de cada género a su cuarto vecino) se
usó para ver qué géneros no se parecen a ninguno; en k-means eso se nota en que <code>comedy</code> termina solo en su familia.</p>{t_fam_lista}
<h4>Por qué «Probá el modelo» corre en Streamlit</h4>
<p>Un bosque de 100 árboles sin límite de profundidad y con {N_GEN} clases no se puede escribir como una fórmula dentro de una página.
La alternativa que sí entra en el navegador es la regresión logística (F1 macro 0,1458 ± 0,0031), que queda lejos del
mejor (0,2245 ± 0,0046): no está a menos de un desvío. Se elige Streamlit y se conserva el mejor modelo. Para que quepa en GitHub, el bosque
se guarda en un formato compacto (28 MB en vez de ~330 MB) que da las mismas probabilidades que scikit-learn.</p>"""

# ---------------------------------------------------------------- la página
PAGINA = Template("""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>¿Se puede predecir el género de una canción por su audio?</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--tinta:#1a1a1a;--gris:#555;--suave:#f5f4f1;--linea:#dedbd4;--tallo:#1d4e89;--copa:#b5541c;--fondo:#fff}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;color:var(--tinta);background:var(--fondo);font:18px/1.55 "Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif}
h1,h2{font-family:Fraunces,Georgia,serif;font-weight:600}
nav{position:fixed;inset:0 auto 0 0;width:230px;padding:28px 18px;border-right:1px solid var(--linea);background:#faf9f7;overflow-y:auto}
nav .marca{font-size:13px;color:var(--gris);margin:0 0 22px}
nav h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin:22px 0 6px;padding-left:12px}
nav .tallo h3{color:var(--tallo)} nav .copa h3{color:var(--copa)}
nav ol{list-style:none;margin:0;padding:0 0 0 12px} nav .tallo ol{border-left:3px solid var(--tallo)} nav .copa ol{border-left:3px solid var(--copa)}
nav li a{display:block;padding:5px 0 5px 12px;color:var(--tinta);text-decoration:none;font-size:15px;line-height:1.3}
nav .tallo li a:hover,nav .tallo li a.activo{color:var(--tallo);font-weight:600} nav .copa li a:hover,nav .copa li a.activo{color:var(--copa);font-weight:600}
nav .copa-nota{font-size:12.5px;color:var(--gris);padding-left:12px;margin:8px 0 0;line-height:1.4}
main{margin-left:230px;padding:0 48px 80px}
section{max-width:960px;margin:0 auto;padding:56px 0 24px;border-bottom:1px solid var(--linea)} section:last-of-type{border-bottom:none}
.etapa{font-size:13px;letter-spacing:.08em;text-transform:uppercase;font-weight:700;margin:0 0 6px}
.etapa.tallo{color:var(--tallo)} .etapa.copa{color:var(--copa)}
h1{font-size:42px;line-height:1.15;margin:0 0 18px;max-width:20ch} h2{font-size:29px;line-height:1.25;margin:0 0 14px;max-width:34ch}
p,li{max-width:72ch} .lead{font-size:21px;color:var(--gris);max-width:62ch}
.oracion{font-size:24px;line-height:1.4;border-left:5px solid var(--tallo);padding:4px 0 4px 20px;margin:26px 0 34px;max-width:52ch}
.nums{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:10px 0 18px}
.num{background:var(--suave);border-radius:8px;padding:18px 20px} .num span{display:block;font-size:42px;font-weight:700;line-height:1.1;color:var(--tallo)}
.num p{margin:6px 0 0;font-size:15.5px;color:var(--gris)} .grafico{margin:22px 0 10px;overflow-x:auto} .nota{font-size:15px;color:var(--gris)}
details{background:var(--suave);border-radius:8px;padding:0 20px;margin:18px 0} details[open]{padding-bottom:12px}
summary{cursor:pointer;font-weight:700;padding:14px 0;color:var(--tallo)} details h4{margin:16px 0 6px;font-size:17px} details p,details li{font-size:16px}
.tabla{overflow-x:auto;margin:8px 0 14px} table{border-collapse:collapse;font-size:15px;background:#fff}
th,td{border:1px solid var(--linea);padding:6px 10px;text-align:left;vertical-align:top} th{background:#efede8}
code{font-size:.9em;background:#efede8;padding:1px 5px;border-radius:4px}
.controles{display:flex;flex-wrap:wrap;gap:8px 22px;margin:18px 0 0;font-size:15px;padding:12px 16px;background:var(--suave);border-radius:8px}
.vega-bind{display:flex;align-items:center;gap:6px} .vega-bind-name{font-weight:600} .vega-bind select{font-size:15px;padding:4px 6px;max-width:260px}
.vega-bind-radio label{margin-right:10px} .vega-bind input[type=range]{accent-color:var(--copa)}
iframe{width:100%;height:1250px;border:1px solid var(--linea);border-radius:8px;background:#fff}
footer{max-width:960px;margin:0 auto;padding:30px 0;color:var(--gris);font-size:14px}
@media (max-width:860px){nav{display:none}main{margin-left:0;padding:0 16px 60px}.nums{grid-template-columns:1fr}h1{font-size:32px}}
</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head><body>
<nav><p class="marca">Ciencia de Datos · UTN FRM · 2026<br>GenreML · Proyecto Integrador</p>
<div class="tallo"><h3>La historia</h3><ol><li><a href="#portada">El proyecto</a></li><li><a href="#datos">Los datos</a></li>
<li><a href="#hallazgos">Lo que encontramos</a></li><li><a href="#modelo">El modelo</a></li></ol></div>
<div class="copa"><h3>Tu turno</h3><ol><li><a href="#explora">Explorá las canciones</a></li><li><a href="#proba">Probá el modelo</a></li></ol>
<p class="copa-nota">La historia se lee en orden. Acá las preguntas las hacés vos.</p></div></nav>
<main>
<section id="portada"><p class="etapa tallo">El proyecto</p>
<h1>¿Se puede predecir el género de una canción por su audio?</h1>
<p class="lead">Cada canción de Spotify trae diez descriptores medibles —energía, acústica, bailabilidad, tempo, volumen— y dos o tres rasgos musicales.
Si los géneros suenan distinto, esos números tendrían que delatar a cuál pertenece.</p>
<p class="oracion">$oracion</p>
<div class="nums"><div class="num"><span>$n_acc %</span><p>de las $n_test canciones del test, con el género exacto bien adivinado. Al azar sería 0,9 %</p></div>
<div class="num"><span>$n_top3 %</span><p>de las veces el género correcto está entre las tres primeras respuestas</p></div>
<div class="num"><span>$n_f1</span><p>F1 macro en el test, contra $n_piso del piso que contesta siempre el género más frecuente</p></div></div>
<p class="nota">Medido sobre $n_total canciones de $n_gen géneros. Es un resultado modesto y honesto: muchos géneros son el mismo sonido con otro nombre.</p></section>

<section id="datos"><p class="etapa tallo">Los datos</p>
<h2>Los $n_gen géneros tienen casi el mismo tamaño, así que el problema no es el desbalance sino la cantidad de géneros</h2>
<p>Hay una mediana de $n_med canciones por género. Un modelo que contestara siempre el género más grande acertaría alrededor de 1 %:
cualquier resultado se lee contra ese piso y no contra 100 %.</p>
<div class="grafico" id="g-datos"></div>
<details><summary>Cómo lo sabemos</summary>$como_datos</details></section>

<section id="hallazgos"><p class="etapa tallo">Lo que encontramos</p>
<h2>$titulo_hallazgo</h2>
<p>Cada barra mide cuánto de la variación de una columna se explica sabiendo el género. En azul están las columnas que usa el modelo;
en naranja, la que se descartó.</p>
<div class="grafico" id="g-eta"></div>
<details><summary>Cómo lo sabemos</summary>$como_hallazgos</details></section>

<section id="modelo"><p class="etapa tallo">El modelo</p>
<h2>Un Random Forest acierta el género exacto de $n_acc de cada 100 canciones: mucho más que el azar, pero lejos de ser infalible</h2>
<p>La métrica es el F1 macro, que pesa igual a los $n_gen géneros. El Random Forest le gana a las otras cinco familias por más que el desvío.
Pero un solo intento castiga cosas que el audio no resuelve: dándole tres, el acierto sube a casi la mitad.</p>
<div class="grafico" id="g-comp"></div><div class="grafico" id="g-topk"></div>
<details><summary>Cómo lo sabemos</summary>$como_modelo</details></section>

<section id="explora"><p class="etapa copa">Tu turno · Explorá</p>
<h2>$n_puntos canciones ubicadas por lo acústicas y lo enérgicas que son</h2>
<p>Empezá por el panorama: casi todo es una sola nube y los géneros se mezclan, que es justo lo que le cuesta al modelo.
Después filtrá por género, por familia, por el resultado del modelo o por la positividad mínima, y pasá el mouse por una canción para ver qué dijo
el modelo y de qué familia es el género que eligió: así se ve si el error fue dentro de la familia o entre familias distintas. Para las canciones de entrenamiento la respuesta sale de la validación cruzada: la dio un modelo que no la había visto.</p>
<div class="controles" id="c-explora"></div><div class="grafico" id="g-explora"></div>
<p class="nota">Se dibuja un sorteo de $n_puntos de las $n_total canciones (semilla 42).</p></section>

<section id="proba"><p class="etapa copa">Tu turno · Probá el modelo</p>
<h2>Armá una canción y mirá qué género le daría el modelo</h2>
<p>Cada deslizador va del mínimo al máximo que vio el entrenamiento, así que no se puede cargar un valor desconocido. El modelo corre
en Streamlit y devuelve los cinco géneros más probables y la familia más probable.
<a href="$url" target="_blank" rel="noopener">Abrir en otra pestaña</a>.</p>
<iframe src="$url/?embed=true" title="Probá el modelo" loading="lazy"></iframe></section>
<footer>Ciencia de Datos · UTN FRM · 2026 · Grupo 5K10-02 · Datos: $fecha.</footer>
</main>
<script>
const OPC = {actions: false, renderer: "canvas",
  formatLocale: {decimal: ",", thousands: ".", grouping: [3], currency: ["", " €"], percent: " %"}};
const SPECS = $specs;
Object.entries(SPECS).forEach(([id, s]) => vegaEmbed("#" + id, s, OPC).catch(e => {
  document.getElementById(id).textContent = "No se pudo dibujar el gráfico: " + e; }));
const enlaces = [...document.querySelectorAll("nav a")];
const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting)
  enlaces.forEach(a => a.classList.toggle("activo", a.getAttribute("href") === "#" + e.target.id)); }), {rootMargin: "-40% 0px -55% 0px"});
document.querySelectorAll("section").forEach(s => obs.observe(s));
</script></body></html>""")

pagina = PAGINA.substitute(
    oracion=oracion, n_acc=f"{acc * 100:.0f}", n_top3=f"{top3 * 100:.0f}", n_f1=es(f1), n_piso=es(PISO_F1, 4),
    n_test=miles(meta["n_test"]), n_total=miles(len(df)), n_gen=N_GEN, n_med=f"{conteo.median():.0f}", n_puntos=miles(N_PUNTOS),
    titulo_hallazgo=f"Los géneros se distinguen sobre todo por {lista(mejores)}; el compás explica apenas el {eta_ts * 100:.0f} % de la variación",
    como_datos=COMO_DATOS, como_hallazgos=COMO_HALLAZGOS, como_modelo=COMO_MODELO,
    url=STREAMLIT_URL.rstrip("/"), fecha=html.escape(FECHA_DATOS), specs=SPECS)

SALIDA.parent.mkdir(exist_ok=True)
SALIDA.write_text(pagina, encoding="utf-8")
print(f"Listo: {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
