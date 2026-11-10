"""Probá el modelo: el usuario arma una canción y ve qué géneros le asigna el Random Forest de la Entrega 3.
Correr en local:  streamlit run app_streamlit.py"""
import altair as alt
import pandas as pd
import streamlit as st

from modelo_app import ModeloGeneros

AZUL, GRIS = "#2a78d6", "#b9bcc2"          # mismos colores que la página: azul = lo que el modelo elige
NOTAS = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
ETIQUETAS = {"danceability": "Bailabilidad", "energy": "Energía", "loudness": "Volumen (dB)",
             "speechiness": "Presencia de habla", "acousticness": "Acústica", "instrumentalness": "Instrumental",
             "liveness": "En vivo", "valence": "Positividad (valence)", "tempo": "Tempo (BPM)",
             "duration_ms": "Duración (segundos)"}

st.set_page_config(page_title="Probá el modelo · GenreML", layout="wide")


@st.cache_resource
def cargar():
    return ModeloGeneros()


m = cargar()
meta = m.meta


def volver_a_la_mediana():
    for c, (_, _, med) in m.rangos.items():
        if c == "duration_ms":
            st.session_state[c] = round(med / 1000)
        elif c in ("key", "mode"):
            st.session_state[c] = int(med)
        else:
            st.session_state[c] = float(med)


st.title("Armá una canción y mirá qué género le daría el modelo")
st.caption(f"Random Forest de la Entrega 3 · {len(m.clases)} géneros · en el test acierta el género exacto "
           f"el {meta['accuracy_test']:.1%} de las veces y lo tiene entre sus tres primeras respuestas el "
           f"{meta['top3_test']:.1%}. Cada deslizador va del mínimo al máximo que vio el entrenamiento.")

izq, der = st.columns([1, 1.15])
valores = {}
with izq:
    st.button("Volver a la mediana", on_click=volver_a_la_mediana)
    for c in ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]:
        lo, hi, med = m.rangos[c]
        valores[c] = st.slider(ETIQUETAS[c], float(lo), float(hi), float(med), step=0.01, key=c)
    lo, hi, med = m.rangos["loudness"]
    valores["loudness"] = st.slider(ETIQUETAS["loudness"], float(lo), float(hi), float(med), step=0.1, key="loudness")
    lo, hi, med = m.rangos["tempo"]
    valores["tempo"] = st.slider(ETIQUETAS["tempo"], float(lo), float(hi), float(med), step=0.5, key="tempo")
    lo, hi, med = m.rangos["duration_ms"]
    segundos = st.slider(ETIQUETAS["duration_ms"], int(-(-lo // 1000)), int(hi // 1000), int(med // 1000), key="duration_ms")
    valores["duration_ms"] = segundos * 1000
    nota = st.selectbox("Tono (key)", m.pre["key"], format_func=lambda k: NOTAS[int(k)], index=int(m.rangos["key"][2]), key="key")
    modo = st.radio("Modo", m.pre["mode"], format_func=lambda k: "Mayor" if k == 1 else "Menor", horizontal=True, key="mode")
    valores["key"], valores["mode"] = nota, modo

with der:
    raros = m.fuera_de_rango(valores)
    if raros:   # los deslizadores lo impiden; queda por si se carga un caso por otra vía
        st.warning("El modelo nunca vio valores así en: " + ", ".join(raros) + ". La predicción no es confiable.")
    if valores["tempo"] == 0:
        st.info("Tempo 0 significa «sin tempo detectado»: el modelo lo reemplaza por la mediana del entrenamiento.")

    p = pd.Series(m.probabilidades(valores), index=m.clases).sort_values(ascending=False)
    top = p.head(5).rename_axis("genero").reset_index(name="probabilidad")
    top["posicion"] = ["1.º"] + [""] * 4
    st.subheader(f"Más probable: {top.genero[0]} ({top.probabilidad[0]:.0%})")
    st.write(f"Entre las tres primeras: {', '.join(top.genero[:3])}, que suman {top.probabilidad[:3].sum():.0%}. "
             f"Con 113 géneros, acertar al azar es 0,9 %: una probabilidad del 20 % ya es mucha.")

    barras = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("genero:N", sort=top.genero.tolist(), title=None),
        x=alt.X("probabilidad:Q", axis=alt.Axis(format="%"), title="Probabilidad que le da el modelo"),
        color=alt.condition(alt.datum.posicion == "1.º", alt.value(AZUL), alt.value(GRIS)),
        tooltip=[alt.Tooltip("genero:N", title="Género"), alt.Tooltip("probabilidad:Q", title="Probabilidad", format=".1%")])
    etiquetas = barras.mark_text(align="left", dx=4).encode(text=alt.Text("probabilidad:Q", format=".0%"), color=alt.value("#1a1a1a"))
    st.altair_chart((barras + etiquetas).properties(height=230), use_container_width=True)
    st.caption("Las probabilidades salen del promedio de los 100 árboles del bosque. El modelo recibe las mismas 12 columnas "
               "que en el notebook (sin time_signature) y el mismo preprocesamiento.")
