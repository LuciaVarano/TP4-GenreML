# GenreML · ¿Se puede predecir el género de una canción por su audio?

Ciencia de Datos · UTN FRM · 2026 · Grupo 5K10-02 — Proyecto integrador (Entrega 4)

| | |
|---|---|
| **Aplicación (página)** | https://luciavarano.github.io/TP4-GenreML/ |
| **Probá el modelo (Streamlit)** | https://tp4-genreml-grupo2.streamlit.app/ |
| **Repositorio** | https://github.com/LuciaVarano/TP4-GenreML |

Con 12 atributos de audio, un Random Forest acierta el género exacto de 29 de cada 100 canciones (entre 113 géneros) y lo tiene entre sus tres
primeras respuestas el 46 % de las veces. El proyecto es una cadena: **pipeline** (Entrega 1) → **análisis** (Entrega 2) →
**modelo** (Entrega 3) → **aplicación** (Entrega 4). Todo lo que muestra la aplicación sale de los datos y del modelo de este repositorio.

## Qué hay en el repositorio

| Pieza | Archivo | Qué hace |
|---|---|---|
| Pipeline (Entrega 1) | `entregas/entrega1_Airflow.py` | DAG `GenreML` de Airflow: baja el dataset de Hugging Face, deja una copia cruda (bronce), arma y valida la capa plata y publica `genreML_tracks_latest.csv`. |
| Análisis (Entrega 2) | `entregas/entrega2_hipotesis.ipynb` | Perfil del dataset y cuatro hipótesis (H1–H4), medidas con un semáforo fijado de antemano. |
| Modelo (Entrega 3) | `entregas/entrega3_modelado.ipynb` | Compara varias familias de modelos contra un piso y elige un Random Forest. |
| Reproducción del modelo | `entrenar.py` + `bosque_compacto.py` | Mismo modelo y misma semilla (42) que la Entrega 3; deja todo en `artefactos/`. |
| Familias | `familias/` | Las 10 familias de géneros de la Entrega 3 (k-means k = 10): `genreml_mapas_familias.json` y `genreml_familias_cv.csv`. Son una capa para leer mejor el resultado; el objetivo sigue siendo el género. |
| Construcción de la página | `build_app.py` → `docs/index.html` | Toma el CSV del pipeline y los artefactos del modelo y genera la página: el tallo y «Explorá», con gráficos de Altair. |
| Probá el modelo | `app_streamlit.py` + `modelo_app.py` | Streamlit: el usuario arma una canción y ve las probabilidades de los cinco géneros más probables y de la familia más probable. |

## Cómo se corre todo

Hace falta Python 3.12 o 3.13.

```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Pipeline (Entrega 1).** Corre en Airflow 3 (el DAG usa `airflow.sdk`) y necesita internet para bajar los Parquet de
`maharshipandya/spotify-tracks-dataset`. Copiá `entregas/entrega1_Airflow.py` a la carpeta `dags/` de tu Airflow, disparalo (DAG `GenreML`) y esperá a que
terminen las seis tareas (`resolve_source_revision`, `land_bronze`, `inspect_bronze`, `build_silver`, `validate_silver` y `publish_dataset`).
Deja en `/usr/local/airflow/include/output/genreML/` dos archivos: `genreML_tracks_<fecha>.csv` (el corte fechado) y `genreML_tracks_latest.csv`.
Copiá el segundo a la raíz de este repositorio. El repositorio ya trae un `genreML_tracks_latest.csv`, así que este paso solo hace falta para actualizar los datos.

**2. Análisis y modelo (Entregas 2 y 3).** Los notebooks de `entregas/` se abren con Jupyter y leen el mismo CSV. De la Entrega 3 salen las
decisiones (qué modelo, qué columnas, qué hiperparámetros) y las dos tablas de `familias/`.

**3. Entrenar y exportar el modelo** (unos 4 minutos; deja `artefactos/`):

```bash
python entrenar.py
```

**4. Antes de generar la página**, revisá dos constantes al principio de `build_app.py`: `FECHA_DATOS` (la fecha del corte del CSV, la del archivo
`genreML_tracks_<fecha>.csv` del pipeline) y `STREAMLIT_URL` (o definí la variable de entorno `STREAMLIT_URL`).

**5. Generar la página:**

```bash
python build_app.py                 # escribe docs/index.html
```

Si falta `artefactos/`, `build_app.py` corre `entrenar.py` solo. Para ver la página en local: `python -m http.server -d docs 8000` y abrí http://localhost:8000.

**6. Probar «Probá el modelo» en local:**

```bash
streamlit run app_streamlit.py
```

## Si los datos se actualizan

1. Corré de nuevo el pipeline y reemplazá `genreML_tracks_latest.csv` en la raíz.
2. Borrá la carpeta `artefactos/` y corré `python entrenar.py` (reentrena con la semilla 42).
3. Corré `python build_app.py`, actualizá `FECHA_DATOS` si cambió el corte, y hacé commit y push de `artefactos/` y `docs/index.html`.

Los números de la página (género más grande, η², resultado en el test, importancias por impureza) se recalculan solos.
Las tablas de comparación de familias, ablación, grilla y permutación son salidas del notebook de la Entrega 3 y están copiadas en las
constantes de `build_app.py`; si se vuelve a correr el notebook con otros datos, hay que actualizarlas. Las familias (`familias/`) también salen de ese
notebook: hay que volver a correrlo y reemplazar esos dos archivos.

## Publicación

* **Página** → GitHub Pages, desde la carpeta `/docs` de la rama `main` (Settings → Pages).
* **Probá el modelo** → Streamlit Community Cloud, con `app_streamlit.py` como archivo principal. La página lo incrusta. La app necesita en el
  repositorio `artefactos/`, `familias/`, `modelo_app.py`, `bosque_compacto.py`, `requirements.txt` y `.streamlit/config.toml`. Las apps gratuitas se
  duermen por inactividad: conviene abrirla unos minutos antes de usarla.

## Decisiones de modelado que conviene saber

* **Familias:** no hay un segundo modelo. La probabilidad de una familia es la suma de las probabilidades de sus géneros (`probabilidades_familia` en `modelo_app.py`); es el modo «jerárquico» del notebook. La familia `K10-05` no recibió nombre en el notebook: se le puede poner uno en `NOMBRES_FAMILIA` de `modelo_app.py`.
* **Modelo:** Random Forest de 100 árboles, sin `time_signature`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, sin límite de profundidad.
* **Métrica:** F1 macro (pesa igual a los 113 géneros). Piso: contestar siempre el género más frecuente.
* **Validación:** `StratifiedGroupKFold` de 5 pedazos, agrupando el audio idéntico; test externo del 20 %, medido una vez.
* **Cómo llega el modelo a la app:** un bosque de 100 árboles con 113 clases pesa ~330 MB comprimido, y no entra en GitHub ni en Streamlit Cloud.
  `bosque_compacto.py` lo guarda en ~28 MB (por nodo: columna, umbral e hijo derecho; las hojas, de forma dispersa) y da las mismas
  probabilidades que scikit-learn. Al cargarlo lee cada arreglo una sola vez y copia cada rebanada: carga en ~1 segundo y ~300 MB de memoria, que es lo que aguanta Streamlit Cloud.
  El preprocesamiento se exporta a `preproceso.json` y la app lo repite con numpy, así que Streamlit no depende de la versión de scikit-learn.
* **Casos raros:** los deslizadores van del mínimo al máximo del entrenamiento; `tempo = 0` («sin tempo detectado») se reemplaza por la mediana.
* **Predicciones de «Explorá»:** para las canciones de entrenamiento son fuera de fold; para las de test, las del modelo final.

## Estructura

```
├── README.md
├── requirements.txt
├── build_app.py                 genera docs/index.html
├── entrenar.py                  genera artefactos/
├── bosque_compacto.py           guarda y carga el bosque en formato compacto
├── modelo_app.py                preprocesamiento y predicción (lo usa Streamlit)
├── app_streamlit.py             «Probá el modelo»
├── genreML_tracks_latest.csv    la plata de la Entrega 1
├── entregas/
│   ├── entrega1_Airflow.py      pipeline
│   ├── entrega2_hipotesis.ipynb análisis
│   └── entrega3_modelado.ipynb  modelo
├── familias/                    del notebook de la Entrega 3
├── artefactos/                  lo genera entrenar.py
├── docs/index.html              lo genera build_app.py (GitHub Pages)
└── .streamlit/config.toml       tema claro de Streamlit
```
