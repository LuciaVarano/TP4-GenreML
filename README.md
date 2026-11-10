# GenreML · ¿Se puede predecir el género de una canción por su audio?

Ciencia de Datos · UTN FRM · 2026 · Grupo 5K10-02 — Proyecto integrador (Entrega 4)

Cadena completa: **pipeline** (Entrega 1) → **análisis** (Entrega 2) → **modelo** (Entrega 3) → **aplicación** (Entrega 4).

| Pieza | Archivo | Qué hace |
|---|---|---|
| Pipeline | `entrega1_pipeline.py` | Produce el CSV de canciones (capa plata). |
| Análisis | `entrega2_hipotesis.ipynb` | Perfil del dataset y cuatro hipótesis (H1–H4). |
| Modelo | `entrega3_modelado.ipynb` | Compara seis familias; elige un Random Forest. |
| Reproducción del modelo | `entrenar.py` + `bosque_compacto.py` | Mismo modelo y misma semilla (42) que la Entrega 3; deja todo en `artefactos/`. |
| Familias | `familias/` | Las 10 familias de géneros de la Entrega 3 (k-means k = 10): `genreml_mapas_familias.json` y `genreml_familias_cv.csv`. Son una capa para leer mejor el resultado; el objetivo sigue siendo el género. |
| Página | `build_app.py` → `docs/index.html` | El tallo y la exploración, con gráficos de Altair. |
| Probá el modelo | `app_streamlit.py` + `modelo_app.py` | Streamlit: el usuario arma una canción y ve las probabilidades. |

## Cómo se corre todo

```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Poné `genreML_tracks_latest.csv` (la plata de la Entrega 1) en la raíz del repositorio.
2. Entrená y exportá el modelo (unos 4 minutos; deja `artefactos/`):
   ```bash
   python entrenar.py
   ```
3. Antes de generar la página, editá dos constantes al principio de `build_app.py`: `FECHA_DATOS` (fecha del corte del CSV) y la URL de Streamlit
   (o definí la variable `STREAMLIT_URL`).
4. Generá la página:
   ```bash
   python build_app.py                 # escribe docs/index.html
   ```
   Para verla: `python -m http.server -d docs 8000` y abrí http://localhost:8000.
5. Probá «Probá el modelo» en local:
   ```bash
   streamlit run app_streamlit.py
   ```

## Si los datos se actualizan

Reemplazá el CSV, borrá la carpeta `artefactos/` y corré de nuevo `python entrenar.py` y `python build_app.py`.
Las familias (`familias/`) salen del notebook de la Entrega 3: si cambian los datos, hay que volver a correrlo y reemplazar esos dos archivos.
Los números de la página (género más grande, η², test, importancias por impureza) se recalculan solos.
Las tablas de comparación de familias, ablación, grilla y permutación son salidas del notebook de la Entrega 3 y están copiadas en
las constantes de `build_app.py`; si se vuelve a correr el notebook con otros datos, hay que actualizarlas.

## Publicación

* **Página** → GitHub Pages, desde la carpeta `/docs` de la rama `main`.
* **Probá el modelo** → Streamlit Community Cloud, con `app_streamlit.py` como archivo principal. La página lo incrusta.

## Decisiones de modelado que conviene saber

* **Familias:** no hay un segundo modelo. La probabilidad de una familia es la suma de las probabilidades de sus géneros (`probabilidades_familia` en `modelo_app.py`); es el modo «jerárquico» del notebook. La familia `K10-05` no recibió nombre en el notebook: se le puede poner uno en `NOMBRES_FAMILIA` de `modelo_app.py`.
* **Modelo:** Random Forest de 100 árboles, sin `time_signature`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, sin límite de profundidad.
* **Métrica:** F1 macro (pesa igual a los 113 géneros). Piso: contestar siempre el género más frecuente.
* **Validación:** `StratifiedGroupKFold` de 5 pedazos, agrupando el audio idéntico; test externo del 20 %, medido una vez.
* **Cómo llega el modelo a la app:** un bosque de 100 árboles con 113 clases pesa ~330 MB comprimido, y no entra en GitHub ni en Streamlit Cloud.
  `bosque_compacto.py` lo guarda en ~28 MB (por nodo: columna, umbral e hijo derecho; las hojas, de forma dispersa) y da las mismas
  probabilidades que scikit-learn. El preprocesamiento se exporta a `preproceso.json` y la app lo repite con numpy, así que Streamlit no
  depende de la versión de scikit-learn.
* **Casos raros:** los deslizadores van del mínimo al máximo del entrenamiento; `tempo = 0` («sin tempo detectado») se reemplaza por la mediana.
* **Predicciones de «Explorá»:** para las canciones de entrenamiento son fuera de fold; para las de test, las del modelo final.

## Estructura

```
├── build_app.py            ├── docs/index.html          (lo genera build_app.py)
├── entrenar.py             ├── artefactos/              (lo genera entrenar.py)
├── bosque_compacto.py      ├── genreML_tracks_latest.csv
├── modelo_app.py           ├── familias/                (del notebook de la Entrega 3)
├── app_streamlit.py        ├── entrega1_pipeline.py · entrega2_hipotesis.ipynb · entrega3_modelado.ipynb
└── requirements.txt
```
