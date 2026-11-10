"""El modelo de la app: carga los artefactos que dejó entrenar.py y predice. No depende de scikit-learn.
También agrupa los 113 géneros en las 10 familias de la Entrega 3 (k-means k = 10), sólo para leer mejor el resultado."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bosque_compacto import BosqueCompacto

CARPETA = Path(__file__).parent / "artefactos"        # lo que genera entrenar.py
FAMILIAS = Path(__file__).parent / "familias"         # lo que dejó el notebook de la Entrega 3
MAPA_K10 = "k-means k=10 (con nombres)"
# Nombre que se muestra de cada familia; el código del notebook queda en las tablas de la página.
# K10-05 no recibió nombre en el notebook (ningún género ancla): cuando le pongan uno, cambienlo acá.
NOMBRES_FAMILIA = {
    "POP_ROCK_FOLK": "Pop, rock y folk",
    "ROCK_Y_ELECTRONICA_ENERGICOS": "Rock y electrónica enérgicos",
    "BAILABLE_URBANO_LATINO": "Bailable, urbano y latino",
    "METAL_Y_ELECTRONICA_DURA": "Metal y electrónica dura",
    "TECHNO_MINIMAL": "Techno minimal",
    "CLASICA_AMBIENT": "Clásica y ambient",
    "ACUSTICO_SUAVE": "Acústico suave",
    "SAMBA_PAGODE_MPB": "Samba, pagode y MPB",
    "HABLADO": "Hablado",
    "K10-05": "Electrónica y alternativo",
}


def etiqueta_familia(codigo):
    return NOMBRES_FAMILIA.get(codigo, codigo)


class ModeloGeneros:
    def __init__(self, carpeta=CARPETA):
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / "metadatos.json").read_text(encoding="utf-8"))
        self.pre = json.loads((carpeta / "preproceso.json").read_text(encoding="utf-8"))
        self.rangos = json.loads((carpeta / "rangos.json").read_text(encoding="utf-8"))   # {columna: [mín, máx, mediana]}
        self.bosque = BosqueCompacto.cargar(carpeta / "bosque.npz")
        self.clases = np.array(self.meta["clases"])
        # familias: cada género pertenece a una; la probabilidad de la familia es la suma de la de sus géneros
        mapas = json.loads((FAMILIAS / "genreml_mapas_familias.json").read_text(encoding="utf-8"))["mapas"]
        self.familia_de = mapas[MAPA_K10]
        faltan = set(self.clases) - set(self.familia_de)
        assert not faltan, f"Géneros sin familia: {sorted(faltan)}"
        self.codigos_familia = sorted(set(self.familia_de.values()))
        self._pertenece = np.array([[self.familia_de[g] == f for f in self.codigos_familia] for g in self.clases], dtype=float)
        self.fam_cv = pd.read_csv(FAMILIAS / "genreml_familias_cv.csv")

    def vector(self, v):
        """Repite a mano lo que hacía el ColumnTransformer: estandarizar, tempo 0 -> mediana, one-hot de key y mode."""
        p = self.pre
        numericas = (np.array([v[c] for c in p["numericas"]], dtype=float) - p["media"]) / p["escala"]
        tempo = v["tempo"] if v["tempo"] != 0 else p["tempo_mediana"]        # 0 = «sin tempo detectado»
        tempo = (tempo - p["tempo_media"]) / p["tempo_escala"]
        key = [float(v["key"] == k) for k in p["key"]]
        mode = [float(v["mode"] == m) for m in p["mode"]]
        return np.concatenate([numericas, [tempo], key, mode])

    def probabilidades(self, v):
        """Una probabilidad por género (113), en el orden de self.clases."""
        return self.bosque.predict_proba(self.vector(v)[None, :])[0]

    def probabilidades_familia(self, p):
        """Una probabilidad por familia (10), en el orden de self.codigos_familia: suma de las de sus géneros."""
        return np.asarray(p) @ self._pertenece

    def resumen_familias(self):
        """Métricas de validación cruzada de k = 10 sumando géneros (la forma en que la app calcula la familia)."""
        d = self.fam_cv
        f = d[(d.Objetivo == MAPA_K10) & d.Modo.str.startswith("jer")].iloc[0]
        return {"accuracy": float(f["Accuracy"]), "piso": float(f["Accuracy piso"]), "kappa": float(f["Kappa"]),
                "top2": float(f["Top-2"]), "top3": float(f["Top-3"])}

    def fuera_de_rango(self, v):
        """Columnas cuyo valor el modelo nunca vio en el entrenamiento."""
        return [c for c, (lo, hi, _) in self.rangos.items() if not lo <= v[c] <= hi]
