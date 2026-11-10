"""El modelo de la app: carga los artefactos que dejó entrenar.py y predice. No depende de scikit-learn."""
import json
from pathlib import Path

import numpy as np

from bosque_compacto import BosqueCompacto

CARPETA = Path(__file__).parent / "artefactos"


class ModeloGeneros:
    def __init__(self, carpeta=CARPETA):
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / "metadatos.json").read_text(encoding="utf-8"))
        self.pre = json.loads((carpeta / "preproceso.json").read_text(encoding="utf-8"))
        self.rangos = json.loads((carpeta / "rangos.json").read_text(encoding="utf-8"))   # {columna: [mín, máx, mediana]}
        self.bosque = BosqueCompacto.cargar(carpeta / "bosque.npz")
        self.clases = np.array(self.meta["clases"])

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

    def fuera_de_rango(self, v):
        """Columnas cuyo valor el modelo nunca vio en el entrenamiento."""
        return [c for c, (lo, hi, _) in self.rangos.items() if not lo <= v[c] <= hi]
