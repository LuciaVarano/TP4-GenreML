"""Reproduce el modelo elegido en la Entrega 3 y exporta lo que usa la app. Misma semilla, misma partición."""
import json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bosque_compacto import ajustar_bosque

SEMILLA = 42
from pathlib import Path
Path("artefactos").mkdir(exist_ok=True)
NUM = ["duration_ms", "danceability", "energy", "loudness", "speechiness", "acousticness",
       "instrumentalness", "liveness", "valence", "tempo"]
CAT = ["key", "mode", "time_signature"]
FEATURES_ELEGIDAS = [c for c in NUM + CAT if c != "time_signature"]      # decisión de la Entrega 3 (Parte 6)
PARAMS = dict(max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=1)

df = pd.read_csv("genreML_tracks_latest.csv")
X, y = df[NUM + CAT], df["track_genre"]
clases = np.array(sorted(y.unique())); yi = pd.Categorical(y, categories=clases).codes
huella = pd.util.hash_pandas_object(X, index=False).to_numpy()          # el mismo audio no cruza train/test
idx_tr, idx_te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y, groups=huella))
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X.iloc[idx_tr], y.iloc[idx_tr], groups=huella[idx_tr]))

def preparador():
    num = [c for c in FEATURES_ELEGIDAS if c in NUM and c != "tempo"]
    return ColumnTransformer([
        ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), num),
        ("tempo", Pipeline([("imputar_ceros", SimpleImputer(missing_values=0, strategy="median")), ("escalar", StandardScaler())]), ["tempo"]),
        ("categoricas", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["key", "mode"])])

Xtr, ytr = X.iloc[idx_tr][FEATURES_ELEGIDAS], yi[idx_tr]
# 1) validación cruzada: la predicción «fuera de fold» de cada canción de entrenamiento
oof_p, f1_folds, t0 = np.zeros((len(idx_tr), len(clases)), np.float32), [], time.time()
for k, (a, b) in enumerate(folds):
    prep = preparador().fit(Xtr.iloc[a])
    bosque, _ = ajustar_bosque(prep.transform(Xtr.iloc[a]), ytr[a], 100, 10, **PARAMS)
    oof_p[b] = bosque.predict_proba(prep.transform(Xtr.iloc[b]))
    f1_folds.append(f1_score(ytr[b], oof_p[b].argmax(1), average="macro", zero_division=0))
    print(f"fold {k+1}: f1 macro {f1_folds[-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# 2) modelo final: todo el entrenamiento
prep = preparador().fit(Xtr)
bosque, imp = ajustar_bosque(prep.transform(Xtr), ytr, 100, 10, **PARAMS)
bosque.guardar("artefactos/bosque.npz")
# lo que necesita la app para predecir SIN scikit-learn: el preprocesamiento como números y los rangos vistos
num, tem, cat = (prep.named_transformers_[k] for k in ("numericas", "tempo", "categoricas"))
json.dump({"numericas": [c for c in FEATURES_ELEGIDAS if c in NUM and c != "tempo"],
           "media": num["escalar"].mean_.tolist(), "escala": num["escalar"].scale_.tolist(),
           "tempo_mediana": float(tem["imputar_ceros"].statistics_[0]),
           "tempo_media": float(tem["escalar"].mean_[0]), "tempo_escala": float(tem["escalar"].scale_[0]),
           "key": cat["onehot"].categories_[0].tolist(), "mode": cat["onehot"].categories_[1].tolist()},
          open("artefactos/preproceso.json", "w"), indent=1)
json.dump({c: [float(Xtr[c].min()), float(Xtr[c].max()), float(Xtr[c].median())] for c in FEATURES_ELEGIDAS},
          open("artefactos/rangos.json", "w"), indent=1)
Xte = X.iloc[idx_te][FEATURES_ELEGIDAS]; yte = yi[idx_te]
pte = bosque.predict_proba(prep.transform(Xte))

# 3) test, una vez
res = {"f1_macro_cv": float(np.mean(f1_folds)), "f1_cv_desvio": float(np.std(f1_folds, ddof=1)), "f1_folds": [float(v) for v in f1_folds],
       "f1_macro_test": float(f1_score(yte, pte.argmax(1), average="macro", zero_division=0)),
       "accuracy_test": float(accuracy_score(yte, pte.argmax(1))),
       "top3_test": float(top_k_accuracy_score(yte, pte, k=3, labels=range(len(clases)))),
       "top5_test": float(top_k_accuracy_score(yte, pte, k=5, labels=range(len(clases)))),
       "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "features": FEATURES_ELEGIDAS, "clases": clases.tolist(),
       "hiperparametros": {k: v for k, v in PARAMS.items() if k != "n_jobs"} | {"n_estimators": 100}, "semilla": SEMILLA,
       "importancia_impureza": dict(zip(prep.get_feature_names_out().tolist(), imp.round(4).tolist()))}
json.dump(res, open("artefactos/metadatos.json", "w"), ensure_ascii=False, indent=1)

rep = pd.DataFrame(classification_report(yte, pte.argmax(1), labels=range(len(clases)), target_names=clases, output_dict=True, zero_division=0)).T.iloc[:-3]
rep.round(4).to_csv("artefactos/reporte_clases.csv")
c = pd.crosstab(clases[yte], clases[pte.argmax(1)]).stack().rename("errores").reset_index(); c.columns = ["real", "predicho", "errores"]
c[c.real != c.predicho].sort_values("errores", ascending=False).head(40).to_csv("artefactos/confusiones.csv", index=False)

# 4) una fila por canción: qué dijo el modelo sin haberla visto (fuera de fold en train, test en test)
P = np.vstack([oof_p, pte]); orden = np.concatenate([idx_tr, idx_te])
top3 = np.argsort(-P, axis=1)[:, :3]
out = pd.DataFrame({"track_id": df.track_id.to_numpy()[orden], "genero": y.to_numpy()[orden],
                    "particion": ["entrenamiento"] * len(idx_tr) + ["test"] * len(idx_te),
                    "pred1": clases[top3[:, 0]], "p1": P.max(1).round(3), "pred2": clases[top3[:, 1]], "pred3": clases[top3[:, 2]]})
out.to_csv("artefactos/predicciones.csv.gz", index=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("clases", "importancia_impureza", "features")}, indent=1)); print("LISTO")
