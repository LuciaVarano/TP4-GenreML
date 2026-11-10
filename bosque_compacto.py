"""Random Forest en formato compacto: mismas probabilidades que scikit-learn, ~40 MB en vez de ~330 MB.
Cada árbol guarda sólo (columna, umbral, hijo derecho) por nodo; el hijo izquierdo es siempre nodo+1
(scikit-learn numera en profundidad). Las hojas guardan su distribución de clases de forma dispersa."""
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier


class BosqueCompacto:
    def __init__(self, n_clases):
        self.n_clases, self.arboles = n_clases, []

    def agregar(self, arbol):
        t = arbol.tree_
        izq = t.children_left
        interno = izq != -1
        assert (izq[interno] == np.arange(len(izq))[interno] + 1).all()   # el supuesto «izquierdo = nodo+1»
        v = t.value[:, 0, :]
        v = v / v.sum(axis=1, keepdims=True)
        hojas = sparse.csr_matrix(np.where(interno[:, None], 0.0, v).astype(np.float32))
        self.arboles.append((t.feature.astype(np.int8), t.threshold.astype(np.float64),
                             t.children_right.astype(np.int32), hojas))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        n, P = len(X), np.zeros((len(X), self.n_clases))
        for col, umbral, der, hojas in self.arboles:
            nodo, activos = np.zeros(n, np.int32), np.arange(n)
            while activos.size:
                nd = nodo[activos]
                sigue = col[nd] >= 0                       # -2 marca una hoja
                activos, nd = activos[sigue], nd[sigue]
                if not activos.size:
                    break
                izquierda = X[activos, col[nd]] <= umbral[nd]
                nodo[activos] = np.where(izquierda, nd + 1, der[nd])
            P += hojas[nodo].toarray()
        return P / len(self.arboles)

    def guardar(self, ruta):
        n_nodos = np.array([len(a[0]) for a in self.arboles])
        np.savez_compressed(ruta, n_clases=self.n_clases, n_nodos=n_nodos,
            col=np.concatenate([a[0] for a in self.arboles]), umbral=np.concatenate([a[1] for a in self.arboles]),
            der=np.concatenate([a[2] for a in self.arboles]),
            datos=np.concatenate([a[3].data for a in self.arboles]),
            indices=np.concatenate([a[3].indices for a in self.arboles]).astype(np.uint8),
            filas=np.concatenate([np.diff(a[3].indptr) for a in self.arboles]).astype(np.uint8))

    @classmethod
    def cargar(cls, ruta):
        z = np.load(ruta)
        # Cada z["..."] descomprime el arreglo entero: se lee UNA vez. Y cada rebanada se COPIA: una rebanada sin copiar
        # es una vista que mantiene vivo el arreglo completo, y con 100 árboles la memoria se multiplica por 100.
        col, umbral, der = z["col"], z["umbral"], z["der"]
        datos, indices, filas_todas = z["datos"], z["indices"], z["filas"]
        b, i, k = cls(int(z["n_clases"])), 0, 0
        for n in z["n_nodos"]:
            filas = filas_todas[i:i + n].astype(np.int64)
            ptr = np.concatenate([[0], np.cumsum(filas)])
            m = sparse.csr_matrix((datos[k:k + ptr[-1]].copy(), indices[k:k + ptr[-1]].astype(np.int32), ptr),
                                  shape=(n, b.n_clases))
            b.arboles.append((col[i:i + n].copy(), umbral[i:i + n].copy(), der[i:i + n].copy(), m))
            i, k = i + n, k + ptr[-1]
        return b


def ajustar_bosque(X, y, n_arboles=100, bloque=10, **params):
    """Entrena el bosque de a bloques de árboles (warm_start) y compacta cada bloque: la memoria no se dispara.
    Con la misma semilla da los mismos árboles que un solo fit de n_arboles."""
    rf = RandomForestClassifier(n_estimators=bloque, warm_start=True, **params)
    bosque, imp = BosqueCompacto(len(np.unique(y))), np.zeros(X.shape[1])
    for k in range(bloque, n_arboles + 1, bloque):
        rf.n_estimators = k
        rf.fit(X, y)
        for j in range(k - bloque, k):
            bosque.agregar(rf.estimators_[j])
            imp += rf.estimators_[j].feature_importances_
            rf.estimators_[j] = None                       # liberar el árbol de scikit-learn
    return bosque, imp / n_arboles
