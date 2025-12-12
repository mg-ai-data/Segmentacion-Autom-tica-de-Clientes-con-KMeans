1.

!pip install gradio scikit-learn pandas

import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# 1) SUBIR ARCHIVO CSV
# -------------------------------------------------------
from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print("Columnas del dataset:", df.columns.tolist())
df.head()

---

2.

!pip install gradio scikit-learn pandas

import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# 1) SUBIR CSV
# -------------------------------------------------------
from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# Tomar solo columnas numéricas
numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(numericas) < 3:
    raise Exception("Tu CSV necesita al menos 3 columnas numéricas para segmentar.")

print("Columnas numéricas detectadas:", numericas)

# -------------------------------------------------------
# 2) ENTRENAR MODELO (KMeans + escalado)
# -------------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[numericas])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data_scaled)

# -------------------------------------------------------
# 3) IDENTIFICAR AUTOMÁTICAMENTE FIEL / NUEVO / INACTIVO
# -------------------------------------------------------
# Criterio: mucha actividad = compras altas, dias_ultima_compra bajos

# Identificar si existe columna de "días desde última compra"
dias_cols = [c for c in numericas if "dias" in c.lower()]
compras_cols = [c for c in numericas if "compra" in c.lower() or "frecuencia" in c.lower()]

# Si no encuentra nombres “obvios”, usa la última columna como días
if len(dias_cols) == 0:
    dias_cols = [numericas[-1]]
if len(compras_cols) == 0:
    compras_cols = [numericas[0]]

dias_col = dias_cols[0]
compras_col = compras_cols[0]

cluster_scores = []

for c in range(3):
    subset = df[kmeans.labels_ == c]
    score = subset[compras_col].mean() - subset[dias_col].mean()
    cluster_scores.append((c, score))

cluster_scores.sort(key=lambda x: x[1], reverse=True)
ordered = [c for c, s in cluster_scores]

cluster_names = {
    ordered[0]: "Cliente FIEL",
    ordered[1]: "Cliente NUEVO",
    ordered[2]: "Cliente INACTIVO"
}

print("Asignación automática de clusters:", cluster_names)

# -------------------------------------------------------
# 4) FUNCIÓN PARA CLASIFICAR NUEVOS CLIENTES
# -------------------------------------------------------
def clasificar(*args):
    X = pd.DataFrame([list(args)], columns=numericas)
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]
    return cluster_names[cluster]

# -------------------------------------------------------
# 5) INTERFAZ GRADIO
# -------------------------------------------------------
inputs = [gr.Number(label=col) for col in numericas]

iface = gr.Interface(
    fn=clasificar,
    inputs=inputs,
    outputs="text",
    title="Segmentación de Clientes con tu CSV",
    description="Subiste un dataset real. Probá nuevos valores y obtené: Fiel / Nuevo / Inactivo."
)

iface.launch()
