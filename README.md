# 👥 Segmentación de Clientes con KMeans

Este proyecto permite subir un CSV real, detectar columnas numéricas y entrenar un modelo de **Segmentación de Clientes** usando KMeans.

El sistema asigna automáticamente los 3 segmentos:
- **Cliente FIEL**
- **Cliente NUEVO**
- **Cliente INACTIVO**

## 🔍 Funcionalidades
- Subida de CSV desde Colab.
- Selección automática de columnas relevantes.
- Escalado (StandardScaler) + KMeans.
- Interpretación automática del perfil de cada cluster.
- Interfaz interactiva para clasificar nuevos clientes.

## ▶️ Instalación
```bash
pip install -r requirements.txt
