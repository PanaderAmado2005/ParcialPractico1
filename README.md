# 🐶 Identificador de Razas de Perros

**Autor:** Santiago Niño Amado

## 🌐 Aplicación Web

> **Streamlit App:** [Insertar link aquí]

---

## 📋 Descripción

Proyecto de clasificación de imágenes que entrena una red neuronal convolucional (CNN) para identificar **120 razas de perros** usando el [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). El pipeline completo incluye descarga del dataset, preprocesamiento con bounding boxes, data augmentation y entrenamiento con early stopping.

---

## 📁 Estructura del Proyecto

```
cuadernoParcial.ipynb              # Notebook principal con todo el pipeline
app.py                             # Aplicación web en Streamlit
modelo_razas.keras                 # Mejor modelo guardado automáticamente
/content/stanford_dogs/            # Dataset original (imágenes + anotaciones)
/content/stanford_dogs_processed/  # Imágenes recortadas y redimensionadas
```

---

## ⚙️ Requisitos

```bash
pip install tensorflow pillow numpy matplotlib streamlit
```

> Se recomienda ejecutar el notebook en **Google Colab** con GPU habilitada.

---

## 🚀 Pipeline del Notebook

### 1. Descarga del Dataset
Descarga automática desde Stanford Vision Lab:
- **Imágenes:** ~20,000 fotos de 120 razas
- **Anotaciones XML:** coordenadas del bounding box de cada perro

### 2. Preprocesamiento
- Lectura del bounding box desde los archivos XML
- Recorte de la región del perro en cada imagen
- Redimensionamiento a **100×100 píxeles**
- Validación de bounding boxes inválidos para evitar imágenes negras
- Imágenes guardadas por carpeta de raza en `/content/stanford_dogs_processed/`

### 3. Carga y Normalización
- División **90% entrenamiento / 10% validación** (seed=42)
- Batches de 32 imágenes
- Normalización de píxeles al rango `[0.0, 1.0]`

### 4. Data Augmentation
Aplicada durante el entrenamiento para reducir overfitting:
- Flip horizontal aleatorio
- Rotación ±5%
- Zoom ±5%
- Contraste aleatorio ±10%
- Clip de valores para evitar artefactos

### 5. Arquitectura del Modelo (CNN)

```
Input: (100, 100, 3)
→ Data Augmentation
→ Conv2D(32)  + LeakyReLU + MaxPooling  →  50×50
→ Conv2D(64)  + LeakyReLU + MaxPooling  →  25×25
→ Conv2D(128) + LeakyReLU + MaxPooling  →  12×12
→ Flatten
→ Dense(64) + BatchNormalization
→ Dense(32) + BatchNormalization
→ Dense(120, softmax)
```

> Se usó `LeakyReLU` en lugar de `ReLU` para evitar el problema de neuronas muertas.

### 6. Entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Loss | Sparse Categorical Crossentropy |
| Épocas máximas | 50 |
| Batch size | 32 |
| Early stopping | patience=5 |

El modelo se guarda automáticamente en `/content/modelo_razas.keras` cada vez que mejora el `val_loss`, y se descarga al PC al finalizar el entrenamiento.

### 7. Evaluación
- Gráficas de accuracy y loss por época (entrenamiento vs validación)
- Línea roja marcando el mejor epoch según `val_loss`
- Reporte del mejor accuracy y loss de validación

---

## 🌐 Aplicación Streamlit

Para correr la app localmente:

```bash
streamlit run app.py
```

Asegúrate de tener `modelo_razas.keras` en la misma carpeta que `app.py`.
