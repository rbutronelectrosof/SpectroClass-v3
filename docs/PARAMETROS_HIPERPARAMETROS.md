# Parámetros e Hiperparámetros — SpectroClass v3.1

## Distinción clave

- **Hiperparámetros**: controlan *cómo* aprende el modelo. Se ajustan **antes** del entrenamiento para mejorar la precisión (ej: `n_neighbors`, `learning_rate`, `dropout_rate`).
- **Parámetros**: definen la *estructura fija* del sistema, o son aprendidos automáticamente durante el entrenamiento (ej: arquitectura de capas, pesos internos de la red, features de entrada).

---

## 🌿 Árbol de Decisión (`decision_tree`)

| Tipo | Nombre | Valor por defecto | Descripción |
|---|---|---|---|
| Hiperparámetro | `max_depth` | `9` | Profundidad máxima del árbol. Mayor = más complejo, mayor riesgo de overfitting. Rango recomendado: 7–12. |
| Hiperparámetro | `test_size` | `0.20` | Fracción de datos reservada para evaluación. Rango típico: 0.10–0.30. |
| Parámetro | `model_type` | `decision_tree` | Algoritmo base: `decision_tree`, `random_forest` o `gradient_boosting`. |
| Parámetro | `n_estimators` | `100` | Solo para Random Forest / Gradient Boosting: número de árboles del ensemble. |

> **Impacto principal**: `max_depth`. Un árbol demasiado profundo memoriza el catálogo (overfitting); demasiado superficial no captura las diferencias entre tipos (underfitting).

---

## 📍 KNN — K-Nearest Neighbors (`knn`)

| Tipo | Nombre | Valor por defecto | Descripción |
|---|---|---|---|
| Hiperparámetro | `n_neighbors` | `9` | Número de vecinos K a consultar. Valores impares evitan empates. Rango recomendado: 5–15. |
| Hiperparámetro | `weights` | `distance` | Ponderación de vecinos: `uniform` (todos igual) o `distance` (más cercanos pesan más). |
| Hiperparámetro | `metric` | `cosine` | Métrica de distancia: `euclidean`, `cosine`, `manhattan`. `cosine` es la más efectiva para espectros normalizados. |
| Hiperparámetro | `test_size` | `0.20` | Fracción de datos para test. |
| Parámetro | Features de entrada | 17–19 EWs + ratios | Anchos equivalentes de líneas diagnóstico (He II, He I, Hα, Ca II K, etc.) más ratios de líneas. |
| Parámetro | Scaler | `StandardScaler` | Normalización Z-score aplicada **antes** del KNN. Imprescindible: KNN es sensible a la escala de los features. |

> **Impacto principal**: `n_neighbors` y `metric`. Con el catálogo ELODIE actual (~850 espectros), KNN alcanza ~84–87% de accuracy. Es el modelo más robusto del sistema con pocos datos.

---

## 〰️ CNN 1D — Red Convolucional 1D (`cnn_1d`)

### Hiperparámetros de entrenamiento

| Tipo | Nombre | Valor por defecto | Descripción |
|---|---|---|---|
| Hiperparámetro | `epochs` | `50` | Máximo de épocas. EarlyStopping puede detener antes si no hay mejora. |
| Hiperparámetro | `batch_size` | `32` | Muestras por lote en cada paso de gradiente. Menor = más ruido, mejor generalización. |
| Hiperparámetro | `learning_rate` | `0.001` | Tasa de aprendizaje del optimizador Adam. Si el modelo no converge, probar `0.0001`. |
| Hiperparámetro | `dropout_rate` | `0.3` | Fracción de neuronas desactivadas aleatoriamente en cada paso (regularización). Rango: 0.2–0.5. |
| Hiperparámetro | `dense_units` | `128` | Neuronas en la capa densa principal. La segunda densa usa `dense_units / 2 = 64`. |
| Hiperparámetro | `spectrum_length` | `1000` | Todos los espectros se interpolban a este número de puntos antes de entrar a la red. |
| Hiperparámetro | `test_size` | `0.20` | Fracción de datos para test. |

### Hiperparámetros de callbacks

| Tipo | Nombre | Valor por defecto | Descripción |
|---|---|---|---|
| Hiperparámetro | `patience` (EarlyStopping) | `8` | Épocas consecutivas sin mejora en `val_loss` antes de detener el entrenamiento. |
| Hiperparámetro | `factor` (ReduceLROnPlateau) | `0.5` | Factor por el que se multiplica el LR cuando se estanca. LR nuevo = LR × 0.5. |
| Hiperparámetro | `patience` (ReduceLROnPlateau) | `4` | Épocas sin mejora antes de reducir el LR. |
| Hiperparámetro | `min_lr` | `1e-6` | LR mínima permitida. El optimizador no bajará de este valor. |

### Parámetros fijos (arquitectura)

| Tipo | Nombre | Valor | Descripción |
|---|---|---|---|
| Parámetro | Bloque 1 | `Conv1D(32, kernel=7) + BN + MaxPool(2) + Dropout(0.15)` | Detecta patrones locales anchos. |
| Parámetro | Bloque 2 | `Conv1D(64, kernel=5) + BN + MaxPool(2) + Dropout(0.21)` | Patrones de complejidad media. |
| Parámetro | Bloque 3 | `Conv1D(128, kernel=3) + BN + MaxPool(2) + Dropout(0.30)` | Patrones de alto nivel. |
| Parámetro | Capas densas | `Dense(128) + BN + Dropout(0.30) → Dense(64) + Dropout(0.15)` | Clasificación final. |
| Parámetro | Salida | `Dense(n_clases, activation='softmax')` | Una neurona por tipo espectral. |
| Parámetro | Optimizador | `Adam` | Fijo. |
| Parámetro | Loss | `sparse_categorical_crossentropy` | Para clasificación multiclase con enteros. |
| Parámetro | `class_weight` | `balanced` (calculado automáticamente) | Compensa el desbalance del catálogo (O=14, M=27 vs F=349). |

> **Impacto principal**: `learning_rate`, `dropout_rate` y `epochs`. Con el catálogo actual la CNN 1D tiene baja accuracy (~45%) debido al desbalance de clases y pocos datos en tipos raros (O, M). Los pesos de clase balanceados ayudan pero no resuelven el problema de fondo.

---

## 🖼️ CNN 2D — Red Convolucional 2D (`cnn_2d`)

| Tipo | Nombre | Valor por defecto | Descripción |
|---|---|---|---|
| Hiperparámetro | `epochs` | `20` | Máximo de épocas. |
| Hiperparámetro | `batch_size` | `32` | Tamaño del lote. |
| Hiperparámetro | `learning_rate` | `0.001` | Tasa de aprendizaje. |
| Hiperparámetro | `dropout_rate` | `0.3` | Dropout de regularización. |
| Hiperparámetro | `image_size` | `64 × 64 px` | Resolución a la que se redimensionan las imágenes PNG. |
| Parámetro | Formato de entrada | PNG escala de grises | Imagen del gráfico del espectro. |
| Parámetro | Estado | Experimental | Peso 0.00 por defecto. Requiere generar imágenes PNG del catálogo. |

---

## ⚖️ Sistema de Votación Ponderada

Estos no son hiperparámetros de aprendizaje sino **pesos de combinación** que el usuario ajusta manualmente.

| Método | Peso por defecto | Rango útil |
|---|---|---|
| Físico | `0.10` | 0.00–0.20 |
| Árbol de Decisión | `0.40` | 0.20–0.60 |
| Template Matching | `0.10` | 0.00–0.20 |
| KNN | `0.20` | 0.00–0.40 |
| CNN 1D | `0.20` | 0.00–0.40 |
| CNN 2D | `0.00` | 0.00–0.30 |
| **Suma total** | **1.00** | Debe ser exactamente 1.00 |

La clasificación final es el tipo espectral con mayor **suma ponderada de votos**:

```
voto[tipo] += peso_método × (confianza_método / 100.0)
```

---

## Recomendaciones de ajuste

### Si la accuracy del KNN es baja (<80%)
1. Probar `metric = euclidean` en lugar de `cosine`
2. Reducir `n_neighbors` a 5 o aumentar a 11
3. Verificar que el catálogo tiene al menos 20 espectros por clase

### Si la CNN 1D no converge (accuracy estancada)
1. Reducir `learning_rate` a `0.0001`
2. Aumentar `epochs` a 100 y `patience` EarlyStopping a 15
3. Reducir `dropout_rate` a `0.2` si hay underfitting
4. Agregar más datos de los tipos escasos (O, M, B)

### Distribución actual del catálogo ELODIE
| Tipo | Muestras | Estado |
|---|---|---|
| F | ~349 | Abundante |
| K | ~257 | Abundante |
| A | ~139 | Aceptable |
| B | ~64 | Escaso |
| M | ~27 | Muy escaso |
| O | ~14 | Crítico |

Los tipos O y M son los más difíciles de clasificar correctamente con CNN 1D por la escasez de datos. El KNN maneja mejor este desbalance gracias a `weights='distance'` y la métrica coseno.
