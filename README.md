<div align="center">

# 🌟 SpectroClass v3.1

### Sistema Automático de Clasificación Espectral Estelar

*Automatic Stellar Spectral Classification System — MK Standard*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Astropy](https://img.shields.io/badge/Astropy-5.0%2B-FF6600?style=for-the-badge)](https://www.astropy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Roberto Butron** · Facultad de Ciencias Exactas y Naturales · UNCuyo · Mendoza, Argentina

[Características](#-características) · [Inicio rápido](#-inicio-rápido) · [Instalación](#-instalación) · [Uso](#-uso) · [Arquitectura](#-arquitectura) · [API](#-api-rest) · [Referencias](#-referencias)

</div>

---

## ¿Qué es SpectroClass?

**SpectroClass** clasifica automáticamente espectros estelares según el sistema **MK (Morgan–Keenan)** —tipos O, B, A, F, G, K, M con clase de luminosidad Ia–V— combinando tres métodos independientes en un sistema de **votación ponderada**:

| Método | Peso | Descripción |
|--------|------|-------------|
| 🔬 Clasificador físico | 10 % | Árbol jerárquico de 26 nodos basado en criterios de Gray & Corbally (2009) |
| 🌿 Árbol de decisión ML | 40 % | scikit-learn entrenado con 856 espectros del catálogo ELODIE (~85 % accuracy) |
| 📋 Template matching | 10 % | Comparación χ² con plantillas de referencia |
| 🧠 KNN / CNN-1D | 40 % | Red neuronal entrenada localmente (KNN ~84 %, CNN-1D ~45 %+) |

El resultado incluye el **tipo MK completo** (ej. `G2V`, `K3III`, `B2Ia`), confianza porcentual y hasta tres alternativas con justificación diagnóstica trazable por línea espectral.

> 📢 Presentado en: *Congreso de Evolución Estelar, Exoplanetas y Dinámica de Sistemas Estelares* — Abril 2026

---

## ✨ Características

### Clasificación espectral
- **Secuencia completa OBAFGKM** — subtipos O2 a M9 con nodos especializados para O3/O4, O tardío y M tardío (VO + CaH)
- **Clase de luminosidad MK** — Ia, Ib, II, III, IV, V mediante 8 indicadores sensibles a gravedad superficial (Sr II/Fe I, Ba II/Fe I, TiO/CaH, Y II/Fe I…)
- **85 líneas diagnóstico** medidas por ancho equivalente (integración trapezoidal, ventanas adaptativas 6–30 Å)
- **21 razones diagnóstico** de ionización (He I/He II, Si III/Si II), temperatura (Ca II K/Hε, Cr I/Fe I) y gravedad
- **Normalización científica** al continuo: sigma-clipping MAD por ventanas, spline cúbico iterativo
- Detección de desplazamiento Doppler y estimación de velocidad radial

### Interfaz web (Flask)
- Carga de archivos `.txt` y `.fits` con calibración WCS automática (`CRVAL1 + (px − CRPIX1) × CDELT1`)
- Visualización interactiva SVG con zoom en líneas diagnóstico y etiquetas escalonadas
- **Árbol interactivo de clasificación** visual — 26 nodos, ~6 preguntas por espectro
- Procesamiento por lotes con exportación a CSV, TXT y PDF
- Extractor FITS → TXT con descarga en ZIP y CSV de metadatos
- Panel de configuración de pesos en tiempo real (sliders + vista previa de distribución)
- Tablas MK completas integradas en la ayuda (tipos O–M, T_eff, log g, luminosidad)

### Modelos incluidos
| Archivo | Método | Accuracy |
|---------|--------|----------|
| `models/decision_tree.pkl` | Árbol de decisión scikit-learn | ~85 % (ver `validation_report.txt`) |
| `models/knn_model.pkl` | K-Nearest Neighbors | ~84 % |
| `models/cnn_model.h5` | CNN-1D TensorFlow *(opcional)* | ~45 %+ |

---

## 🚀 Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/RobertoButron/SpectroClass.git
cd SpectroClass

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar servidor
python webapp/app.py
```

Abrir **http://localhost:5000** en el navegador, cargar un espectro `.txt` o `.fits` y obtener la clasificación en segundos.

**Windows:** ejecutar `INSTALAR.bat` para instalación automática.

---

## 📦 Instalación

### Requisitos
- Python **3.10** o superior
- Git

### Con entorno virtual (recomendado)

```bash
git clone https://github.com/RobertoButron/SpectroClass.git
cd SpectroClass

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Opcional: soporte CNN con TensorFlow
pip install tensorflow>=2.10
```

### Dependencias principales

| Paquete | Versión mínima | Uso |
|---------|---------------|-----|
| `numpy` | ≥ 1.21 | Cómputo numérico |
| `scipy` | ≥ 1.7 | Spline, find_peaks, integración |
| `pandas` | ≥ 1.3 | Gestión de datos tabulares |
| `matplotlib` | ≥ 3.4 | Visualización y exportación PDF |
| `scikit-learn` | ≥ 1.0 | Árbol de decisión ML y KNN |
| `Flask` | ≥ 2.0 | Servidor web y API REST |
| `astropy` | ≥ 5.0 | Lectura de archivos FITS |
| `joblib` | ≥ 1.0 | Serialización de modelos |
| `tensorflow` | ≥ 2.10 | CNN-1D *(opcional)* |

---

## 🔭 Uso

### Aplicación web

```bash
python webapp/app.py
# → http://localhost:5000
```

La interfaz organiza las funciones en 7 pestañas:

| Pestaña | Función |
|---------|---------|
| 🔭 Clasificador Espectral | Carga un espectro → tipo MK + luminosidad + confianza + plot |
| 📊 Análisis Detallado | Tabla de las 85 líneas con EW medidos y calidad |
| 📁 Procesamiento por Lote | Múltiples espectros → CSV / PDF consolidado |
| 🛠️ Herramientas | Entrenamiento del árbol ML + extractor FITS |
| 🧠 Redes Neuronales | Entrenamiento y predicción KNN / CNN-1D |
| 🌿 Árbol Interactivo | Clasificación visual paso a paso (26 nodos) |
| ❓ Ayuda | Guía completa + tablas MK integradas |

### Uso programático (Python)

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
)
from luminosity_classification import estimate_luminosity_class, combine_spectral_and_luminosity

# Cargar espectro (dos columnas: longitud de onda, flujo)
data = np.loadtxt('mi_espectro.txt', delimiter=',', skiprows=1)
wavelengths, flux = data[:, 0], data[:, 1]

# 1. Normalizar al continuo (sigma-clipping MAD + spline cúbico)
flux_norm, continuum = normalize_to_continuum(wavelengths, flux)

# 2. Medir anchos equivalentes (85 líneas diagnóstico)
measurements = measure_diagnostic_lines(wavelengths, flux_norm)

# 3. Clasificar tipo espectral
tipo, subtipo, diagnostics = classify_star_corrected(measurements, wavelengths, flux_norm)

# 4. Estimar clase de luminosidad MK
lum_class = estimate_luminosity_class(measurements, tipo)
mk_full = combine_spectral_and_luminosity(subtipo, lum_class)

print(f"Tipo espectral: {mk_full}")        # Ej: G2V
print(f"Confianza:      {diagnostics['confianza']} %")
```

### Sistema multi-método (votación ponderada)

```python
from spectral_validation import SpectralValidator

validator = SpectralValidator(models_dir='models/')
result = validator.classify(wavelengths, flux)

print(result['tipo_final'])       # Ej: 'G'
print(result['subtipo_final'])    # Ej: 'G2'
print(result['confianza'])        # Ej: 87.4
print(result['alternativas'])     # Top 3 con justificación por línea
```

### Línea de comandos

```bash
# Un espectro con reporte PDF detallado
python scripts/procesar_una_estrella.py

# Directorio completo
python scripts/procesar_lote_estrellas.py
```

---

## 🏗️ Arquitectura

```
Espectro (.txt / .fits)
         │
         ▼
normalize_to_continuum()     ← Sigma-clip MAD por ventanas + spline cúbico iterativo
         │
         ▼
measure_diagnostic_lines()   ← 85 líneas, ventanas adaptativas, calidad OK / BLENDED / …
         │
         ▼
compute_spectral_ratios()    ← 21 razones: ionización, temperatura, gravedad
         │
    ┌────┴──────────────────────────┐
    │                               │
    ▼                               ▼
classify_star_corrected()    SpectralValidator (ensemble)
  (árbol físico 26 nodos)      ├─ 🔬 Físico       (10 %)
                                ├─ 🌿 Árbol ML     (40 %)
                                ├─ 📋 Template     (10 %)
                                └─ 🧠 KNN / CNN    (40 %)
                                         │
                                         ▼
                               estimate_luminosity_class()
                                         │
                                         ▼
                           Tipo MK completo + Confianza + Alternativas
                           Ej: G2V  |  87.4 %  |  [G3V, G1V, G2IV]
```

### Líneas diagnóstico incluidas

| Grupo | Líneas |
|-------|--------|
| Balmer | Hα, Hβ, Hγ, Hδ, Hε (ventanas 16–30 Å) |
| He I | λ3820, 4026, 4121, 4144, 4388, 4471, 4713, 4922, 5876, 6678, 7065 |
| He II | λ4200, 4542, 4686 |
| Si IV / III / II | λ4089, 4116, 4128, 4131, 4553, 4568 |
| N III/V, C III, O II | λ4603–4641, 4647–4652, 4070–4076 |
| Ca II (H&K), Ca I | λ3933, 3968, 4227 |
| Fe I, Fe II, Cr I, Ti II | λ4046, 4144, 4250, 4383, 4957 · λ4254, 4275, 4290 |
| Sr II, Ba II, Y II | λ4077 · λ4554 · λ4376 |
| CH G-band, Mg I b, Mg II | λ4300, 5167 · λ4481 |
| TiO, VO, CaH, MgH | λ4762, 4955, 5167, 5448, 6158, 6651 |

---

## 📁 Estructura del Proyecto

```
SpectroClass/
├── webapp/
│   ├── app.py                       # Servidor Flask + API REST (20+ endpoints)
│   ├── templates/index.html         # UI: 7 pestañas + árbol interactivo
│   └── static/
│       ├── script.js                # Lógica de árbol, gráficos SVG, SSE
│       └── style.css                # Diseño responsivo + badges MK
│
├── src/
│   ├── spectral_classification_corrected.py  # Motor físico (85 líneas, 21 razones)
│   ├── luminosity_classification.py          # Clase de luminosidad MK (Ia–V)
│   ├── spectral_validation.py                # Ensemble multi-método + votación
│   ├── neural_classifiers.py                 # Inferencia KNN / CNN-1D
│   ├── train_neural_models.py                # Entrenamiento KNN / CNN
│   └── train_and_validate.py                 # Entrenamiento árbol ML + reporte
│
├── scripts/
│   ├── procesar_una_estrella.py     # Clasificación individual con PDF
│   └── procesar_lote_estrellas.py   # Procesamiento por lotes
│
├── data/
│   └── elodie/                      # Catálogo ELODIE (856 espectros etiquetados)
│
├── models/
│   ├── decision_tree.pkl            # Árbol ML entrenado
│   ├── knn_model.pkl                # Modelo KNN
│   ├── cnn_model.h5                 # CNN-1D (opcional, requiere TensorFlow)
│   ├── confusion_matrix.png         # Matriz de confusión
│   └── validation_report.txt        # Métricas por tipo espectral
│
├── docs/                            # Documentación técnica
├── requirements.txt                 # Dependencias base
├── INSTALAR.bat                     # Instalador automático Windows
└── LICENSE                          # MIT
```

---

## 📡 API REST

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/upload` | Clasificar espectro(s) → tipo MK + luminosidad + plot |
| `POST` | `/upload_single` | Guardar archivo en servidor para uso posterior |
| `POST` | `/api/luminosity` | Estimar clase de luminosidad independientemente |
| `POST` | `/fits_extract_one` | Convertir un FITS a TXT con metadatos |
| `POST` | `/fits_extract_batch` | Convertir lote de FITS → ZIP con TXT + CSV |
| `POST` | `/train_model_stream` | Entrenar árbol ML con progreso en tiempo real (SSE) |
| `POST` | `/train_neural_stream` | Entrenar KNN/CNN con progreso en tiempo real (SSE) |
| `POST` | `/export_csv` | Exportar resultados a CSV |
| `GET`  | `/get_metrics` | Métricas del árbol ML activo |
| `GET`  | `/neural_metrics` | Métricas de modelos KNN/CNN |
| `GET`  | `/list_catalogs` | Listar catálogos disponibles |
| `GET`  | `/confusion_matrix` | Imagen de la matriz de confusión |
| `GET`  | `/health` | Estado del servidor |

### Ejemplo de respuesta `/upload`

```json
{
  "tipo_clasificado": "G",
  "subtipo": "G2",
  "luminosity_class": "V",
  "mk_full": "G2V",
  "confianza": 87.4,
  "alternativas": [
    { "tipo": "G", "confianza": 87.4, "justificacion": "ML (91% prob.), KNN (85%)" },
    { "tipo": "F", "confianza": 61.2, "justificacion": "Clasificador físico" },
    { "tipo": "K", "confianza": 38.1, "justificacion": "Template (χ²=1.8)" }
  ],
  "n_lineas": 23,
  "rango_lambda": [3900.0, 6800.0]
}
```

---

## 📋 Formato de Datos

### Archivos `.txt`

```
wavelength,flux
3900.0,0.9234
3900.5,0.9245
4000.0,0.8721
```

- **Columna 1:** longitud de onda en Ångströms
- **Columna 2:** flujo (normalizado o en cuentas)
- Separador: coma, tabulador o espacio (detección automática)
- El encabezado es opcional (se detecta automáticamente)

### Archivos `.fits`

Requiere cabecera WCS: `CRVAL1`, `CDELT1`, `CRPIX1`

```
λ(px) = CRVAL1 + (px − CRPIX1) × CDELT1
```

---

## 🧠 Entrenamiento de Modelos

### Desde la interfaz web

Pestañas **Herramientas** y **Redes Neuronales** — progreso en tiempo real via Server-Sent Events.

### Desde línea de comandos

```bash
# Árbol de decisión ML
python src/train_and_validate.py --catalog data/elodie/ --output models/ --model decision_tree

# KNN (recomendado para empezar, ~84 % accuracy)
python src/train_neural_models.py --model knn --catalog data/elodie/ --output models/

# CNN-1D (requiere TensorFlow)
python src/train_neural_models.py --model cnn_1d --catalog data/elodie/ --epochs 50
```

Los modelos se guardan automáticamente en `models/` y quedan activos para la siguiente clasificación.

---

## 🌿 Árbol Interactivo — 26 Nodos

Replica el flujo de clasificación visual clásica MK:

```
inicio
├── ¿He II presente? (λ4542, λ4686)
│   ├── SÍ → Tipo O
│   │   ├── ¿N V 4603 fuerte? → O3/O4
│   │   ├── ¿He I > He II? → O7-O9
│   │   └── ¿Si III ~ He II? → O9.5
│   └── NO → ¿He I sin He II?
│       ├── SÍ → Tipo B
│       │   ├── ¿Si IV ~ Si III? → B0
│       │   ├── ¿Si III > Si II? → B1-B3
│       │   └── ¿Mg II > He I? → B8-B9
│       └── NO → ¿Balmer máximo?
│           ├── SÍ → Tipo A  (Ca II K vs Hε)
│           └── NO → ¿Ca II K ~ Hε?
│               ├── SÍ → Tipo F  (CH G-band 4300)
│               └── NO → ¿muchos metales?
│                   ├── Tipo G  (Cr I/Fe I, Y II 4376)
│                   ├── Tipo K  (Ca I > Fe I, Na I D)
│                   └── Tipo M  (TiO + VO + CaH + MgH)
```

---

## 📊 Catálogos Soportados

| Catálogo | Espectros | Uso |
|----------|-----------|-----|
| ELODIE (Prugniel & Soubiran 2001) | 856 | Entrenamiento ML y KNN |
| Espectros de referencia propios | ~50 | Template matching |
| Cualquier directorio con `_tipoXN.txt` | Ilimitado | Entrenamiento personalizado |

El sistema reconoce el tipo espectral directamente desde el nombre del archivo:
`HD001835_tipoG3.txt` → tipo G, subtipo G3.

---

## 📖 Referencias

- Gray, R. O., & Corbally, C. J. (2009). *Stellar Spectral Classification*. Princeton University Press.
- Morgan, W. W., Keenan, P. C., & Kellman, E. (1943). *An Atlas of Stellar Spectra*. University of Chicago Press.
- Prugniel, P., & Soubiran, C. (2001). *A database of high and medium-resolution stellar spectra* (ELODIE). A&A, 369, 1048.
- Sota, A., et al. (2014). *The Galactic O-Star Spectroscopic Survey (GOSSS) II*. ApJS, 211, 10.
- Kirkpatrick, J. D., Reid, I. N., & Liebert, J. (1999). *Dwarfs cooler than M*. ApJ, 519, 802.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para proponer cambios:

1. Haz un fork del repositorio
2. Crea una rama: `git checkout -b feature/descripcion`
3. Haz commit de los cambios: `git commit -m 'Descripción clara del cambio'`
4. Abre un Pull Request

### Áreas de mejora sugeridas

- [ ] Soporte para espectros de baja resolución (R < 1000)
- [ ] Clasificación de estrellas peculiares (Am, Ap, Ba, Be)
- [ ] Exportación a VO-Table (formato estándar IVOA)
- [ ] Tests unitarios para los módulos `src/`
- [ ] Integración con bases de datos online (SIMBAD, VizieR)

---

## 📄 Licencia

Distribuido bajo licencia **MIT**. Ver archivo [LICENSE](LICENSE) para más información.

---

<div align="center">

Desarrollado en la **Facultad de Ciencias Exactas y Naturales — UNCuyo**
Mendoza, Argentina · 2025–2026

*"Classification is the art of making distinctions."* — W. W. Morgan

</div>
