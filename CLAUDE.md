# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**SpectroClass v3.1** — automatic stellar spectral classification according to the MK (Morgan–Keenan) system (types O, B, A, F, G, K, M with luminosity classes Ia–V). It combines four independent methods in a weighted voting ensemble and exposes them through a Flask web app.

## Running the app

```bash
# Start the web server (opens http://localhost:5000)
python webapp/app.py

# Windows shortcut
iniciar.bat
```

TensorFlow (CNN-1D support) is optional. The app runs without it using only the physical classifier, decision tree, and KNN.

## Training models

```bash
# Decision tree (scikit-learn, ~85% accuracy)
python src/train_and_validate.py --catalog data/elodie/ --output-dir models/

# KNN (~84%) or CNN-1D (requires tensorflow)
python src/train_neural_models.py --model knn --catalog data/elodie/ --output models/
python src/train_neural_models.py --model cnn_1d --catalog data/elodie/ --epochs 50

# Models can also be trained from the web UI (Herramientas / Redes Neuronales tabs)
```

## Batch / single spectrum processing

```bash
python scripts/procesar_una_estrella.py     # single spectrum → PDF report
python scripts/procesar_lote_estrellas.py   # full directory
```

## Architecture

Classification pipeline (executed for every spectrum):

```
Spectrum (.txt/.fits)
    → normalize_to_continuum()          src/spectral_classification_corrected.py
    → measure_diagnostic_lines()        ~85 lines, EW by trapezoidal integration
    → compute_spectral_ratios()         21 ratios (ionization, temperature, gravity)
    → SpectralValidator.classify()      src/spectral_validation.py
          ├─ Physical classifier  10%   26-node MK decision tree (Gray & Corbally 2009)
          ├─ ML decision tree     40%   models/decision_tree.pkl
          ├─ Template matching    10%   χ² vs reference spectra
          └─ KNN / CNN-1D        40%   models/knn_model.pkl / cnn_model.h5
    → estimate_luminosity_class()       src/luminosity_classification.py
    → MK type + confidence + top-3 alternatives
```

### Key source files

| File | Role |
|------|------|
| `src/spectral_classification_corrected.py` | Physical engine: continuum normalization, EW measurement, 26-node spectral decision tree |
| `src/luminosity_classification.py` | Luminosity class Ia–V from 8 gravity-sensitive line ratios |
| `src/spectral_validation.py` | `SpectralValidator` — loads models, runs all methods, weighted vote |
| `src/neural_classifiers.py` | `NeuralClassifierManager` — inference for KNN and CNN-1D |
| `src/train_and_validate.py` | Train/validate decision tree; writes `models/` artifacts |
| `src/train_neural_models.py` | Train KNN / CNN-1D |
| `webapp/app.py` | Flask server: 20+ REST endpoints, SSE for training progress, FITS extractor |
| `webapp/templates/index.html` | Single-page UI — 7 tabs |
| `webapp/static/script.js` | Interactive classification tree (SVG), spectrum plots, SSE client |

### Data

- `data/elodie/` — 891 labeled spectra from the ELODIE catalog (training)
- `eloidecompleto/` — 1572 spectra including synthetic augmentations (`_synNNN.txt` suffix)
- Spectrum files: two-column CSV (`Longitud_de_onda_A,espectro`), wavelength in Å, space/tab/comma separated, optional header
- Filename convention encodes the label: `HD001835_tipoG3.txt` → type G, subtype G3

### Models directory

| File | Content |
|------|---------|
| `models/decision_tree.pkl` | Trained scikit-learn `DecisionTreeClassifier` |
| `models/knn_model.pkl` + `knn_scaler.pkl` | KNN model and its feature scaler |
| `models/cnn_model.h5` | CNN-1D (TensorFlow, optional) |
| `models/active_models.json` | Which neural models are active and their runtime weights |
| `models/metadata.json` | Training metadata for the decision tree |

### Optional SUPPNet normalizer

`webapp/app.py` tries to import a neural continuum normalizer from a hardcoded path (`~/Desktop/suppnet-main`). The app runs without it; `NORMALIZADOR_DISPONIBLE` will be `False` and the sigma-clipping normalizer in `src/spectral_classification_corrected.py` is used instead.

## Dependencies

```bash
pip install -r requirements.txt          # base (no TF)
pip install tensorflow>=2.10             # add CNN-1D support
```

Python ≥ 3.10 required.
