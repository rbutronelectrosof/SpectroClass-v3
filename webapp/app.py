#!/usr/bin/env python3
"""
APLICACIÓN WEB - Clasificación Espectral Interactiva
====================================================

Permite subir espectros (.txt o .fits) y obtener:
- Clasificación automática
- Visualización con zoom en líneas diagnóstico
- Valores característicos (anchos equivalentes)
- Exportación de resultados en PDF/CSV

Uso:
    python app.py

Luego abrir navegador en: http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import pandas as pd
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime

# ── Módulo de normalización de espectros crudos ────────────────────────────
_NORMALIZADOR_PATH = r"C:\Users\Eduardo\Desktop\suppnet-main"
_normalizador_nn = None          # Se carga una vez al arranque
NORMALIZADOR_DISPONIBLE = False
_norm_cache = {}                 # {spectrum_id: {wave, flux, continuum, continuum_std}}

try:
    import sys as _sys
    if _NORMALIZADOR_PATH not in _sys.path:
        _sys.path.insert(0, _NORMALIZADOR_PATH)

    # ── Parche de compatibilidad TF 2.13+ / Keras 3 ──────────────────────
    # En versiones nuevas de Keras, tf.reshape() no puede recibir un
    # KerasTensor directamente dentro del grafo funcional. Se reemplaza la
    # función UpSampling1D_layers del módulo original por una versión que usa
    # capas Keras (Lambda + expand_dims/squeeze) en lugar de tf.reshape().
    import suppnet.SUPPNet as _suppnet_mod
    import tensorflow as _tf

    def _UpSampling1D_layers_compat(inputs, size=2):
        """Versión compatible con Keras 3 / TF ≥ 2.13."""
        x = _tf.keras.layers.Lambda(
            lambda t: _tf.expand_dims(t, axis=2))(inputs)
        x = _tf.keras.layers.UpSampling2D(
            size=(size, 1), data_format=None, interpolation="bilinear")(x)
        x = _tf.keras.layers.Lambda(
            lambda t: _tf.squeeze(t, axis=2))(x)
        return x

    _suppnet_mod.UpSampling1D_layers = _UpSampling1D_layers_compat

    # ── Parche de carga de pesos para Keras 3 ────────────────────────────
    # Keras 3 ya no acepta checkpoints TF2 en model.load_weights().
    # Se reemplaza get_suppnet_model por una versión que usa
    # tf.train.load_checkpoint y asigna los tensores variable a variable.
    import os as _os

    def _load_weights_keras3(model, path):
        """Carga un checkpoint TF2 en un modelo Keras 3 variable por variable.

        Estrategias en orden de preferencia:
          1. layer_with_weights-N/vars/M  — formato estándar TF2 / Keras ≥2.12
          2. nombre de variable Keras2     — formato antiguo conv1d/kernel/…
          3. emparejamiento por posición+forma como último recurso
        """
        reader   = _tf.train.load_checkpoint(path)
        ckpt_map = reader.get_variable_to_shape_map()

        loaded = 0

        # ── Estrategia 1: layer_with_weights-N/vars/M ─────────────────────
        weight_layers = [l for l in model.layers if l.weights]
        for layer_idx, layer in enumerate(weight_layers):
            for var_idx, var in enumerate(layer.weights):
                key = (f'layer_with_weights-{layer_idx}'
                       f'/vars/{var_idx}/.ATTRIBUTES/VARIABLE_VALUE')
                if key in ckpt_map:
                    tensor = reader.get_tensor(key)
                    if list(tensor.shape) == list(var.shape):
                        var.assign(tensor)
                        loaded += 1

        # ── Estrategia 2: nombre de variable Keras 2 ──────────────────────
        if loaded == 0:
            ckpt_keys = set(ckpt_map.keys())
            for var in model.variables:
                base = var.name.split(':')[0]
                key  = base + '/.ATTRIBUTES/VARIABLE_VALUE'
                if key in ckpt_keys:
                    tensor = reader.get_tensor(key)
                    if list(tensor.shape) == list(var.shape):
                        var.assign(tensor)
                        loaded += 1

        # ── Estrategia 3: posición + forma ────────────────────────────────
        if loaded == 0:
            weight_keys = sorted(
                k for k in ckpt_map if '.ATTRIBUTES/VARIABLE_VALUE' in k)
            model_vars  = [v for l in weight_layers for v in l.weights]
            for var, key in zip(model_vars, weight_keys):
                tensor = reader.get_tensor(key)
                if list(tensor.shape) == list(var.shape):
                    try:
                        var.assign(tensor)
                        loaded += 1
                    except Exception:
                        pass

        print(f"[Normalización] Pesos cargados: {loaded}/{len(model.variables)}")
        if loaded == 0:
            print("[Normalización] AVISO: ningún peso cargado. "
                  "Claves disponibles en checkpoint (primeras 5):")
            for k in list(ckpt_map)[:5]:
                print(f"  {k}")
            print("[Normalización] Variables del modelo (primeras 5):")
            for v in list(model.variables)[:5]:
                print(f"  {v.name}  shape={v.shape}")
        return loaded

    def _get_suppnet_model_compat(norm_only=True, which_weights="active"):
        """Versión de get_suppnet_model compatible con Keras 3."""
        _tf.keras.backend.clear_session()
        print("Construyendo modelo de normalización…")
        model = _suppnet_mod.create_SUPPNet_model(input_shape=(8192, 1))
        print("Modelo construido. Cargando pesos…")

        suppnet_dir = _os.path.join(_NORMALIZADOR_PATH, "suppnet")
        weight_map = {
            "active":   "supp_weights/SUPPNet_active",
            "synth":    "supp_weights/SUPPNet_synth",
            "emission": "supp_weights/SUPPNet_18_powr",
        }
        rel_path = weight_map.get(which_weights, "supp_weights/SUPPNet_active")
        full_path = _os.path.join(suppnet_dir, rel_path)

        _load_weights_keras3(model, full_path)
        print("Pesos cargados correctamente.")
        return _suppnet_mod.modelWrapper(model, norm_only=norm_only)

    _suppnet_mod.get_suppnet_model = _get_suppnet_model_compat

    # NN_utility.py importa get_suppnet_model en su propio namespace al
    # cargarse (via suppnet/__init__.py). Hay que parchear esa referencia
    # local también, o seguirá usando la función original.
    import suppnet.NN_utility as _nn_utility_mod
    _nn_utility_mod.get_suppnet_model = _get_suppnet_model_compat
    # ─────────────────────────────────────────────────────────────────────

    from suppnet.NN_utility import get_suppnet, get_smoothed_continuum
    NORMALIZADOR_DISPONIBLE = True
except ImportError:
    pass
# ───────────────────────────────────────────────────────────────────────────

# Importar módulos de clasificación (desde src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
    plot_spectrum_corrected,
    SPECTRAL_LINES
)
try:
    from luminosity_classification import (
        estimate_luminosity_class,
        combine_spectral_and_luminosity,
    )
    _LUM_AVAILABLE = True
except ImportError:
    _LUM_AVAILABLE = False


def convert_numpy_types(obj):
    """
    Convierte tipos numpy a tipos Python nativos para serialización JSON.
    Maneja recursivamente diccionarios y listas.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj

# Importar validador multi-método (NUEVO)
try:
    from spectral_validation import SpectralValidator
    MULTI_METHOD_AVAILABLE = True
except ImportError:
    MULTI_METHOD_AVAILABLE = False
    print("[!] spectral_validation.py no disponible. Solo se usara el clasificador fisico.")

# Configuración de Flask
app = Flask(__name__)
# Usar rutas absolutas para que send_file y os.path coincidan
_webapp_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER']  = os.path.join(_webapp_dir, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(_webapp_dir, 'results')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'fits', 'fit'}

# Crear directorios si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_spectrum_file(filepath):
    """
    Carga un espectro desde archivo .txt o .fits

    Returns
    -------
    wavelengths, flux, error, metadata
    """
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'txt':
        try:
            # Detectar delimitador automáticamente (probar diferentes encodings)
            first_lines = []
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        first_lines = [f.readline() for _ in range(5)]
                    break
                except:
                    continue

            # Buscar línea con datos numéricos
            data_line = None
            delimiter = None  # Por defecto: whitespace
            for line in first_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Verificar si tiene números
                    parts_tab = line.split('\t')
                    parts_comma = line.split(',')
                    parts_space = line.split()

                    # Intentar detectar delimitador
                    if len(parts_tab) >= 2:
                        try:
                            float(parts_tab[0])
                            data_line = line
                            delimiter = '\t'
                            break
                        except:
                            pass
                    if len(parts_comma) >= 2:
                        try:
                            float(parts_comma[0])
                            data_line = line
                            delimiter = ','
                            break
                        except:
                            pass
                    if len(parts_space) >= 2:
                        try:
                            float(parts_space[0])
                            data_line = line
                            delimiter = None  # whitespace
                            break
                        except:
                            pass

            # Cargar datos con delimitador detectado
            try:
                if delimiter == '\t':
                    data = np.loadtxt(filepath, delimiter='\t', comments='#', encoding='latin-1')
                elif delimiter == ',':
                    # Intentar con skiprows=1 primero (tiene header)
                    try:
                        data = np.loadtxt(filepath, delimiter=',', skiprows=1, encoding='latin-1')
                    except:
                        data = np.loadtxt(filepath, delimiter=',', comments='#', encoding='latin-1')
                else:
                    # Whitespace delimited
                    data = np.loadtxt(filepath, comments='#', encoding='latin-1')
            except Exception as load_error:
                # Fallback: intentar con pandas que es más flexible
                import pandas as pd
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', comment='#', header=None, encoding='latin-1')
                    data = df.values
                except:
                    raise load_error

            wavelengths = data[:, 0]
            flux = data[:, 1]

            # Extraer tipo original del nombre si está presente (catálogo ELODIE)
            filename = os.path.basename(filepath)
            tipo_original = ''   # vacío si no hay tipo en el nombre
            if '_tipo' in filename.lower():
                parts = filename.lower().split('_tipo')
                if len(parts) > 1:
                    tipo_original = parts[1].replace('.txt', '').upper()

            metadata = {
                'objeto': filename.split('_')[0],
                'tipo_original': tipo_original  # '' si no viene en el nombre
            }
            return wavelengths, flux, None, metadata
        except Exception as e:
            return None, None, f"Error al leer archivo TXT: {str(e)}", None

    elif ext in ['fits', 'fit']:
        try:
            from astropy.io import fits
            hdul = fits.open(filepath)
            data = hdul[0].data
            header = hdul[0].header
            hdul.close()

            # Calcular eje de longitud de onda según cabecera FITS WCS (World Coordinate System)
            # Fórmula estándar: λ(px) = CRVAL1 + (px - CRPIX1) × CDELT1
            #   CRVAL1 = longitud de onda del píxel de referencia (en Å)
            #   CRPIX1 = número del píxel de referencia (generalmente 1)
            #   CDELT1 = dispersión en Å/píxel
            crval1 = header.get('CRVAL1', 0)
            crpix1 = header.get('CRPIX1', 1)
            cdelt1 = header.get('CDELT1', 1)

            num_pixels = len(data)
            pixels = np.arange(1, num_pixels + 1)
            wavelengths = crval1 + (pixels - crpix1) * cdelt1

            # Cabecera FITS completa (todas las tarjetas)
            fits_header = []
            for card in header.cards:
                kw = str(card.keyword).strip()
                if not kw:
                    continue
                fits_header.append({
                    'clave':      kw,
                    'valor':      str(card.value).strip(),
                    'comentario': str(card.comment).strip()
                })

            # Resumen estructurado de campos clave para display rápido
            # Grupos: obs=Observación, spec=Espectro, est=Estelar, inst=Instrumento, astr=Astrometría
            _FITS_LABELS = {
                'OBJECT':    ('Objeto',              'obs'),
                'DATE-OBS':  ('Fecha observación',   'obs'),
                'DATE':      ('Fecha',               'obs'),
                'EXPTIME':   ('T. exposición (s)',   'obs'),
                'RA':        ('AR (α)',               'obs'),
                'DEC':       ('Dec (δ)',              'obs'),
                'AIRMASS':   ('Masa de aire',         'obs'),
                'TELESCOP':  ('Telescopio',           'obs'),
                'INSTRUME':  ('Instrumento',          'obs'),
                'OBSERVER':  ('Observador',           'obs'),
                'OBSERVAT':  ('Observatorio',         'obs'),
                'SITENAME':  ('Sitio',                'obs'),
                'SPTYPE':    ('Tipo esp. referencia', 'est'),
                'SP_TYPE':   ('Tipo esp. referencia', 'est'),
                'SPECTYPE':  ('Tipo espectral',       'est'),
                'MTYPE':     ('Tipo MK',              'est'),
                'OBJTYPE':   ('Tipo objeto',          'est'),
                'VMAG':      ('Mag V',                'est'),
                'BMAG':      ('Mag B',                'est'),
                'NAXIS1':    ('Nº píxeles',           'spec'),
                'CRVAL1':    ('λ₀ — CRVAL1 (Å)',     'spec'),
                'CRPIX1':    ('Píxel ref. — CRPIX1', 'spec'),
                'CDELT1':    ('Dispersión (Å/px)',    'spec'),
                'CTYPE1':    ('Tipo eje λ',           'spec'),
                'CUNIT1':    ('Unidad eje λ',         'spec'),
                'BUNIT':     ('Unidad flujo',         'spec'),
                'SNR':       ('SNR',                  'spec'),
                'S/N':       ('S/N estimado',         'spec'),
                'GAIN':      ('Ganancia (e⁻/ADU)',    'inst'),
                'RDNOISE':   ('Ruido de lectura',     'inst'),
                'SATURATE':  ('Nivel saturación',     'inst'),
                'EQUINOX':   ('Equinoccio',           'astr'),
                'RADECSYS':  ('Sistema coord.',       'astr'),
                'EPOCH':     ('Época',                'astr'),
            }
            fits_summary = []
            for k, (label, grupo) in _FITS_LABELS.items():
                val = header.get(k)
                if val is not None:
                    val_str = str(val).strip()
                    if val_str and val_str not in ('', 'N/A', 'UNKNOWN'):
                        fits_summary.append({
                            'clave': k,
                            'label': label,
                            'valor': val_str,
                            'grupo': grupo,
                        })

            # Usar cadena vacía cuando el campo no está en la cabecera
            _obj  = str(header.get('OBJECT', '') or '').strip()
            _sptp = str(header.get('SPTYPE', '') or header.get('SP_TYPE', '') or
                        header.get('SPECTYPE', '') or header.get('MTYPE', '') or '').strip()

            metadata = {
                'objeto':       _obj,   # '' si no hay OBJECT en la cabecera
                'tipo_original':_sptp,  # '' si no hay SPTYPE / SP_TYPE
                'fits_header':  fits_header,
                'fits_summary': fits_summary,
                'file_format':  'FITS'
            }

            return wavelengths, data, None, metadata
        except Exception as e:
            return None, None, f"Error al leer archivo FITS: {str(e)}", None
    else:
        return None, None, "Formato no soportado", None


def process_spectrum(filepath, filename, use_multi_method=True,
                     include_neural=True,
                     knn_weight=0.20, cnn_1d_weight=0.20, cnn_2d_weight=0.00,
                     physical_weight=0.10, dt_weight=0.40, template_weight=0.10,
                     preferred_neural='auto'):
    """
    Procesa un espectro completo: carga, normaliza, mide, clasifica

    Parameters
    ----------
    filepath : str
        Ruta al archivo
    filename : str
        Nombre del archivo
    use_multi_method : bool
        Si True, usar validación multi-método
    include_neural : bool
        Si incluir los modelos neuronales en la votación
    knn_weight / cnn_1d_weight / cnn_2d_weight : float
        Pesos individuales para cada modelo neural (0.0–1.0)
    physical_weight / dt_weight / template_weight : float
        Pesos para los métodos clásicos
    preferred_neural : str
        Ignorado (todos los modelos votan independientemente)

    Returns
    -------
    dict con resultados (incluye confianza y alternativas si multi_method activo)
    """
    # Cargar
    wavelengths, flux, error, metadata = load_spectrum_file(filepath)
    if error:
        return {'error': error}

    # Construir pesos personalizados — cada método tiene su propio peso
    if include_neural:
        custom_weights = {
            'physical':          physical_weight,
            'decision_tree':     dt_weight,
            'template_matching': template_weight,
            'knn':               knn_weight,
            'cnn_1d':            cnn_1d_weight,
            'cnn_2d':            cnn_2d_weight,
        }
    else:
        # Sin neural: normalizar los pesos clásicos para que sumen 1.0
        total_classic = physical_weight + dt_weight + template_weight
        if total_classic > 0:
            f = 1.0 / total_classic
            custom_weights = {
                'physical':          round(physical_weight  * f, 4),
                'decision_tree':     round(dt_weight        * f, 4),
                'template_matching': round(template_weight  * f, 4),
            }
        else:
            custom_weights = {'physical': 0.15, 'decision_tree': 0.70, 'template_matching': 0.15}

    # Valores por defecto (se sobreescriben en los bloques siguientes)
    tipo_fisico    = None
    subtipo_fisico = None

    # Usar sistema multi-método si está disponible
    if MULTI_METHOD_AVAILABLE and use_multi_method:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, 'models')
            validator = SpectralValidator(
                models_dir=models_dir,
                weights=custom_weights,
                use_neural=include_neural,
            )
            result_multimethod = validator.classify(wavelengths, flux, verbose=False)

            # Extraer resultados
            spectral_type  = result_multimethod['tipo_final']
            subtype        = result_multimethod['subtipo_final']   # coherente con tipo_final
            tipo_fisico    = result_multimethod.get('tipo_fisico', spectral_type)
            subtipo_fisico = result_multimethod.get('subtipo_fisico', subtype)
            confianza      = result_multimethod['confianza']
            alternativas   = result_multimethod['alternativas']
            measurements   = result_multimethod['measurements']
            flux_normalized = result_multimethod.get('flux_normalized')  # Si lo incluimos

            # Si flux_normalized no está en result, normalizar manualmente
            if flux_normalized is None:
                flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            diagnostics = result_multimethod['detalles']['physical']['diagnostics']

        except Exception as e:
            import traceback
            print(f"[!] Error en clasificacion multi-metodo: {e}")
            print("   Usando solo clasificador físico como fallback")
            traceback.print_exc()
            use_multi_method = False  # Fallback a método simple

    # Clasificación simple (físico solamente)
    if not use_multi_method or not MULTI_METHOD_AVAILABLE:
        try:
            # Normalizar
            flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            # Medir líneas
            measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

            # Clasificar
            spectral_type, subtype, diagnostics = classify_star_corrected(
                measurements, wavelengths, flux_normalized
            )

            # Sin métricas de confianza ni alternativas en modo simple
            confianza = None
            alternativas = []

        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print("ERROR EN CLASIFICACIÓN:")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            return {'error': f"Error en clasificación: {str(e)}"}

    # Generar visualización
    try:
        fig_path = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_plot.png")
        plot_spectrum_corrected(
            wavelengths, flux_normalized, measurements,
            spectral_type, subtype,
            metadata['objeto'], metadata['tipo_original'],
            save_path=fig_path
        )
    except Exception as e:
        return {'error': f"Error generando gráfico: {str(e)}"}

    # ── Agregar líneas del Árbol de Decisión a lineas_usadas ───────────────
    # El árbol usa siempre las mismas 17 features + 6 ratios.
    # Se añaden las que realmente están medidas (EW > 0) al diagnóstico
    # para que aparezcan resaltadas en el espectro SVG.
    DT_FEATURE_LINES = [
        'He_II_4686', 'He_I_4471',
        'H_beta', 'H_gamma', 'H_delta', 'H_epsilon',
        'Si_IV_4089', 'Si_III_4553', 'Si_II_4128',
        'Mg_II_4481',
        'Ca_II_K', 'Ca_II_H', 'Ca_I_4227',
        'Fe_I_4046', 'Fe_I_4144', 'Fe_I_4383', 'Fe_I_4957',
    ]
    existing_lu = set(diagnostics.get('lineas_usadas', []))
    dt_lineas_en_rango = []
    for ln in DT_FEATURE_LINES:
        data = measurements.get(ln, {})
        if data.get('ew', 0) > 0.05:          # detectada y en rango
            display = ln.replace('_', ' ')
            dt_lineas_en_rango.append(display)
            existing_lu.add(display)           # evitar duplicados
    diagnostics['lineas_usadas'] = sorted(existing_lu)
    diagnostics['dt_lineas'] = dt_lineas_en_rango  # campo separado para el frontend

    # Preparar líneas detectadas (EW significativo) y medición completa
    detected_lines = []
    todas_lineas   = []
    for line_name, data in measurements.items():
        entry = {
            'nombre':           line_name.replace('_', ' '),
            'longitud_onda':    data['wavelength'],
            'ancho_equivalente':round(data['ew'],    3),
            'profundidad':      round(data['depth'], 3)
        }
        todas_lineas.append(entry)
        if data['ew'] > 0.05:
            detected_lines.append(entry)

    # Ordenar por EW decreciente
    detected_lines.sort(key=lambda x: x['ancho_equivalente'], reverse=True)
    todas_lineas.sort(key=lambda x: x['longitud_onda'])

    # Datos de espectro para visualización SVG
    try:
        _wav  = np.asarray(wavelengths, dtype=float)
        _flux = np.nan_to_num(np.asarray(flux_normalized, dtype=float), nan=1.0, posinf=1.0, neginf=0.0)

        # Recortar rayos cósmicos y spikes de emisión para no distorsionar el eje Y
        _flux = np.clip(_flux, 0.0, 1.5)

        # Submuestreo con preservación de mínimos (mantiene visibles las líneas de absorción)
        # Para espectros de alta resolución (p.ej. 0.05 Å/px) el stride simple elimina
        # líneas estrechas; tomar el mínimo en cada bin garantiza que las absorpciones
        # más profundas siempre se incluyen en la curva final.
        TARGET_PTS = 4000
        _samp = max(1, len(_wav) // TARGET_PTS)
        if _samp > 1:
            n_bins = len(_wav) // _samp
            wav_out  = _wav[:n_bins * _samp].reshape(n_bins, _samp).mean(axis=1)
            flux_out = _flux[:n_bins * _samp].reshape(n_bins, _samp).min(axis=1)
        else:
            wav_out  = _wav
            flux_out = _flux

        spectrum_data = {
            'wavelength': wav_out.tolist(),
            'flux':       flux_out.tolist(),
            'wmin': float(_wav.min()),
            'wmax': float(_wav.max())
        }
    except Exception:
        spectrum_data = None

    # ── Clase de luminosidad MK ──────────────────────────────────────────
    # IMPORTANTE: luminosity_class y mk_full SIEMPRE se calculan desde el tipo
    # final votado (spectral_type), no desde diagnostics del clasificador físico.
    # El físico puede haber dado un tipo diferente al voto final, por lo que
    # usar su mk_full produciría combinaciones incoherentes (ej. "O6-O7II" cuando
    # el tipo final es "M5").
    _lum_names_map = {
        'Ia':  'Supergigante muy luminosa',
        'Ib':  'Supergigante',
        'II':  'Gigante brillante',
        'III': 'Gigante',
        'IV':  'Subgigante',
        'V':   'Secuencia principal (enana)',
    }

    # Paso 1 — calcular clase de luminosidad a partir del tipo FINAL
    luminosity_class = ''
    if _LUM_AVAILABLE:
        try:
            luminosity_class = estimate_luminosity_class(measurements, spectral_type)
        except Exception:
            luminosity_class = ''
    # Fallback: intentar recuperar del diagnóstico físico solo si no se obtuvo
    if not luminosity_class:
        luminosity_class = diagnostics.get('luminosity_class', 'V') or 'V'

    # Paso 2 — construir mk_full SIEMPRE desde el tipo final votado
    base_type = subtype if subtype else spectral_type
    if _LUM_AVAILABLE:
        try:
            mk_full = combine_spectral_and_luminosity(base_type, luminosity_class)
        except Exception:
            mk_full = base_type + luminosity_class
    else:
        mk_full = base_type + luminosity_class

    lum_name = _lum_names_map.get(luminosity_class, luminosity_class)

    # Sincronizar diagnostics para que la UI sea coherente
    diagnostics['luminosity_class'] = luminosity_class
    diagnostics['mk_full']          = mk_full
    diagnostics['lum_name']         = lum_name

    # Construir resultado
    result = {
        'success': True,
        'filename': filename,
        'objeto': metadata['objeto'],
        'tipo_original': metadata['tipo_original'],
        'tipo_clasificado': spectral_type,
        'subtipo':          subtype,
        'tipo_fisico':      tipo_fisico    or spectral_type,
        'subtipo_fisico':   subtipo_fisico or subtype,
        # ── Luminosidad MK — accesibles desde el nivel raíz del resultado ──
        'luminosity_class': luminosity_class,
        'mk_full':          mk_full,
        'lum_name':         lum_name,
        # ───────────────────────────────────────────────────────────────────
        'diagnostics': diagnostics,
        'lineas_detectadas': detected_lines,
        'todas_lineas':      todas_lineas,
        'spectrum_data':     spectrum_data,
        'fits_header':       metadata.get('fits_header',  None),
        'fits_summary':      metadata.get('fits_summary', None),
        'file_format':       metadata.get('file_format', 'TXT'),
        'n_lineas': len(detected_lines),
        'rango_lambda': [round(wavelengths[0], 1), round(wavelengths[-1], 1)],
        'n_puntos': len(wavelengths),
        'plot_path': fig_path,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'multi_method_used': use_multi_method and MULTI_METHOD_AVAILABLE
    }

    # Agregar métricas adicionales si multi-método activo
    if use_multi_method and MULTI_METHOD_AVAILABLE:
        result['confianza'] = round(confianza, 1)
        result['alternativas'] = alternativas

    # Convertir tipos numpy a tipos Python nativos para JSON
    return convert_numpy_types(result)


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesa archivo(s) subido(s)"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No se encontraron archivos'}), 400

    files = request.files.getlist('files[]')

    if len(files) == 0:
        return jsonify({'error': 'No se seleccionaron archivos'}), 400

    # Pesos de votación enviados desde el frontend (cada método independiente)
    include_neural   = request.form.get('include_neural', '1') == '1'
    physical_weight  = float(request.form.get('physical_weight',  0.10))
    dt_weight        = float(request.form.get('dt_weight',         0.40))
    template_weight  = float(request.form.get('template_weight',   0.10))
    knn_weight       = float(request.form.get('knn_weight',        0.20))
    cnn_1d_weight    = float(request.form.get('cnn_1d_weight',     0.20))
    cnn_2d_weight    = float(request.form.get('cnn_2d_weight',     0.00))

    results = []
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Procesar
            result = process_spectrum(
                filepath, filename,
                include_neural=include_neural,
                physical_weight=physical_weight,
                dt_weight=dt_weight,
                template_weight=template_weight,
                knn_weight=knn_weight,
                cnn_1d_weight=cnn_1d_weight,
                cnn_2d_weight=cnn_2d_weight,
            )

            if 'error' in result:
                errors.append({'file': filename, 'error': result['error']})
            else:
                results.append(result)

            # Limpiar archivo subido
            try:
                os.remove(filepath)
            except:
                pass
        else:
            errors.append({'file': file.filename, 'error': 'Formato no permitido'})

    return jsonify({
        'results': results,
        'errors': errors,
        'n_success': len(results),
        'n_errors': len(errors)
    })


@app.route('/upload_single', methods=['POST'])
def upload_single():
    """Sube un archivo al servidor y devuelve su ruta absoluta para usarla en test_spectrum_advanced."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ningún archivo'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío'}), 400

        if not allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'sin extensión'
            return jsonify({'success': False, 'error': f'Formato .{ext} no permitido. Usa .txt, .fits o .fit'}), 400

        filename = secure_filename(file.filename)
        if not filename:
            filename = 'espectro_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.txt'

        # Usar ruta absoluta para evitar problemas con CWD
        upload_dir = os.path.join(project_root, 'webapp', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        return jsonify({'success': True, 'server_path': filepath, 'filename': filename})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error al guardar archivo: {str(e)}'}), 500


@app.route('/result/<filename>')
def show_result(filename):
    """Muestra resultado de clasificación"""
    # Buscar archivo de resultado guardado
    # (Aquí podrías implementar caché de resultados si lo necesitas)
    return render_template('result.html', filename=filename)


@app.route('/plot/<filename>')
def get_plot(filename):
    """Devuelve el gráfico generado"""
    # RESULTS_FOLDER ya es ruta absoluta; send_file necesita ruta absoluta
    plot_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.isfile(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': f'Grafico no encontrado: {filename}'}), 404


@app.route('/export_csv', methods=['POST'])
def export_csv():
    """Exporta resultados a CSV"""
    data = request.json

    if not data or 'results' not in data:
        return jsonify({'error': 'No hay datos para exportar'}), 400

    results = data['results']

    # Crear DataFrame
    rows = []
    for r in results:
        rows.append({
            'Archivo': r['filename'],
            'Objeto': r['objeto'],
            'Tipo Original': r['tipo_original'],
            'Tipo Clasificado': r['tipo_clasificado'],
            'Subtipo': r['subtipo'],
            'Líneas Detectadas': r['n_lineas'],
            'Rango λ (Å)': f"{r['rango_lambda'][0]}-{r['rango_lambda'][1]}",
            'Puntos Espectrales': r['n_puntos'],
            'Fecha Procesamiento': r['timestamp']
        })

    df = pd.DataFrame(rows)

    # Guardar en memoria
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'clasificacion_espectral_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route('/export_detailed_csv/<filename>')
def export_detailed_csv(filename):
    """Exporta mediciones detalladas de líneas a CSV"""
    # Recargar resultados desde caché si es necesario
    # Por ahora, devolver error
    return jsonify({'error': 'Función no implementada aún'}), 501


@app.route('/fits_extract_one', methods=['POST'])
def fits_extract_one():
    """
    Procesa UN solo archivo FITS y devuelve el TXT + metadatos.
    El JS lo llama en bucle, de a uno, para evitar límites de tamaño.
    """
    import io as _io
    try:
        from astropy.io import fits as _fits
    except ImportError:
        return jsonify({'ok': False, 'error': 'astropy no instalado. Ejecuta: pip install astropy'}), 500

    f = request.files.get('file')
    if not f:
        return jsonify({'ok': False, 'error': 'No se recibió archivo'}), 400

    orig_name = f.filename
    try:
        raw = f.read()
        with _fits.open(_io.BytesIO(raw)) as hdul:
            header = hdul[0].header
            data   = hdul[0].data

            objeto = str(header.get('OBJECT',  header.get('OBJNAME', ''))).strip()
            sptype = str(header.get('SPTYPE',  header.get('SPTTYPE', ''))).strip()

            if not objeto:
                h_ident = str(header.get('H_IDENT', '')).strip()
                if h_ident:
                    parts = h_ident.split('/')
                    objeto = parts[0].strip()
                    if not sptype and len(parts) > 1:
                        sptype = parts[1].strip()

            objeto_safe = (objeto or os.path.splitext(orig_name)[0]).replace('/', '-').replace(' ', '_')
            sptype_safe = (sptype or 'tipo_desconocido').replace('/', '-').replace(' ', '_')

            if data is None:
                raise ValueError('HDU sin datos')
            if data.ndim > 1:
                data = data.flatten()
            if len(data) == 0:
                raise ValueError('Array vacío')

            crval1 = float(header.get('CRVAL1', 1.0))
            crpix1 = float(header.get('CRPIX1', 1.0))
            cdelt1 = float(header.get('CDELT1', 1.0))
            pixels = np.arange(1, len(data) + 1, dtype=float)
            wavelengths = crval1 + (pixels - crpix1) * cdelt1

            nombre_out = f"{objeto_safe}_tipo{sptype_safe}.txt"
            lines_out  = ['Longitud_de_onda_A,espectro']
            for w, v in zip(wavelengths, data):
                lines_out.append(f'{w:.4f},{v:.6g}')
            txt_content = '\n'.join(lines_out)

            return jsonify({
                'ok':          True,
                'original':    orig_name,
                'objeto':      objeto,
                'sptype':      sptype,
                'nombre_out':  nombre_out,
                'rango_lambda': f'{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} Å',
                'n_puntos':    int(len(data)),
                'txt_content': txt_content
            })

    except Exception as exc:
        return jsonify({'ok': False, 'original': orig_name, 'error': str(exc)})


@app.route('/fits_extract_batch', methods=['POST'])
def fits_extract_batch():
    """
    Recibe uno o varios archivos FITS, extrae metadatos y convierte a TXT.
    Devuelve un ZIP con los TXT + un CSV de metadatos.
    """
    import zipfile
    import io as _io
    try:
        from astropy.io import fits as _fits
        _ASTROPY = True
    except ImportError:
        _ASTROPY = False

    if not _ASTROPY:
        return jsonify({'error': 'astropy no está instalado. Ejecuta: pip install astropy'}), 500

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No se recibieron archivos'}), 400

    results   = []   # lista de dicts con metadatos
    txt_files = {}   # nombre_salida -> contenido TXT

    for f in files:
        orig_name = f.filename
        entry = {
            'original':  orig_name,
            'objeto':    '',
            'sptype':    '',
            'nombre_out': '',
            'rango_lambda': '',
            'n_puntos':  0,
            'ok':        False,
            'error':     ''
        }
        try:
            raw = f.read()
            with _fits.open(_io.BytesIO(raw)) as hdul:
                header = hdul[0].header
                data   = hdul[0].data

                # Metadatos clave
                objeto  = str(header.get('OBJECT',  header.get('OBJNAME', ''))).strip().replace('/', '-').replace(' ', '_')
                sptype  = str(header.get('SPTYPE',  header.get('SPTTYPE', ''))).strip().replace('/', '-').replace(' ', '_')
                # H_IDENT como alternativa
                if not objeto:
                    h_ident = str(header.get('H_IDENT', '')).strip()
                    if h_ident:
                        parts = h_ident.split('/')
                        objeto = parts[0].strip().replace(' ', '_')
                        if not sptype and len(parts) > 1:
                            sptype = parts[1].strip().replace(' ', '_')

                objeto  = objeto  or os.path.splitext(orig_name)[0]
                sptype  = sptype  or 'tipo_desconocido'

                entry['objeto'] = objeto
                entry['sptype'] = sptype

                # Validar que hay datos 1D
                if data is None:
                    raise ValueError('El HDU primario no contiene datos')
                if data.ndim > 1:
                    data = data.flatten()
                if len(data) == 0:
                    raise ValueError('Array de datos vacío')

                # Calcular longitudes de onda
                crval1 = float(header.get('CRVAL1', 1.0))
                crpix1 = float(header.get('CRPIX1', 1.0))
                cdelt1 = float(header.get('CDELT1', 1.0))
                pixels = np.arange(1, len(data) + 1, dtype=float)
                wavelengths = crval1 + (pixels - crpix1) * cdelt1

                entry['rango_lambda'] = f"{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} Å"
                entry['n_puntos']     = len(data)

                # Nombre de salida: OBJETO_tipoSPTYPE.txt
                nombre_out = f"{objeto}_tipo{sptype}.txt"
                entry['nombre_out'] = nombre_out

                # Construir TXT con dos columnas
                lines_out = ['Longitud_de_onda_A,espectro']
                for w, v in zip(wavelengths, data):
                    lines_out.append(f'{w:.4f},{v:.6g}')
                txt_files[nombre_out] = '\n'.join(lines_out)

                entry['ok'] = True

        except Exception as exc:
            entry['error'] = str(exc)

        results.append(entry)

    # Construir ZIP en memoria
    zip_buf = _io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for nombre, contenido in txt_files.items():
            zf.writestr(nombre, contenido)

        # CSV de metadatos
        meta_lines = ['original,objeto,sptype,nombre_salida,rango_lambda,n_puntos,ok,error']
        for r in results:
            meta_lines.append(
                f'"{r["original"]}","{r["objeto"]}","{r["sptype"]}",'
                f'"{r["nombre_out"]}","{r["rango_lambda"]}",{r["n_puntos"]},'
                f'{r["ok"]},"{r["error"]}"'
            )
        zf.writestr('_metadatos.csv', '\n'.join(meta_lines))

    zip_buf.seek(0)

    ok_count  = sum(1 for r in results if r['ok'])
    err_count = len(results) - ok_count

    # Devolver ZIP como base64 + resumen JSON
    import base64 as _b64
    zip_b64 = _b64.b64encode(zip_buf.read()).decode()

    return jsonify({
        'ok':        ok_count,
        'errores':   err_count,
        'resultados': results,
        'zip_b64':   zip_b64,
        'meta_csv':  '\n'.join(meta_lines)
    })


@app.route('/info')
def info():
    """Página con información sobre el método"""
    return render_template('info.html')


@app.route('/health')
def health():
    """Endpoint de salud para verificar que el servidor está corriendo"""
    return jsonify({
        'status': 'OK',
        'lines_configured': len(SPECTRAL_LINES),
        'version': '3.0'
    })


@app.route('/run_script', methods=['POST'])
def run_script():
    """Ejecuta un script .bat desde la webapp"""
    data = request.json
    script_name = data.get('script')

    # Scripts permitidos (por seguridad)
    allowed_scripts = {
        '1_INSTALAR_DEPENDENCIAS': '1_INSTALAR_DEPENDENCIAS.bat',
        '2_ENTRENAR_MODELOS': '2_ENTRENAR_MODELOS.bat',
        '4_TEST_ESPECTRO': '4_TEST_ESPECTRO.bat',
        '5_VER_METRICAS': '5_VER_METRICAS.bat',
    }

    if script_name not in allowed_scripts:
        return jsonify({
            'success': False,
            'error': f'Script no permitido: {script_name}'
        }), 400

    # Ruta al script
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, allowed_scripts[script_name])

    if not os.path.exists(script_path):
        return jsonify({
            'success': False,
            'error': f'Script no encontrado: {script_path}'
        }), 404

    try:
        # Determinar si estamos en Windows o WSL
        import platform
        utf8_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        if platform.system() == 'Windows':
            # Windows nativo
            result = subprocess.run(
                ['cmd', '/c', script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,  # 10 minutos máximo
                cwd=project_root,
                env=utf8_env
            )
        else:
            # WSL o Linux - ejecutar .bat a través de cmd.exe
            # Convertir ruta WSL a Windows si es necesario
            win_path = script_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            result = subprocess.run(
                ['cmd.exe', '/c', win_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,
                cwd=project_root,
                env=utf8_env
            )

        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None,
            'return_code': result.returncode
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El script excedió el tiempo límite (10 minutos)'
        }), 408

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/list_scripts')
def list_scripts():
    """Lista los scripts disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scripts = []
    for filename in os.listdir(project_root):
        if filename.endswith('.bat') and filename[0].isdigit():
            scripts.append({
                'name': filename.replace('.bat', ''),
                'filename': filename,
                'exists': True
            })

    return jsonify({'scripts': sorted(scripts, key=lambda x: x['name'])})


# ============================================================================
# NUEVOS ENDPOINTS PARA HERRAMIENTAS AVANZADAS
# ============================================================================

@app.route('/train_model', methods=['POST'])
def train_model():
    """Entrena el modelo con opciones personalizadas"""
    data = request.json

    catalog_path = data.get('catalog_path', 'data/elodie/')
    model_type = data.get('model_type', 'decision_tree')
    max_depth = data.get('max_depth', 9)
    test_size = data.get('test_size', 20) / 100.0
    n_estimators = data.get('n_estimators', 100)
    output_path = data.get('output_path', 'models/')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validar ruta del catálogo
    full_catalog_path = os.path.join(project_root, catalog_path)
    if not os.path.exists(full_catalog_path):
        return jsonify({
            'success': False,
            'error': f'Catálogo no encontrado: {catalog_path}'
        }), 404

    # Contar archivos en el catálogo
    spectrum_files = [f for f in os.listdir(full_catalog_path) if '_tipo' in f.lower() and f.endswith('.txt')]
    if len(spectrum_files) == 0:
        return jsonify({
            'success': False,
            'error': f'No se encontraron espectros etiquetados en: {catalog_path}'
        }), 400

    try:
        import time
        start_time = time.time()

        # Importar módulo de entrenamiento
        sys.path.insert(0, project_root)

        # Ejecutar entrenamiento usando train_and_validate.py
        # Construir comando Python (el script está en src/)
        train_script = os.path.join(project_root, 'src', 'train_and_validate.py')

        if not os.path.exists(train_script):
            return jsonify({
                'success': False,
                'error': 'Script de entrenamiento no encontrado: src/train_and_validate.py'
            }), 404

        # Ejecutar script de entrenamiento con parámetros
        import subprocess
        cmd = [
            sys.executable, train_script,
            '--catalog', full_catalog_path,
            '--output', os.path.join(project_root, output_path),
            '--model', model_type,
            '--max-depth', str(max_depth),
            '--test-size', str(test_size),
        ]

        if model_type in ['random_forest', 'gradient_boosting']:
            cmd.extend(['--n-estimators', str(n_estimators)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
            cwd=project_root,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            # Leer métricas del modelo entrenado
            metadata_path = os.path.join(project_root, output_path, 'metadata.json')
            metrics = {}
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metrics = json.load(f)

            return jsonify({
                'success': True,
                'output': result.stdout,
                'elapsed_time': round(elapsed_time, 2),
                'n_samples': metrics.get('n_train', 0) + metrics.get('n_test', 0),
                'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2),
                'model_type': model_type,
                'catalog': catalog_path,
                'n_files': len(spectrum_files)
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr or 'Error desconocido durante el entrenamiento',
                'output': result.stdout
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El entrenamiento excedió el tiempo límite (10 minutos)'
        }), 408

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# ============================================================================
# ENDPOINT STREAMING PARA ENTRENAMIENTO EN TIEMPO REAL
# ============================================================================

@app.route('/train_model_stream', methods=['POST'])
def train_model_stream():
    """Entrena modelo con salida en tiempo real via Server-Sent Events."""
    import json as _json
    import re as _re
    from flask import Response, stream_with_context

    data = request.json or {}
    catalog_path   = data.get('catalog_path', 'data/elodie/')
    model_type     = data.get('model_type', 'decision_tree')
    max_depth      = int(data.get('max_depth', 9))
    test_size      = int(data.get('test_size', 20)) / 100.0
    n_estimators   = int(data.get('n_estimators', 100))
    output_path    = data.get('output_path', 'models/')

    full_catalog = (catalog_path if os.path.isabs(catalog_path)
                    else os.path.join(project_root, catalog_path))
    full_output  = (output_path if os.path.isabs(output_path)
                    else os.path.join(project_root, output_path))

    train_script = os.path.join(project_root, 'src', 'train_and_validate.py')

    def generate():
        """
        Generador SSE (Server-Sent Events).

        Protocolo SSE: cada mensaje tiene formato 'data: <JSON>\n\n'.
        El navegador lo recibe en tiempo real a través de EventSource.
        Aquí lanzamos el proceso Python de entrenamiento y transmitimos
        su salida línea a línea al navegador conforme va apareciendo.

        Tipos de evento JSON:
          'line'  → línea de texto del proceso (mostrar en consola)
          'done'  → entrenamiento terminó (con success, accuracy, n_samples)
        El campo 'pct' opcional indica el porcentaje de progreso (0-100).
        """
        import time as _time

        def evt(msg, pct=None, tipo='line'):
            # Empaquetar mensaje como JSON y formatear en protocolo SSE
            if pct is not None:
                return f"data: {_json.dumps({'type': tipo, 'pct': pct, 'msg': msg})}\n\n"
            return f"data: {_json.dumps({'type': tipo, 'msg': msg})}\n\n"

        yield evt('Verificando archivos...', pct=2)

        if not os.path.exists(train_script):
            yield evt('Script no encontrado: src/train_and_validate.py', tipo='done')
            return

        if not os.path.isdir(full_catalog):
            yield evt(f'Catalogo no encontrado: {full_catalog}', tipo='done')
            return

        yield evt(f'Catalogo OK. Lanzando proceso Python...', pct=4)

        cmd = [
            sys.executable, '-u', train_script,
            '--catalog', full_catalog,
            '--output', full_output,
            '--model', model_type,
            '--max-depth', str(max_depth),
            '--test-size', str(test_size),
        ]
        if model_type in ['random_forest', 'gradient_boosting']:
            cmd.extend(['--n-estimators', str(n_estimators)])

        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        pct = 5

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,          # sin buffer en el lado del padre
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=project_root,
                env=env
            )
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'Error al lanzar proceso: {e}'})}\n\n"
            return

        yield evt(f'Proceso PID {proc.pid} iniciado. Esperando salida...', pct=5)

        last_keepalive = _time.time()

        # Leer línea a línea con keepalive cada 3 segundos.
        # IMPORTANTE: readline() bloquea hasta que llega una línea o el proceso termina.
        # Si el proceso no emite output por >3s enviamos un comentario SSE de keepalive
        # para mantener la conexión HTTP activa (nginx/proxies cierran conexiones ociosas).
        while True:
            line = proc.stdout.readline()

            # readline() devuelve '' cuando el proceso termina y cierra stdout
            if line == '':
                if proc.poll() is not None:
                    # El proceso terminó y cerró stdout → salir del loop
                    break
                # Proceso vivo pero sin output aún → keepalive
                if _time.time() - last_keepalive >= 3:
                    yield ': keepalive\n\n'
                    last_keepalive = _time.time()
                _time.sleep(0.05)
                continue

            last_keepalive = _time.time()
            line = line.rstrip('\r\n')
            if not line.strip():
                continue

            # Mapeo de palabras clave del output → porcentaje de progreso
            # Permite mostrar una barra de progreso en el navegador
            if 'Archivos encontrados' in line:
                pct = 10
            elif 'Procesados:' in line:
                m = _re.search(r'\((\d+\.?\d*)%\)', line)
                if m:
                    pct = int(10 + float(m.group(1)) * 0.40)
            elif 'Espectros procesados exitosamente' in line:
                pct = 52
            elif any(k in line for k in ['ENTRENAMIENTO', 'RANDOM FOREST', 'GRADIENT BOOSTING']):
                pct = 58
            elif 'Accuracy (entrenamiento)' in line:
                pct = 75
            elif 'Accuracy (prueba)' in line or 'Accuracy en test' in line:
                pct = 82
            elif 'VALIDACION CRUZADA' in line or 'VALIDACIÓN CRUZADA' in line:
                pct = 85
            elif 'Accuracy promedio' in line:
                pct = 90
            elif 'VALIDACION DEL CLASIFICADOR' in line or 'VALIDACIÓN DEL CLASIFICADOR' in line:
                pct = 93
            elif '[OK]' in line and 'guardado' in line.lower():
                pct = 97

            yield evt(line, pct=pct)

        proc.wait()

        if proc.returncode == 0:
            metrics = {}
            metadata_path = os.path.join(full_output, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metrics = _json.load(f)
            n_samples = metrics.get('n_train', 0) + metrics.get('n_test', 0)
            accuracy  = round(metrics.get('accuracy_test', 0) * 100, 2)
            yield f"data: {_json.dumps({'type': 'done', 'success': True, 'accuracy': accuracy, 'n_samples': n_samples})}\n\n"
        else:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'El proceso termino con codigo {proc.returncode}'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ============================================================================
# ENDPOINT STREAMING PARA ENTRENAMIENTO NEURAL EN TIEMPO REAL
# ============================================================================

@app.route('/train_neural_stream', methods=['POST'])
def train_neural_stream():
    """Entrena modelo KNN/CNN con salida en tiempo real via Server-Sent Events."""
    import json as _json
    import re as _re
    from flask import Response, stream_with_context

    data = request.json or {}
    model_type   = data.get('model_type', 'knn')
    catalog_path = data.get('catalog_path', 'data/elodie/')
    test_size    = float(data.get('test_size', 0.2))
    output_path  = data.get('output_path', 'models/')

    full_catalog = (catalog_path if os.path.isabs(catalog_path)
                    else os.path.join(project_root, catalog_path))
    full_output  = (output_path if os.path.isabs(output_path)
                    else os.path.join(project_root, output_path))

    train_script = os.path.join(project_root, 'src', 'train_neural_models.py')

    def generate():
        import time as _time

        def evt(msg, pct=None, tipo='line'):
            payload = {'type': tipo, 'msg': msg}
            if pct is not None:
                payload['pct'] = pct
            return f"data: {_json.dumps(payload)}\n\n"

        yield evt('Verificando archivos...', pct=2)

        if not os.path.exists(train_script):
            yield evt('Script no encontrado: src/train_neural_models.py', tipo='done')
            return
        if not os.path.isdir(full_catalog):
            yield evt(f'Catalogo no encontrado: {full_catalog}', tipo='done')
            return

        yield evt(f'Catalogo OK. Lanzando proceso {model_type.upper()}...', pct=5)

        cmd = [
            sys.executable, '-u', train_script,
            '--model', model_type,
            '--catalog', full_catalog,
            '--output', full_output,
            '--test-size', str(test_size),
        ]
        if model_type == 'knn':
            cmd += ['--n-neighbors', str(data.get('n_neighbors', 5)),
                    '--weights',     data.get('weights', 'distance'),
                    '--metric',      data.get('metric', 'euclidean')]
        else:
            cmd += ['--epochs',           str(data.get('epochs', 50)),
                    '--batch-size',       str(data.get('batch_size', 32)),
                    '--learning-rate',    str(data.get('learning_rate', 0.001)),
                    '--dropout',          str(data.get('dropout_rate', 0.3)),
                    '--dense-units',      str(data.get('dense_units', 128)),
                    '--max-augment-ratio',str(data.get('max_augment_ratio', 5))]
            if not data.get('downsample_majority', True):
                cmd.append('--no-downsample')
            if model_type == 'cnn_2d':
                cmd += ['--image-size', str(data.get('image_size', 64)),
                        '--image-dir',  data.get('image_dir', 'espectros png/'),
                        '--labels-csv', data.get('labels_csv', 'espectros png/clasificacion_estrellas.csv')]

        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        pct = 5

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=project_root,
                env=env
            )
        except Exception as e:
            yield evt(f'Error al lanzar proceso: {e}', tipo='done')
            return

        yield evt(f'Proceso PID {proc.pid} iniciado. Esperando salida...', pct=6)

        last_keepalive = _time.time()

        while True:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() is not None:
                    break
                if _time.time() - last_keepalive >= 3:
                    yield ': keepalive\n\n'
                    last_keepalive = _time.time()
                _time.sleep(0.05)
                continue

            last_keepalive = _time.time()
            line = line.rstrip('\r\n')
            if not line.strip():
                continue

            # Detectar línea JSON de época (EPOCH_DATA:{...})
            if line.startswith('EPOCH_DATA:'):
                try:
                    ep_data = _json.loads(line[len('EPOCH_DATA:'):])
                    yield f"data: {_json.dumps({'type': 'epoch', **ep_data})}\n\n"
                    pct = min(55 + int(ep_data.get('epoch', 0) * 2), 90)
                except Exception:
                    pass
                continue

            # Calcular porcentaje según contenido
            if 'Archivos encontrados' in line:
                pct = 12
            elif 'Procesados:' in line:
                m = _re.search(r'(\d+)/(\d+)', line)
                if m:
                    pct = int(12 + (int(m.group(1)) / max(int(m.group(2)), 1)) * 30)
            elif 'Espectros cargados exitosamente' in line:
                pct = 45
            elif 'ENTRENANDO KNN' in line:
                pct = 55
            elif 'ENTRENANDO CNN' in line:
                pct = 55
            elif 'Entrenamiento completado' in line:
                pct = 85
            elif 'Accuracy en test' in line:
                pct = 90
            elif 'Epocas ejecutadas' in line or 'EarlyStopping' in line.lower():
                pct = 88
            elif '[OK]' in line or 'guardado' in line.lower():
                pct = 95

            yield evt(line, pct=pct)

        proc.wait()

        if proc.returncode == 0:
            metrics = {}
            metadata_path = os.path.join(full_output, f'{model_type}_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metrics = _json.load(f)
            yield f"data: {_json.dumps({'type': 'done', 'success': True, 'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2), 'n_samples': metrics.get('n_samples', 0), 'n_classes': len(metrics.get('classes', []))})}\n\n"
        else:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'El proceso termino con codigo {proc.returncode}'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ============================================================================
# ENDPOINTS PARA REDES NEURONALES (KNN, CNN)
# ============================================================================

@app.route('/train_neural', methods=['POST'])
def train_neural():
    """Entrena modelos KNN o CNN para clasificación espectral"""
    data = request.json

    model_type = data.get('model_type', 'knn')  # 'knn', 'cnn_1d', 'cnn_2d'
    catalog_path = data.get('catalog_path', 'data/elodie/')
    # test_size ya viene como decimal desde JS (ej: 0.2 para 20%)
    test_size = data.get('test_size', 0.2)
    if test_size > 1:  # Si viene como porcentaje, convertir
        test_size = test_size / 100.0
    output_path = data.get('output_path', 'models/')

    # Parámetros KNN
    n_neighbors = data.get('n_neighbors', 5)
    weights = data.get('weights', 'uniform')
    metric = data.get('metric', 'euclidean')

    # Parámetros CNN
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 32)
    learning_rate = data.get('learning_rate', 0.001)
    dropout_rate = data.get('dropout_rate', 0.3)
    dense_units = data.get('dense_units', 128)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validar ruta del catálogo
    full_catalog_path = os.path.join(project_root, catalog_path)
    if not os.path.exists(full_catalog_path):
        return jsonify({
            'success': False,
            'error': f'Catálogo no encontrado: {catalog_path}'
        }), 404

    # Contar archivos
    spectrum_files = [f for f in os.listdir(full_catalog_path)
                      if '_tipo' in f.lower() and f.endswith('.txt')]
    if len(spectrum_files) == 0:
        return jsonify({
            'success': False,
            'error': f'No se encontraron espectros etiquetados en: {catalog_path}'
        }), 400

    try:
        import time
        start_time = time.time()

        # Verificar que existe el script de entrenamiento (en src/)
        train_script = os.path.join(project_root, 'src', 'train_neural_models.py')
        if not os.path.exists(train_script):
            return jsonify({
                'success': False,
                'error': 'Script de entrenamiento no encontrado: src/train_neural_models.py'
            }), 404

        # Construir comando
        cmd = [
            sys.executable, train_script,
            '--model', model_type,
            '--catalog', full_catalog_path,
            '--output', os.path.join(project_root, output_path),
            '--test-size', str(test_size),
        ]

        # Parámetros de augmentación/balanceo
        downsample_majority = data.get('downsample_majority', True)
        max_augment_ratio   = data.get('max_augment_ratio', 5)

        # Agregar parámetros específicos según tipo de modelo
        if model_type == 'knn':
            cmd.extend([
                '--n-neighbors', str(n_neighbors),
                '--weights', weights,
                '--metric', metric,
            ])
        else:  # CNN
            cmd.extend([
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--learning-rate', str(learning_rate),
                '--dropout', str(dropout_rate),
                '--dense-units', str(dense_units),
                '--max-augment-ratio', str(max_augment_ratio),
            ])
            if not downsample_majority:
                cmd.append('--no-downsample')

        # Ejecutar entrenamiento
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800,  # 30 minutos para CNN
            cwd=project_root,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        elapsed_time = time.time() - start_time

        # Log para debug
        print(f"[train_neural] Comando ejecutado: {' '.join(cmd)}")
        print(f"[train_neural] Return code: {result.returncode}")
        if result.stdout:
            print(f"[train_neural] STDOUT (ultimos 1500 chars):")
            print(result.stdout[-1500:])
        if result.stderr:
            print(f"[train_neural] STDERR:")
            print(result.stderr[-2000:])

        if result.returncode == 0:
            # Leer métricas del modelo entrenado
            metadata_path = os.path.join(project_root, output_path, f'{model_type}_metadata.json')
            metrics = {}
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metrics = json.load(f)

            return jsonify({
                'success': True,
                'output': result.stdout,
                'elapsed_time': round(elapsed_time, 2),
                'training_time': f"{int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}",
                'n_samples': metrics.get('n_samples', 0),
                'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2),
                'model_type': model_type,
                'catalog': catalog_path,
                'n_files': len(spectrum_files),
                'classes': metrics.get('classes', []),
                'n_classes': len(metrics.get('classes', []))
            })
        else:
            # Extraer el error más relevante del stderr
            error_msg = result.stderr or 'Error desconocido'
            # Buscar la última línea de error
            if error_msg:
                lines = error_msg.strip().split('\n')
                # Buscar líneas con Error o Exception
                error_lines = [l for l in lines if 'Error' in l or 'Exception' in l or 'error' in l.lower()]
                error_summary = error_lines[-1] if error_lines else lines[-1]
            else:
                error_summary = 'Error desconocido durante el entrenamiento'

            return jsonify({
                'success': False,
                'error': error_summary,
                'stderr': result.stderr[-1500:] if result.stderr else '',
                'stdout': result.stdout[-500:] if result.stdout else ''
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El entrenamiento excedió el tiempo límite (30 minutos)'
        }), 408

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/neural_metrics', methods=['GET'])
def get_neural_metrics():
    """Obtiene métricas de los modelos neuronales disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    metrics = {
        'knn': None,
        'cnn_1d': None,
        'cnn_2d': None
    }

    # Buscar métricas de cada modelo
    for model_type in metrics.keys():
        metadata_path = os.path.join(models_dir, f'{model_type}_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metrics[model_type] = json.load(f)

    # También buscar modelo KNN genérico
    knn_generic = os.path.join(models_dir, 'knn_metadata.json')
    if os.path.exists(knn_generic) and metrics['knn'] is None:
        import json
        with open(knn_generic, 'r') as f:
            metrics['knn'] = json.load(f)

    # Verificar qué modelos existen
    available_models = []
    if os.path.exists(os.path.join(models_dir, 'knn_model.pkl')):
        available_models.append('knn')
    if os.path.exists(os.path.join(models_dir, 'cnn_model.h5')) or \
       os.path.exists(os.path.join(models_dir, 'cnn_1d_model.h5')):
        available_models.append('cnn_1d')
    if os.path.exists(os.path.join(models_dir, 'cnn_2d_model.h5')):
        available_models.append('cnn_2d')

    # Construir dict 'models' con los campos que espera el frontend JS
    models_frontend = {}
    for model_type in available_models:
        meta = metrics.get(model_type) or {}
        acc_raw = meta.get('accuracy_test')
        models_frontend[model_type] = {
            'tipo':            meta.get('model_type', model_type),
            'clases':          meta.get('classes', []),
            'accuracy':        round(acc_raw * 100, 1) if acc_raw is not None else None,
            'n_neighbors':     meta.get('n_neighbors'),
            'spectrum_length': meta.get('spectrum_length'),
            'n_samples':       meta.get('n_samples'),
            'epochs_trained':  meta.get('epochs_trained'),
        }

    return jsonify({
        'success': True,
        'models': models_frontend,
        'metrics': metrics,
        'available_models': available_models
    })


@app.route('/activate_model', methods=['POST'])
def activate_model():
    """Marca un modelo entrenado como activo para que participe en la votación ensemble.

    Escribe / actualiza models/active_models.json con los modelos activados.
    El archivo guarda: {model_type: {active, weight, accuracy, activated_at}}
    """
    import json as _json
    from datetime import datetime as _dt

    data = request.json or {}
    model_type = data.get('model_type')          # 'decision_tree'|'random_forest'|'gradient_boosting'|'knn'|'cnn_1d'
    accuracy   = data.get('accuracy')            # float (porcentaje, ej: 84.5)
    weight     = float(data.get('weight', 0.0))  # 0 = auto (se calculará al clasificar)

    if not model_type:
        return jsonify({'success': False, 'error': 'model_type requerido'}), 400

    models_dir  = os.path.join(project_root, 'models')
    config_path = os.path.join(models_dir, 'active_models.json')

    # Verificar que el modelo existe en disco
    model_files = {
        'knn':               ['knn_model.pkl'],
        'cnn_1d':            ['cnn_model.h5', 'cnn_1d_model.h5'],
        'cnn_2d':            ['cnn_2d_model.h5'],
        'decision_tree':     ['decision_tree.pkl'],
        'random_forest':     ['decision_tree.pkl'],      # mismo archivo, tipo guardado en metadata
        'gradient_boosting': ['decision_tree.pkl'],
    }
    files_to_check = model_files.get(model_type, [])
    exists = any(os.path.exists(os.path.join(models_dir, f)) for f in files_to_check)
    if not exists:
        return jsonify({'success': False,
                        'error': f'No se encontró el archivo del modelo {model_type} en models/'}), 404

    # Cargar config actual
    active = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                active = _json.load(f)
        except Exception:
            active = {}

    # Actualizar / agregar entrada
    active[model_type] = {
        'active':       True,
        'weight':       weight,
        'accuracy':     accuracy,
        'activated_at': _dt.now().isoformat(timespec='seconds'),
    }

    with open(config_path, 'w') as f:
        _json.dump(active, f, indent=2)

    return jsonify({
        'success': True,
        'model_type': model_type,
        'active_models': list(active.keys()),
        'message': f'Modelo {model_type} guardado y activado para clasificación.'
    })


@app.route('/active_models', methods=['GET'])
def get_active_models():
    """Devuelve los modelos actualmente activados."""
    import json as _json
    models_dir  = os.path.join(project_root, 'models')
    config_path = os.path.join(models_dir, 'active_models.json')
    if not os.path.exists(config_path):
        return jsonify({'success': True, 'active_models': {}})
    try:
        with open(config_path, 'r') as f:
            return jsonify({'success': True, 'active_models': _json.load(f)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/neural_history', methods=['GET'])
def neural_history():
    """Devuelve historiales de entrenamiento, matrices de confusión y métricas por clase."""
    import json as _json
    models_dir = os.path.join(project_root, 'models')

    def load_json(path):
        try:
            with open(path, 'r') as f:
                return _json.load(f)
        except Exception:
            return None

    def load_meta_pcm(model_type):
        """Carga per_class_metrics del metadata correspondiente."""
        for name in [f'{model_type}_metadata.json', f'{model_type.replace("_","")}_metadata.json']:
            p = os.path.join(models_dir, name)
            if os.path.exists(p):
                meta = load_json(p)
                if meta:
                    return meta.get('per_class_metrics'), meta.get('accuracy_test'), meta.get('classes', [])
        return None, None, []

    def load_classic_meta(model_type):
        """Carga metadata del modelo clásico (DT/RF/GB guardados en metadata.json)."""
        meta = load_json(os.path.join(models_dir, 'metadata.json'))
        if not meta:
            return None, None, None, []
        # Solo devolver si el modelo guardado coincide con el tipo solicitado
        if meta.get('model_type') != model_type:
            return None, None, None, []
        pcm   = meta.get('per_class_metrics')
        acc   = meta.get('accuracy_test')
        cv    = meta.get('accuracy_cv_mean')
        clss  = meta.get('classes', [])
        return pcm, acc, cv, clss

    # CNN 1D
    cnn1d_hist = load_json(os.path.join(models_dir, 'cnn_1d_history.json'))
    cnn1d_cm   = load_json(os.path.join(models_dir, 'cnn_1d_confusion_matrix.json'))
    cnn1d_pcm, cnn1d_acc, cnn1d_classes = load_meta_pcm('cnn_1d')

    # KNN
    knn_cm  = load_json(os.path.join(models_dir, 'knn_confusion_matrix.json'))
    knn_pcm, knn_acc, knn_classes = load_meta_pcm('knn')

    # Clásicos (DT / RF / GB) — todos comparten el mismo metadata.json y dt_confusion_matrix.json
    dt_meta = load_json(os.path.join(models_dir, 'metadata.json'))
    dt_cm   = load_json(os.path.join(models_dir, 'dt_confusion_matrix.json'))

    def _classic_entry(model_type):
        if not dt_meta or dt_meta.get('model_type') != model_type:
            return {'accuracy': None, 'per_class': None, 'confusion_matrix': None, 'classes': []}
        acc = dt_meta.get('accuracy_test')
        pcm = dt_meta.get('per_class_metrics')
        # Reordenar per_class si existe
        if pcm:
            SPECTRAL = ['O','B','A','F','G','K','M']
            pcm = {k: pcm[k] for k in sorted(pcm, key=lambda c: SPECTRAL.index(c.upper()) if c.upper() in SPECTRAL else 99)}
        return {
            'accuracy':         round(acc * 100, 1) if acc is not None else None,
            'accuracy_cv':      round(dt_meta.get('accuracy_cv_mean', 0) * 100, 1) if dt_meta.get('accuracy_cv_mean') else None,
            'per_class':        pcm,
            'confusion_matrix': dt_cm,
            'classes':          dt_meta.get('classes', []),
            'n_samples':        dt_meta.get('n_samples') or (dt_meta.get('n_train', 0) + dt_meta.get('n_test', 0)),
        }

    return jsonify({
        'success': True,
        'cnn_1d': {
            'history':          cnn1d_hist,
            'confusion_matrix': cnn1d_cm,
            'per_class':        cnn1d_pcm,
            'accuracy':         round(cnn1d_acc * 100, 1) if cnn1d_acc else None,
            'classes':          cnn1d_classes,
        },
        'knn': {
            'confusion_matrix': knn_cm,
            'per_class':        knn_pcm,
            'accuracy':         round(knn_acc * 100, 1) if knn_acc else None,
            'classes':          knn_classes,
        },
        'decision_tree':     _classic_entry('decision_tree'),
        'random_forest':     _classic_entry('random_forest'),
        'gradient_boosting': _classic_entry('gradient_boosting'),
    })


@app.route('/metrics_all', methods=['GET'])
def metrics_all():
    """Devuelve métricas de TODOS los modelos: DT/RF/GB + KNN + CNN 1D/2D."""
    import json as _json
    models_dir = os.path.join(project_root, 'models')

    def load_json(path):
        try:
            with open(path, 'r') as f:
                return _json.load(f)
        except Exception:
            return None

    def build_entry(meta, cm_data):
        if not meta:
            return None
        acc_raw = meta.get('accuracy_test')
        return {
            'model_type':    meta.get('model_type', '?'),
            'accuracy':      round(acc_raw * 100, 1) if acc_raw is not None else None,
            'accuracy_cv':   round(meta.get('accuracy_cv_mean', 0) * 100, 1) if meta.get('accuracy_cv_mean') else None,
            'accuracy_cv_std': round(meta.get('accuracy_cv_std', 0) * 100, 1) if meta.get('accuracy_cv_std') else None,
            'n_samples':     meta.get('n_samples') or (meta.get('n_train', 0) + meta.get('n_test', 0)),
            'n_classes':     meta.get('n_classes'),
            'classes':       meta.get('classes', []),
            'per_class':     meta.get('per_class_metrics'),
            'confusion_matrix': cm_data,
            'params':        meta.get('params', {}),
            'timestamp':     meta.get('timestamp'),
            'history':       None,  # rellenado más abajo para CNN 1D
        }

    # ── DT / RF / GB ─────────────────────────────────────────────────────────
    dt_meta  = load_json(os.path.join(models_dir, 'metadata.json'))
    dt_cm    = load_json(os.path.join(models_dir, 'dt_confusion_matrix.json'))
    dt_entry = build_entry(dt_meta, dt_cm)
    if dt_entry:
        # Curva de diagnóstico: DT → accuracy vs profundidad
        dt_diag = load_json(os.path.join(models_dir, 'dt_depth_curve.json'))
        # RF → error vs n_estimators
        rf_diag = load_json(os.path.join(models_dir, 'rf_estimators_curve.json'))
        dt_entry['diag_curve'] = dt_diag or rf_diag  # usa la que exista según último entrenamiento
        dt_entry['extra'] = {
            'max_depth':    dt_meta.get('params', {}).get('max_depth'),
            'n_estimators': dt_meta.get('params', {}).get('n_estimators'),
        }

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn_meta  = load_json(os.path.join(models_dir, 'knn_metadata.json'))
    knn_cm    = load_json(os.path.join(models_dir, 'knn_confusion_matrix.json'))
    knn_entry = build_entry(knn_meta, knn_cm)
    if knn_entry:
        knn_entry['extra'] = {
            'n_neighbors': knn_meta.get('n_neighbors'),
            'weights':     knn_meta.get('weights'),
            'metric':      knn_meta.get('metric'),
            'cv_mean':     round(knn_meta.get('accuracy_cv_mean', 0) * 100, 1) if knn_meta.get('accuracy_cv_mean') else None,
        }
        knn_entry['diag_curve'] = load_json(os.path.join(models_dir, 'knn_k_curve.json'))

    # ── CNN 1D ───────────────────────────────────────────────────────────────
    cnn1d_meta  = load_json(os.path.join(models_dir, 'cnn_1d_metadata.json'))
    cnn1d_cm    = load_json(os.path.join(models_dir, 'cnn_1d_confusion_matrix.json'))
    cnn1d_hist  = load_json(os.path.join(models_dir, 'cnn_1d_history.json'))
    cnn1d_entry = build_entry(cnn1d_meta, cnn1d_cm)
    if cnn1d_entry:
        cnn1d_entry['history'] = cnn1d_hist
        cnn1d_entry['extra'] = {
            'epochs_trained':  cnn1d_meta.get('epochs_trained'),
            'spectrum_length': cnn1d_meta.get('spectrum_length'),
            'learning_rate':   cnn1d_meta.get('learning_rate'),
            'dropout_rate':    cnn1d_meta.get('dropout_rate'),
        }

    # ── CNN 2D ───────────────────────────────────────────────────────────────
    cnn2d_meta  = load_json(os.path.join(models_dir, 'cnn_2d_metadata.json'))
    cnn2d_entry = build_entry(cnn2d_meta, None)

    return jsonify({
        'success': True,
        'models': {
            'decision_tree': dt_entry,
            'knn':           knn_entry,
            'cnn_1d':        cnn1d_entry,
            'cnn_2d':        cnn2d_entry,
        },
    })


@app.route('/training_log', methods=['GET'])
def training_log():
    """Devuelve el historial acumulativo de todos los entrenamientos."""
    import json as _json
    models_dir = os.path.join(project_root, 'models')
    log_path = os.path.join(models_dir, 'training_log.json')
    try:
        entries = _json.load(open(log_path)) if os.path.exists(log_path) else []
        # Más recientes primero
        entries = sorted(entries, key=lambda e: e.get('timestamp', ''), reverse=True)
        return jsonify({'success': True, 'entries': entries})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'entries': []})


@app.route('/verify_neural_models', methods=['GET'])
def verify_neural_models():
    """Verifica qué modelos neuronales están disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    status = {
        'knn': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'knn_model.pkl')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'knn_metadata.json')),
            'scaler_exists': os.path.exists(os.path.join(models_dir, 'knn_scaler.pkl'))
        },
        'cnn_1d': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'cnn_model.h5')) or
                           os.path.exists(os.path.join(models_dir, 'cnn_1d_model.h5')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'cnn_1d_metadata.json')) or
                              os.path.exists(os.path.join(models_dir, 'cnn_metadata.json'))
        },
        'cnn_2d': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'cnn_2d_model.h5')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'cnn_2d_metadata.json'))
        }
    }

    return jsonify({
        'success': True,
        'status': status,
        'models_dir': models_dir
    })


@app.route('/test_spectrum_advanced', methods=['POST'])
def test_spectrum_advanced():
    """Prueba un espectro con opciones avanzadas"""
    data = request.json

    spectrum_path = data.get('spectrum_path', '')
    method = data.get('method', 'multi_method')
    detail_level = data.get('detail_level', 'detailed')
    save_plot = data.get('save_plot', True)

    if not spectrum_path:
        return jsonify({'success': False, 'error': 'Ruta de espectro no especificada'}), 400

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Resolver la ruta del espectro
    # 1. Si es ruta absoluta y existe, usarla directamente
    # 2. Si es solo nombre de archivo, buscar en carpetas conocidas
    # 3. Si es ruta relativa, buscar desde la raíz del proyecto

    filename_only = os.path.basename(spectrum_path)

    search_dirs = [
        project_root,
        os.path.join(project_root, 'data', 'elodie'),
        os.path.join(project_root, 'data', 'espectros'),
        os.path.join(project_root, 'elodie'),
        os.path.join(project_root, 'espectros'),
    ]

    possible_paths = [spectrum_path]                                            # Ruta tal cual (absoluta)
    possible_paths += [os.path.join(project_root, spectrum_path)]              # Relativa al proyecto
    possible_paths += [os.path.join(project_root, 'webapp', 'uploads', filename_only)]  # Subida reciente
    possible_paths += [os.path.join(d, filename_only) for d in search_dirs]   # Solo nombre en dirs conocidos

    full_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            full_path = os.path.abspath(path)
            break

    if not full_path:
        return jsonify({
            'success': False,
            'error': (
                f'Archivo no encontrado: {spectrum_path}\n'
                f'Puedes usar:\n'
                f'  - Ruta absoluta: C:\\Users\\...\\espectro.txt  o  E:\\pen\\espectro.txt\n'
                f'  - Ruta relativa al proyecto: data/elodie/HD000108_tipoO6pe.txt\n'
                f'  - Solo el nombre si está en data/elodie/ o data/espectros/: HD000108_tipoO6pe.txt'
            )
        }), 404

    try:
        # Procesar espectro
        use_multi = (method == 'multi_method')
        result = process_spectrum(full_path, os.path.basename(spectrum_path), use_multi_method=use_multi)

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500

        # Extraer tipo original del nombre
        original_type = 'N/A'
        filename = os.path.basename(spectrum_path)
        if '_tipo' in filename.lower():
            parts = filename.lower().split('_tipo')
            if len(parts) > 1:
                original_type = parts[1].replace('.txt', '').upper()[:5]

        # Determinar si coincide
        classified_main = result['tipo_clasificado'][0] if result['tipo_clasificado'] else ''
        original_main = original_type[0] if original_type != 'N/A' else ''
        is_match = classified_main == original_main

        # Preparar respuesta
        response = {
            'success': True,
            'filename': filename,
            'tipo_clasificado': result['tipo_clasificado'],
            'subtipo': result['subtipo'],
            'confianza': result.get('confianza'),
            'original_type': original_type,
            'is_match': is_match,
            'n_lineas': result['n_lineas'],
            'rango_lambda': result['rango_lambda'],
            'method_used': method,
            'lineas_detectadas': ([] if detail_level == 'basic'
                                   else result['lineas_detectadas'] if detail_level == 'debug'
                                   else result['lineas_detectadas'][:20])
        }

        if save_plot and 'plot_path' in result:
            response['plot_url'] = f"/plot/{os.path.basename(result['plot_path'])}"

        if detail_level == 'debug' and 'diagnostics' in result:
            response['diagnostics'] = result['diagnostics']

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/get_metrics')
def get_metrics():
    """Obtiene métricas del modelo actual"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    metadata_path = os.path.join(models_dir, 'metadata.json')
    report_path = os.path.join(models_dir, 'validation_report.txt')
    confusion_matrix_path = os.path.join(models_dir, 'confusion_matrix.png')

    if not os.path.exists(metadata_path):
        return jsonify({
            'success': False,
            'error': 'Modelo no encontrado. Ejecuta el entrenamiento primero.'
        }), 404

    try:
        import json

        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Cargar reporte si existe
        report_text = ''
        accuracy_by_type = {}
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()

            # Parsear accuracy por tipo del reporte
            import re
            pattern = r'(\w):\s+([\d.]+)%\s+\((\d+)/(\d+)\)'
            matches = re.findall(pattern, report_text)
            for match in matches:
                tipo, acc, correct, total = match
                accuracy_by_type[tipo] = {
                    'accuracy': float(acc),
                    'correct': int(correct),
                    'total': int(total)
                }

        # Verificar matriz de confusión
        has_confusion_matrix = os.path.exists(confusion_matrix_path)

        # Top features
        top_features = []
        if 'feature_names' in metadata:
            # Intentar cargar importancias si están disponibles
            # Por ahora, listar los features
            top_features = metadata['feature_names'][:10]

        return jsonify({
            'success': True,
            'accuracy_test': round(metadata.get('accuracy_test', 0) * 100, 2),
            'accuracy_cv_mean': round(metadata.get('accuracy_cv_mean', 0) * 100, 2),
            'accuracy_physical': round(metadata.get('accuracy_physical', 0) * 100, 2),
            'n_train': metadata.get('n_train', 0),
            'n_test': metadata.get('n_test', 0),
            'timestamp': metadata.get('timestamp', 'N/A'),
            'model_type': metadata.get('model_type', 'Árbol de Decisión'),
            'accuracy_by_type': accuracy_by_type,
            'has_confusion_matrix': has_confusion_matrix,
            'top_features': top_features,
            'report_text': report_text
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/list_catalogs')
def list_catalogs():
    """Lista los catálogos de espectros disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    catalogs = []

    def scan_directory(base_path, prefix=''):
        """Escanea un directorio buscando catálogos de espectros"""
        if not os.path.exists(base_path):
            return
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Contar archivos de espectros
                try:
                    spectrum_files = [f for f in os.listdir(item_path) if f.endswith('.txt') and '_tipo' in f.lower()]
                except PermissionError:
                    continue

                if len(spectrum_files) > 0:
                    # Contar tipos
                    types_count = {}
                    for f in spectrum_files:
                        if '_tipo' in f.lower():
                            parts = f.lower().split('_tipo')
                            if len(parts) > 1:
                                tipo = parts[1].replace('.txt', '').upper()
                                main_type = tipo[0] if tipo else '?'
                                types_count[main_type] = types_count.get(main_type, 0) + 1

                    rel_path = prefix + item + '/' if prefix else item + '/'
                    catalogs.append({
                        'name': item,
                        'path': rel_path,
                        'n_files': len(spectrum_files),
                        'types': types_count
                    })

    # Buscar en la raíz del proyecto
    scan_directory(project_root)

    # Buscar en la carpeta data/
    data_path = os.path.join(project_root, 'data')
    scan_directory(data_path, 'data/')

    return jsonify({
        'success': True,
        'catalogs': catalogs
    })


@app.route('/list_catalog_files/<path:catalog>')
def list_catalog_files(catalog):
    """Lista archivos de un catálogo específico con filtro opcional"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, catalog)

    if not os.path.exists(catalog_path) or not os.path.isdir(catalog_path):
        return jsonify({'success': False, 'error': f'Catálogo no encontrado: {catalog}'}), 404

    # Filtro opcional
    filter_text = request.args.get('filter', '').lower()
    filter_type = request.args.get('type', '').upper()
    limit = int(request.args.get('limit', 100))

    files = []
    for f in sorted(os.listdir(catalog_path)):
        if not f.endswith('.txt'):
            continue

        # Aplicar filtros
        if filter_text and filter_text not in f.lower():
            continue

        # Extraer tipo del nombre
        tipo = ''
        if '_tipo' in f.lower():
            parts = f.lower().split('_tipo')
            if len(parts) > 1:
                tipo = parts[1].replace('.txt', '').upper()

        if filter_type and not tipo.startswith(filter_type):
            continue

        files.append({
            'filename': f,
            'path': f'{catalog}/{f}',
            'tipo': tipo
        })

        if len(files) >= limit:
            break

    return jsonify({
        'success': True,
        'catalog': catalog,
        'files': files,
        'total': len(files)
    })


@app.route('/verify_model')
def verify_model():
    """Verifica el estado del modelo"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    checks = {
        'model_file': os.path.exists(os.path.join(models_dir, 'decision_tree.pkl')),
        'metadata': os.path.exists(os.path.join(models_dir, 'metadata.json')),
        'confusion_matrix': os.path.exists(os.path.join(models_dir, 'confusion_matrix.png')),
        'validation_report': os.path.exists(os.path.join(models_dir, 'validation_report.txt')),
        'multi_method_available': MULTI_METHOD_AVAILABLE
    }

    all_ok = all([checks['model_file'], checks['metadata']])

    # Cargar metadata si existe
    model_info = {}
    if checks['metadata']:
        try:
            import json
            with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                model_info = {
                    'accuracy': round(metadata.get('accuracy_test', 0) * 100, 2),
                    'n_samples': metadata.get('n_train', 0) + metadata.get('n_test', 0),
                    'timestamp': metadata.get('timestamp', 'N/A')
                }
        except:
            pass

    return jsonify({
        'success': True,
        'status': 'OK' if all_ok else 'INCOMPLETE',
        'checks': checks,
        'model_info': model_info
    })


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Limpia archivos temporales y resultados"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cleared = {
        'uploads': 0,
        'results': 0
    }

    # Limpiar uploads
    uploads_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            try:
                os.remove(os.path.join(uploads_dir, f))
                cleared['uploads'] += 1
            except:
                pass

    # Limpiar results
    results_dir = app.config['RESULTS_FOLDER']
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            try:
                os.remove(os.path.join(results_dir, f))
                cleared['results'] += 1
            except:
                pass

    return jsonify({
        'success': True,
        'cleared': cleared,
        'message': f"Eliminados {cleared['uploads']} uploads y {cleared['results']} resultados"
    })


@app.route('/confusion_matrix')
def get_confusion_matrix():
    """Devuelve la imagen de la matriz de confusión"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(project_root, 'models', 'confusion_matrix.png')

    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return "Matriz de confusión no encontrada", 404


# ============================================================================
# ÁRBOL INTERACTIVO — devuelve datos crudos del espectro para visualización
# ============================================================================

@app.route('/spectrum_raw', methods=['POST'])
def spectrum_raw():
    """
    Carga un espectro y devuelve arrays wavelength/flux normalizados como JSON.
    Usado por el Árbol Interactivo para mostrar el espectro real en cada paso.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió archivo'}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Formato no válido (.txt, .fits, .fit)'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        wavelengths, flux, error, metadata = load_spectrum_file(filepath)

        if error:
            return jsonify({'success': False, 'error': error}), 400

        # Normalizar al continuo para mostrar correctamente
        flux_norm, _ = normalize_to_continuum(wavelengths, flux)

        # Reducir puntos si el espectro es muy largo (máx 2000 puntos para el SVG)
        n = len(wavelengths)
        if n > 2000:
            step = n // 2000
            wavelengths = wavelengths[::step]
            flux_norm   = flux_norm[::step]

        return jsonify({
            'success': True,
            'wavelength': wavelengths.tolist(),
            'flux': flux_norm.tolist(),
            'wmin': float(wavelengths.min()),
            'wmax': float(wavelengths.max()),
            'filename': filename,
            'n_points': len(wavelengths),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINT: CLASE DE LUMINOSIDAD MK (para el Árbol Interactivo)
# ============================================================================

@app.route('/api/luminosity', methods=['POST'])
def api_luminosity():
    """
    Calcula la clase de luminosidad MK para un espectro ya normalizado.

    Acepta JSON con:
        wavelength   : list[float]  — longitudes de onda en Å
        flux         : list[float]  — flujo normalizado al continuo
        spectral_type: str          — tipo espectral determinado por el árbol
                                      (p. ej. "G2", "B1", "M4")

    Devuelve JSON con:
        success         : bool
        luminosity_class: str   — "Ia", "Ib", "II", "III", "IV" o "V"
        mk_full         : str   — tipo MK completo (p. ej. "G2V", "B1Ib")
        lum_name        : str   — nombre legible de la clase
        indicators      : dict  — razones usadas para el diagnóstico
        error           : str   — presente solo si success == False
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'Se esperaba JSON'}), 400

        wavelength    = np.asarray(data.get('wavelength', []), dtype=float)
        flux          = np.asarray(data.get('flux',       []), dtype=float)
        spectral_type = str(data.get('spectral_type', '')).strip()

        if len(wavelength) < 10 or len(flux) < 10:
            return jsonify({'success': False, 'error': 'Datos de espectro insuficientes'}), 400

        if not spectral_type:
            return jsonify({'success': False, 'error': 'Se requiere spectral_type'}), 400

        # Medir líneas diagnóstico con las ventanas optimizadas
        measurements = measure_diagnostic_lines(wavelength, flux)

        # Importar módulo de luminosidad
        from luminosity_classification import (
            estimate_luminosity_class,
            combine_spectral_and_luminosity
        )
        from spectral_classification_corrected import compute_spectral_ratios

        lum_class = estimate_luminosity_class(measurements, spectral_type)
        mk_full   = combine_spectral_and_luminosity(spectral_type, lum_class)

        # Nombre legible de la clase
        lum_names = {
            'Ia': 'Supergigante muy luminosa',
            'Ib': 'Supergigante',
            'II': 'Gigante brillante',
            'III': 'Gigante',
            'IV': 'Subgigante',
            'V': 'Secuencia principal (enana)',
        }
        lum_name = lum_names.get(lum_class, lum_class)

        # Calcular razones diagnóstico relevantes para mostrar en UI
        ratios = compute_spectral_ratios(measurements)
        key_ratios = {
            k: round(float(v), 3)
            for k, v in ratios.items()
            if k in ('HeI_HeII', 'SrII_FeI', 'CaI_FeI', 'BaII_FeI',
                     'NaI_CaI', 'TiO_CaH', 'NIII_HeII', 'MgIb_FeI')
        }

        return jsonify(convert_numpy_types({
            'success':          True,
            'luminosity_class': lum_class,
            'mk_full':          mk_full,
            'lum_name':         lum_name,
            'indicators':       key_ratios,
        }))

    except ImportError as e:
        return jsonify({'success': False,
                        'error': f'Módulo de luminosidad no disponible: {e}'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# MANEJADORES DE ERROR GLOBALES — siempre devuelven JSON, nunca HTML
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': f'Solicitud incorrecta: {str(e)}'}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': f'Ruta no encontrada: {request.path}'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False, 'error': f'Método no permitido en {request.path}'}), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'Archivo demasiado grande (máximo 2 GB)'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': f'Error interno del servidor: {str(e)}'}), 500


# ── NORMALIZACIÓN DE ESPECTROS CRUDOS ─────────────────────────────────────

def _norm_generar_imagenes(wave, flux, continuum, continuum_std, smooth_factor=1.0):
    """Genera las dos gráficas PNG (base64) y el texto descargable dado un smooth_factor."""
    from scipy.interpolate import UnivariateSpline

    # Aplicar smooth_factor a continuum_std para controlar suavizado de la spline
    std_scaled = continuum_std * smooth_factor

    # Misma lógica que get_smoothed_continuum pero con smooth_factor aplicado
    try:
        mask = ~(np.isclose(std_scaled, 0) | np.isclose(continuum, 0) |
                 np.isnan(std_scaled) | np.isnan(continuum))
        if mask.sum() >= 4:
            w_spl = wave[mask]
            c_spl = continuum[mask]
            s_spl = std_scaled[mask]
            weights = 1.0 / np.where(s_spl > 0, s_spl, 1e-6)
            spl = UnivariateSpline(w_spl, c_spl, w=weights, k=3, ext=3)
            continuum_suave = spl(wave)
        else:
            continuum_suave = continuum.copy()
    except Exception:
        continuum_suave = continuum.copy()

    continuum_suave = np.where(continuum_suave > 0, continuum_suave, 1.0)
    flux_norm = flux / continuum_suave

    # ── Gráfica 1: espectro original + continuo ──────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(wave, flux, color='#4a9eff', linewidth=0.8, label='Espectro original', alpha=0.9)
    ax1.plot(wave, continuum_suave, color='#ff6b35', linewidth=1.5, label='Continuo estimado')
    ax1.set_xlabel('Longitud de onda (Å)')
    ax1.set_ylabel('Flujo relativo')
    ax1.set_title(f'Espectro crudo + continuo estimado  (smooth={smooth_factor:.2f})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig1)
    img1_b64 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    # ── Gráfica 2: espectro normalizado ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(wave, flux_norm, color='#4a9eff', linewidth=0.8, alpha=0.9)
    ax2.axhline(1.0, color='#ff6b35', linewidth=1.0, linestyle='--', label='Continuo = 1.0')
    ax2.set_xlabel('Longitud de onda (Å)')
    ax2.set_ylabel('Flujo normalizado')
    ax2.set_title(f'Espectro normalizado  (smooth={smooth_factor:.2f})')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig2)
    img2_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    # Texto descargable
    lineas = ['# Longitud_de_onda(A)  Flujo_normalizado  Continuo']
    for w, fn, c in zip(wave, flux_norm, continuum_suave):
        lineas.append(f'{w:.4f}\t{fn:.6f}\t{c:.6f}')
    texto_descarga = '\n'.join(lineas)

    return img1_b64, img2_b64, texto_descarga


def _cargar_normalizador(which_weights='active'):
    """Carga el modelo de normalización (lento, ~10 s). Se llama una vez."""
    global _normalizador_nn
    _normalizador_nn = get_suppnet(
        resampling_step=0.05,
        step_size=256,
        norm_only=False,
        which_weights=which_weights
    )


@app.route('/normalizacion_estado')
def normalizacion_estado():
    """Devuelve si el módulo de normalización está disponible y cargado."""
    return jsonify({
        'disponible': NORMALIZADOR_DISPONIBLE,
        'modelo_cargado': _normalizador_nn is not None
    })


@app.route('/normalizacion_cargar', methods=['POST'])
def normalizacion_cargar():
    """Carga (o recarga) el modelo de normalización con el modelo indicado."""
    if not NORMALIZADOR_DISPONIBLE:
        return jsonify({'success': False, 'error': 'Módulo de normalización no encontrado. Verifica la ruta de instalación.'}), 503
    modelo = request.json.get('modelo', 'active')
    try:
        _cargar_normalizador(which_weights=modelo)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/normalizacion_procesar', methods=['POST'])
def normalizacion_procesar():
    """
    Recibe un espectro (.txt dos columnas λ flux), lo normaliza con la red
    neuronal y devuelve: imagen base64 del espectro original+continuo,
    imagen base64 del espectro normalizado, y el texto del archivo descargable.
    """
    if not NORMALIZADOR_DISPONIBLE:
        return jsonify({'error': 'Módulo de normalización no disponible'}), 503
    if _normalizador_nn is None:
        return jsonify({'error': 'Modelo no cargado. Haz clic en "Cargar modelo" primero.'}), 503

    f = request.files.get('archivo')
    if f is None:
        return jsonify({'error': 'No se recibió ningún archivo'}), 400

    try:
        import tempfile
        ext = (f.filename or '').rsplit('.', 1)[-1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext)
        f.save(tmp.name)
        tmp.close()

        try:
            if ext in ('fits', 'fit'):
                # Leer FITS: extraer flujo y reconstruir eje λ con WCS estándar
                from astropy.io import fits as _fits
                with _fits.open(tmp.name, memmap=False) as hdul:
                    data_fits = hdul[0].data
                    header   = hdul[0].header
                crval1 = float(header.get('CRVAL1', 0))
                crpix1 = float(header.get('CRPIX1', 1))
                cdelt1 = float(header.get('CDELT1', 1))
                pixels = np.arange(1, len(data_fits) + 1)
                wave = crval1 + (pixels - crpix1) * cdelt1
                flux = data_fits.astype(float)
            else:
                # Leer texto: dos columnas λ flux
                # Intentar primero con np.loadtxt (rápido); si falla por cabecera
                # de texto sin '#', reintentar saltando filas iniciales no numéricas.
                def _leer_columnas_txt(filepath):
                    try:
                        return np.loadtxt(filepath, comments='#')
                    except ValueError:
                        pass
                    # Detectar cuántas filas iniciales son no numéricas
                    skip = 0
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                skip += 1
                                continue
                            try:
                                parts = line.split()
                                float(parts[0]); float(parts[1])
                                break   # primera fila numérica encontrada
                            except (ValueError, IndexError):
                                skip += 1
                    return np.loadtxt(filepath, comments='#', skiprows=skip)

                data = _leer_columnas_txt(tmp.name)
                if data.ndim != 2 or data.shape[1] < 2:
                    return jsonify({'error': 'El archivo debe tener dos columnas: longitud de onda y flujo'}), 400
                wave = data[:, 0]
                flux = data[:, 1]
        finally:
            os.unlink(tmp.name)

        # Normalizar al mediano antes de pasar a la red (igual que el script original)
        mediana = np.nanmedian(flux)
        if mediana > 0:
            flux = flux / mediana

        # Predecir continuo
        resultado = _normalizador_nn.normalize(wave, flux)
        if len(resultado) == 4:
            continuum, continuum_std, segmentation, segmentation_std = resultado
        else:
            # Fallback por si el modelo se cargó con norm_only=True
            continuum, continuum_std = resultado
            segmentation, segmentation_std = None, None

        # Guardar en caché para ajustes de smooth posteriores
        import uuid as _uuid
        spectrum_id = str(_uuid.uuid4())
        # Mantener solo los últimos 5 espectros para no acumular memoria
        if len(_norm_cache) >= 5:
            oldest = next(iter(_norm_cache))
            del _norm_cache[oldest]
        _norm_cache[spectrum_id] = {
            'wave': wave,
            'flux': flux,
            'continuum': continuum,
            'continuum_std': continuum_std,
            'segmentation': segmentation,
            'segmentation_std': segmentation_std,
        }

        img1_b64, img2_b64, texto_descarga = _norm_generar_imagenes(
            wave, flux, continuum, continuum_std, smooth_factor=1.0)

        return jsonify({
            'imagen_original':    img1_b64,
            'imagen_normalizado': img2_b64,
            'texto_descarga':     texto_descarga,
            'spectrum_id':        spectrum_id,
            'n_puntos':           int(len(wave)),
            'rango_lambda':       f'{wave[0]:.1f} – {wave[-1]:.1f} Å',
        })

    except Exception as e:
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500


@app.route('/normalizacion_ajustar_smooth', methods=['POST'])
def normalizacion_ajustar_smooth():
    """
    Recalcula el continuo y las gráficas con un nuevo smooth_factor sin
    volver a correr la red neuronal (usa el caché del último espectro).
    """
    data = request.json or {}
    spectrum_id  = data.get('spectrum_id')
    smooth_factor = float(data.get('smooth_factor', 1.0))
    smooth_factor = max(0.05, min(smooth_factor, 20.0))  # clamp

    if not spectrum_id or spectrum_id not in _norm_cache:
        return jsonify({'error': 'Espectro no encontrado en caché. Procesa el espectro primero.'}), 404

    cached = _norm_cache[spectrum_id]
    try:
        img1_b64, img2_b64, texto_descarga = _norm_generar_imagenes(
            cached['wave'], cached['flux'],
            cached['continuum'], cached['continuum_std'],
            smooth_factor=smooth_factor,
        )
        return jsonify({
            'imagen_original':   img1_b64,
            'imagen_normalizado': img2_b64,
            'texto_descarga':    texto_descarga,
        })
    except Exception as e:
        return jsonify({'error': f'Error al recalcular: {str(e)}'}), 500

# ──────────────────────────────────────────────────────────────────────────

# ============================================================================
# PANEL DE GESTIÓN DE DATASET
# ============================================================================

import re as _re_dm
from collections import Counter, defaultdict

def _dm_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _dm_resolve_catalog(catalog):
    """Resolve catalog to absolute path, raising ValueError if not a directory."""
    root = _dm_project_root()
    path = catalog if os.path.isabs(catalog) else os.path.join(root, catalog)
    return os.path.normpath(path)

def _dm_extract_sptype(filename):
    """Extract full sptype string from filename (e.g. 'O6pe', 'B3V', 'A5')."""
    m = _re_dm.search(r'_tipo_?([A-Za-z][a-zA-Z0-9\-]*)', filename, _re_dm.IGNORECASE)
    return m.group(1).upper() if m else '?'

def _dm_main_class(sptype):
    """Return first letter of sptype as main class."""
    return sptype[0].upper() if sptype and sptype != '?' else '?'

def _dm_scan_files(catalog_dir, subdir=None):
    """Return list of file-info dicts from catalog_dir (or a subdir of it)."""
    scan_dir = os.path.join(catalog_dir, subdir) if subdir else catalog_dir
    if not os.path.isdir(scan_dir):
        return []
    result = []
    for f in sorted(os.listdir(scan_dir)):
        if not f.endswith('.txt'):
            continue
        fpath = os.path.join(scan_dir, f)
        sptype = _dm_extract_sptype(f)
        result.append({
            'filename': f,
            'sptype':   sptype,
            'class':    _dm_main_class(sptype),
            'size_kb':  round(os.path.getsize(fpath) / 1024, 1),
            'mtime':    datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d'),
            'source':   subdir if subdir else 'catalog',
        })
    return result


@app.route('/dataset/overview')
def dataset_overview():
    catalog = request.args.get('catalog', 'data/elodie/')
    try:
        cdir = _dm_resolve_catalog(catalog)
        if not os.path.isdir(cdir):
            return jsonify({'error': f'Catálogo no encontrado: {catalog}'}), 404

        files      = _dm_scan_files(cdir)
        discarded  = _dm_scan_files(cdir, '_descartados')
        augmented  = _dm_scan_files(cdir, '_augmented')

        class_counts = Counter(f['class'] for f in files)
        total = len(files)
        distribution = sorted(
            [{'class': cls, 'count': cnt,
              'pct': round(cnt / total * 100, 1) if total else 0}
             for cls, cnt in class_counts.items()],
            key=lambda x: x['class']
        )
        max_cls = max(class_counts, key=class_counts.get) if class_counts else None
        min_cls = min(class_counts, key=class_counts.get) if class_counts else None

        return jsonify({
            'total':            total,
            'total_discarded':  len(discarded),
            'total_augmented':  len(augmented),
            'n_classes':        len(class_counts),
            'classes_present':  sorted(class_counts.keys()),
            'most_represented': {'class': max_cls, 'count': class_counts.get(max_cls, 0)} if max_cls else None,
            'least_represented':{'class': min_cls, 'count': class_counts.get(min_cls, 0)} if min_cls else None,
            'distribution':     distribution,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/spectra')
def dataset_spectra():
    catalog           = request.args.get('catalog', 'data/elodie/')
    page              = max(1, int(request.args.get('page', 1)))
    per_page          = min(100, max(5, int(request.args.get('per_page', 25))))
    filter_class      = request.args.get('filter_class', '').upper().strip()
    filter_name       = request.args.get('filter_name', '').strip().lower()
    include_discarded = request.args.get('include_discarded', 'false').lower() == 'true'

    try:
        cdir = _dm_resolve_catalog(catalog)
        if not os.path.isdir(cdir):
            return jsonify({'error': f'Catálogo no encontrado: {catalog}'}), 404

        files = _dm_scan_files(cdir)
        if include_discarded:
            disc = _dm_scan_files(cdir, '_descartados')
            files.extend(disc)

        if filter_class:
            files = [f for f in files if f['class'] == filter_class or f['sptype'].startswith(filter_class)]
        if filter_name:
            files = [f for f in files if filter_name in f['filename'].lower()]

        total     = len(files)
        start     = (page - 1) * per_page
        page_data = files[start: start + per_page]

        return jsonify({
            'total':    total,
            'page':     page,
            'per_page': per_page,
            'pages':    max(1, (total + per_page - 1) // per_page),
            'files':    page_data,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/spectrum_data')
def dataset_spectrum_data():
    catalog  = request.args.get('catalog', 'data/elodie/')
    filename = request.args.get('file', '')
    source   = request.args.get('source', 'catalog')

    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Nombre de archivo inválido'}), 400

    try:
        cdir = _dm_resolve_catalog(catalog)
        subdir_map = {'discarded': '_descartados', 'augmented': '_augmented'}
        sub = subdir_map.get(source, '')
        fpath = os.path.join(cdir, sub, filename) if sub else os.path.join(cdir, filename)

        if not os.path.isfile(fpath):
            return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

        wavelengths, flux, _, _ = load_spectrum_file(fpath)

        # Downsample for plot performance
        if len(wavelengths) > 2000:
            step = len(wavelengths) // 2000
            wavelengths = wavelengths[::step]
            flux        = flux[::step]

        return jsonify({
            'filename': filename,
            'sptype':   _dm_extract_sptype(filename),
            'wave':     [round(float(w), 2) for w in wavelengths],
            'flux':     [round(float(v), 6) for v in flux],
            'wave_min': float(wavelengths.min()),
            'wave_max': float(wavelengths.max()),
            'n_points': len(wavelengths),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/discard', methods=['POST'])
def dataset_discard():
    data    = request.json or {}
    catalog = data.get('catalog', 'data/elodie/')
    files   = data.get('files', [])
    if not files:
        return jsonify({'error': 'No se especificaron archivos'}), 400
    try:
        cdir     = _dm_resolve_catalog(catalog)
        disc_dir = os.path.join(cdir, '_descartados')
        os.makedirs(disc_dir, exist_ok=True)
        moved, errors = [], []
        for fname in files:
            if '..' in fname or '/' in fname or '\\' in fname:
                errors.append(f'{fname}: nombre inválido'); continue
            src = os.path.join(cdir, fname)
            dst = os.path.join(disc_dir, fname)
            if not os.path.isfile(src):
                errors.append(f'{fname}: no encontrado'); continue
            if os.path.exists(dst):
                base, ext = os.path.splitext(fname)
                dst = os.path.join(disc_dir, f'{base}_dup{int(os.path.getmtime(src))}{ext}')
            os.rename(src, dst)
            moved.append(fname)
        return jsonify({'moved': moved, 'errors': errors, 'count': len(moved)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/restore', methods=['POST'])
def dataset_restore():
    data    = request.json or {}
    catalog = data.get('catalog', 'data/elodie/')
    files   = data.get('files', [])
    if not files:
        return jsonify({'error': 'No se especificaron archivos'}), 400
    try:
        cdir     = _dm_resolve_catalog(catalog)
        disc_dir = os.path.join(cdir, '_descartados')
        moved, errors = [], []
        for fname in files:
            if '..' in fname or '/' in fname or '\\' in fname:
                errors.append(f'{fname}: nombre inválido'); continue
            src = os.path.join(disc_dir, fname)
            dst = os.path.join(cdir, fname)
            if not os.path.isfile(src):
                errors.append(f'{fname}: no en _descartados/'); continue
            if os.path.exists(dst):
                errors.append(f'{fname}: ya existe en catálogo'); continue
            os.rename(src, dst)
            moved.append(fname)
        return jsonify({'moved': moved, 'errors': errors, 'count': len(moved)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/reclassify', methods=['POST'])
def dataset_reclassify():
    data      = request.json or {}
    catalog   = data.get('catalog', 'data/elodie/')
    filename  = data.get('file', '')
    new_sptype = data.get('new_sptype', '').strip()
    source    = data.get('source', 'catalog')

    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Nombre de archivo inválido'}), 400
    if not new_sptype:
        return jsonify({'error': 'Nuevo tipo espectral requerido'}), 400

    new_sptype_safe = _re_dm.sub(r'[^A-Za-z0-9]', '', new_sptype)[:12]
    if not new_sptype_safe:
        return jsonify({'error': 'Tipo espectral inválido (use letras y números)'}), 400

    try:
        cdir   = _dm_resolve_catalog(catalog)
        fdir   = os.path.join(cdir, '_descartados') if source == 'discarded' else cdir
        src    = os.path.join(fdir, filename)
        if not os.path.isfile(src):
            return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

        # Replace _tipo<sptype> part in filename
        m = _re_dm.search(r'(_tipo_?)([A-Za-z][a-zA-Z0-9\-]*)(\.(txt|fit|fits)$)',
                          filename, _re_dm.IGNORECASE)
        if m:
            new_filename = filename[:m.start(1)] + m.group(1) + new_sptype_safe + m.group(3)
        else:
            base, ext = os.path.splitext(filename)
            new_filename = f'{base}_tipo{new_sptype_safe}{ext}'

        dst = os.path.join(fdir, new_filename)
        if os.path.exists(dst):
            return jsonify({'error': f'Ya existe un archivo con ese nombre: {new_filename}'}), 409

        os.rename(src, dst)
        return jsonify({'old_filename': filename, 'new_filename': new_filename,
                        'new_sptype': new_sptype_safe})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/quality')
def dataset_quality():
    catalog = request.args.get('catalog', 'data/elodie/')
    try:
        cdir = _dm_resolve_catalog(catalog)
        if not os.path.isdir(cdir):
            return jsonify({'error': f'Catálogo no encontrado: {catalog}'}), 404

        files   = _dm_scan_files(cdir)
        results = []

        for fi in files:
            fpath  = os.path.join(cdir, fi['filename'])
            issues = []
            stats  = {}
            try:
                wave, flux, _, _ = load_spectrum_file(fpath)

                flux_max  = float(np.max(np.abs(flux)))
                flux_mean = float(np.mean(flux))

                if flux_max < 1e-10:
                    issues.append('flujo_cero')
                elif flux_max < 1e-3:
                    issues.append('flujo_muy_bajo')

                if not np.all(np.isfinite(flux)):
                    issues.append(f'nan_inf ({int(np.sum(~np.isfinite(flux)))} pts)')

                if float(wave.max()) - float(wave.min()) < 100:
                    issues.append('rango_lambda_corto')

                if len(wave) < 50:
                    issues.append(f'pocos_puntos ({len(wave)})')

                # SNR estimate via residuals of moving average
                if len(flux) >= 20:
                    win = min(50, max(3, len(flux) // 8))
                    smoothed  = np.convolve(flux, np.ones(win) / win, mode='same')
                    res_std   = float(np.std(flux - smoothed)) + 1e-12
                    snr       = float(abs(flux_mean)) / res_std
                    stats['snr_est'] = round(snr, 1)
                    if snr < 3:
                        issues.append(f'snr_bajo ({snr:.1f})')

                stats.update({
                    'wave_min': round(float(wave.min()), 1),
                    'wave_max': round(float(wave.max()), 1),
                    'n_points': len(wave),
                    'flux_max': round(flux_max, 4),
                })
            except Exception as ex:
                issues.append(f'error_lectura: {str(ex)[:60]}')

            results.append({**fi, 'issues': issues, 'ok': len(issues) == 0, **stats})

        n_ok = sum(1 for r in results if r['ok'])
        return jsonify({'total': len(results), 'ok': n_ok,
                        'with_issues': len(results) - n_ok, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/augment_preview', methods=['POST'])
def dataset_augment_preview():
    data               = request.json or {}
    catalog            = data.get('catalog', 'data/elodie/')
    target_per_class   = data.get('target_per_class')
    max_augment_ratio  = int(data.get('max_augment_ratio', 5))
    try:
        cdir         = _dm_resolve_catalog(catalog)
        files        = _dm_scan_files(cdir)
        class_counts = Counter(f['class'] for f in files)
        if not class_counts:
            return jsonify({'error': 'No hay espectros en el catálogo'}), 400

        import statistics as _stats
        median_count = int(_stats.median(class_counts.values()))
        target       = int(target_per_class) if target_per_class else median_count

        preview = []
        for cls in sorted(class_counts):
            orig    = class_counts[cls]
            max_new = orig * max_augment_ratio - orig
            needed  = max(0, min(target - orig, max_new))
            preview.append({
                'class':     cls,
                'original':  orig,
                'synthetic': needed,
                'total':     orig + needed,
                'capped':    (target - orig) > max_new and orig < target,
            })

        return jsonify({
            'target_per_class': target,
            'median_count':     median_count,
            'total_original':   len(files),
            'total_synthetic':  sum(p['synthetic'] for p in preview),
            'total_after':      sum(p['total']     for p in preview),
            'preview':          preview,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/augment_apply', methods=['POST'])
def dataset_augment_apply():
    data              = request.json or {}
    catalog           = data.get('catalog', 'data/elodie/')
    target_per_class  = data.get('target_per_class')
    max_augment_ratio = int(data.get('max_augment_ratio', 5))
    noise_factor      = float(data.get('noise_factor', 0.015))

    try:
        cdir    = _dm_resolve_catalog(catalog)
        aug_dir = os.path.join(cdir, '_augmented')
        os.makedirs(aug_dir, exist_ok=True)

        files = _dm_scan_files(cdir)
        files_by_class = defaultdict(list)
        for f in files:
            files_by_class[f['class']].append(f['filename'])

        class_counts = {cls: len(v) for cls, v in files_by_class.items()}
        if not class_counts:
            return jsonify({'error': 'No hay espectros en el catálogo'}), 400

        import statistics as _stats
        median_count = int(_stats.median(class_counts.values()))
        target       = int(target_per_class) if target_per_class else median_count
        rng          = np.random.RandomState(42)
        generated    = []

        for cls, flist in sorted(files_by_class.items()):
            orig    = len(flist)
            if orig >= target:
                continue
            max_new = orig * max_augment_ratio - orig
            n_new   = max(0, min(target - orig, max_new))
            if n_new == 0:
                continue

            # Load class spectra
            class_spectra = []
            for fname in flist:
                try:
                    w, f_arr, _, _ = load_spectrum_file(os.path.join(cdir, fname))
                    class_spectra.append((w, f_arr, fname))
                except Exception:
                    pass
            if not class_spectra:
                continue

            for i in range(n_new):
                src_w, src_f, src_name = class_spectra[rng.randint(len(class_spectra))]
                noise    = rng.normal(0, noise_factor * (np.std(src_f) + 1e-8),
                                      size=src_f.shape).astype(np.float32)
                syn_flux = src_f + noise
                base     = os.path.splitext(src_name)[0]
                syn_name = f'{base}_syn{i+1:03d}.txt'
                syn_path = os.path.join(aug_dir, syn_name)
                lines    = ['Longitud_de_onda_A,espectro']
                for w_v, f_v in zip(src_w, syn_flux):
                    lines.append(f'{w_v:.4f},{f_v:.6g}')
                with open(syn_path, 'w') as fout:
                    fout.write('\n'.join(lines))
                generated.append({'file': syn_name, 'class': cls, 'source': src_name})

        return jsonify({
            'generated':       len(generated),
            'target_per_class': target,
            'aug_dir':         '_augmented/',
            'files':           generated,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/export')
def dataset_export():
    catalog = request.args.get('catalog', 'data/elodie/')
    fmt     = request.args.get('format', 'csv')
    try:
        cdir      = _dm_resolve_catalog(catalog)
        active    = _dm_scan_files(cdir)
        discarded = _dm_scan_files(cdir, '_descartados')
        augmented = _dm_scan_files(cdir, '_augmented')
        all_files = (
            [{'status': 'active',    **f} for f in active]   +
            [{'status': 'discarded', **f} for f in discarded] +
            [{'status': 'augmented', **f} for f in augmented]
        )
        import io as _io_exp
        if fmt == 'json':
            content = json.dumps({
                'catalog':     catalog,
                'exported_at': datetime.now().isoformat(),
                'summary':     {'active': len(active), 'discarded': len(discarded),
                                'augmented': len(augmented)},
                'files': all_files,
            }, indent=2)
            return send_file(_io_exp.BytesIO(content.encode()),
                             mimetype='application/json', as_attachment=True,
                             download_name='dataset_export.json')
        else:
            lines = ['status,filename,sptype,class,size_kb,mtime']
            for f in all_files:
                lines.append(f'"{f["status"]}","{f["filename"]}","{f["sptype"]}",'
                             f'"{f["class"]}",{f["size_kb"]},"{f["mtime"]}"')
            content = '\n'.join(lines)
            return send_file(_io_exp.BytesIO(content.encode()),
                             mimetype='text/csv', as_attachment=True,
                             download_name='dataset_export.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    print("="*70)
    print("CLASIFICACION ESPECTRAL - Servidor Web")
    print("="*70)
    print(f"Lineas espectrales configuradas: {len(SPECTRAL_LINES)}")
    print(f"Directorio de subidas: {app.config['UPLOAD_FOLDER']}")
    print(f"Directorio de resultados: {app.config['RESULTS_FOLDER']}")
    print("\nIniciando servidor en: http://localhost:5000")
    print("Presiona Ctrl+C para detener")
    print("="*70)

    app.run(debug=False, host='0.0.0.0', port=5000)
