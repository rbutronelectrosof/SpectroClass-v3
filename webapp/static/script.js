// ===========================
// CLASIFICACIÓN ESPECTRAL v2.0 - JavaScript Mejorado
// ===========================

// Variables globales
let currentFile = null;
let batchFiles = [];
let batchResults = null;

// Pesos globales de votación (modificables desde el panel de utilidades)
let globalWeights = {
    physical:      0.10,
    decision_tree: 0.40,
    template:      0.10,
    knn:           0.20,
    cnn_1d:        0.20,
    cnn_2d:        0.00,
};

// ===========================
// MODAL DE AYUDA
// ===========================

function openModal(id) {
    document.getElementById(id).classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeModal(id) {
    document.getElementById(id).classList.remove('open');
    document.body.style.overflow = '';
}

function closeModalOutside(event, id) {
    // Cerrar solo si se hizo clic en el overlay (fuera del modal-box)
    if (event.target === document.getElementById(id)) {
        closeModal(id);
    }
}

// Cerrar con Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.open').forEach(m => {
            closeModal(m.id);
        });
    }
});

// ===========================
// INICIALIZACIÓN
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupSingleAnalysis();
    setupBatchProcessing();
    setupFitsExtractor();
    initPreambuloUpload();
    dmSetupSubnav();
});

// ===========================
// TABS NAVIGATION
// ===========================

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');

            // Recargar banner de modelo actual al entrar en Herramientas
            if (tabName === 'herramientas') loadCurrentModelBanner();
        });
    });
}

// ===========================
// SINGLE SPECTRUM ANALYSIS
// ===========================

function setupSingleAnalysis() {
    const dropZone = document.getElementById('dropZoneSingle');
    const fileInput = document.getElementById('fileInputSingle');
    const selectedFileDiv = document.getElementById('selectedFile');
    const removeBtn = document.getElementById('removeFileBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnSelectFile = document.getElementById('btnSelectFile');

    // Botón "Seleccionar Archivo" - detener propagación para evitar doble disparo
    btnSelectFile.addEventListener('click', (e) => {
        e.stopPropagation();  // Evitar que el click propague al dropZone
        if (currentFile) return;
        fileInput.click();
    });

    // Click en zona de arrastrar (pero no en el botón)
    dropZone.addEventListener('click', (e) => {
        // Si el click fue en el botón, ignorar (ya lo maneja el botón)
        if (e.target === btnSelectFile || e.target.closest('.btn-upload')) return;
        if (currentFile) return;
        fileInput.click();
    });

    // Drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleSingleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSingleFile(e.target.files[0]);
        }
        // Resetear input para permitir seleccionar el mismo archivo de nuevo si es necesario
        e.target.value = '';
    });

    // Remove file
    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';  // Resetear input
        selectedFileDiv.style.display = 'none';
        dropZone.style.display = 'block';
        document.getElementById('resultArea').style.display = 'none';
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeSingleSpectrum);

    // Another spectrum button
    document.getElementById('analyzeAnotherBtn').addEventListener('click', (e) => {
        e.preventDefault();
        currentFile = null;
        fileInput.value = '';  // Resetear input
        selectedFileDiv.style.display = 'none';
        dropZone.style.display = 'block';
        document.getElementById('resultArea').style.display = 'none';

        // Scroll al inicio
        document.querySelector('.upload-box').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Toggle buttons
    setupToggleButtons();
}

function handleSingleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['txt', 'fits', 'fit'].includes(ext)) {
        alert('Formato no permitido. Solo .txt, .fits, .fit');
        return;
    }

    currentFile = file;

    // Update UI
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = `${(file.size / 1024).toFixed(2)} KB`;
    document.getElementById('dropZoneSingle').style.display = 'none';
    document.getElementById('selectedFile').style.display = 'block';
}

async function analyzeSingleSpectrum() {
    if (!currentFile) return;

    // Hide result area
    document.getElementById('resultArea').style.display = 'none';

    // Show processing
    const processingDiv = document.getElementById('processingIndicator');
    processingDiv.style.display = 'block';

    // Scroll to processing
    processingDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Leer configuración de votación (usa pesos globales si ya fueron configurados)
    const includeNeural = document.getElementById('nnIncludeInVoting')?.checked ?? true;

    // Create form data
    const formData = new FormData();
    formData.append('files[]', currentFile);
    formData.append('include_neural',  includeNeural ? '1' : '0');
    formData.append('physical_weight', globalWeights.physical.toString());
    formData.append('dt_weight',       globalWeights.decision_tree.toString());
    formData.append('template_weight', globalWeights.template.toString());
    formData.append('knn_weight',      globalWeights.knn.toString());
    formData.append('cnn_1d_weight',   globalWeights.cnn_1d.toString());
    formData.append('cnn_2d_weight',   globalWeights.cnn_2d.toString());

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide processing
        processingDiv.style.display = 'none';

        if (data.results && data.results.length > 0) {
            displaySingleResult(data.results[0]);
        } else if (data.errors && data.errors.length > 0) {
            alert(`Error: ${data.errors[0].error}`);
        }

    } catch (error) {
        processingDiv.style.display = 'none';
        alert(`Error al procesar: ${error.message}`);
    }
}

function displaySingleResult(result) {
    const resultArea = document.getElementById('resultArea');
    resultArea.style.display = 'block';
    resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // 1. Type and subtype
    // Mostrar subtipo completo en el badge principal (ej. "F5" en lugar de solo "F")
    const subtipo = result.subtipo || '';
    const subtipoDisplay = subtipo && subtipo !== result.tipo_clasificado ? subtipo : result.tipo_clasificado;
    document.getElementById('typeBadge').textContent = subtipoDisplay;

    // 1b. Clase de luminosidad MK
    const diag0 = result.diagnostics || {};
    const lumClass = diag0.luminosity_class || '';
    const mkFull   = diag0.mk_full || '';
    const mkFullRow = document.getElementById('mkFullRow');
    if (lumClass && mkFull) {
        const lumNames = {
            'Ia': 'Supergigante muy luminosa', 'Ib': 'Supergigante',
            'II': 'Gigante brillante', 'III': 'Gigante',
            'IV': 'Subgigante', 'V': 'Secuencia principal'
        };
        document.getElementById('mkFullBadge').textContent = mkFull;
        document.getElementById('lumClassName').textContent = lumNames[lumClass] || lumClass;
        mkFullRow.style.display = 'flex';
    } else {
        mkFullRow.style.display = 'none';
    }

    // 1c. FITS info card — debajo de la clasificación
    const fitsInfoCard = document.getElementById('fitsInfoCard');
    const fitsInfoBody = document.getElementById('fitsInfoBody');
    if (fitsInfoCard && fitsInfoBody) {
        if (result.fits_summary && result.fits_summary.length > 0) {
            const _gl = {
                obs:  { label: '📡 Observación',  color: '#3b82f6' },
                spec: { label: '📊 Espectro',      color: '#10b981' },
                est:  { label: '⭐ Estelar',        color: '#f59e0b' },
                inst: { label: '🔬 Instrumento',   color: '#8b5cf6' },
                astr: { label: '🌐 Astrometría',   color: '#06b6d4' },
            };
            const grp = {};
            result.fits_summary.forEach(item => {
                if (!grp[item.grupo]) grp[item.grupo] = [];
                grp[item.grupo].push(item);
            });
            let html = '<div class="fits-card-grid">';
            for (const [gkey, items] of Object.entries(grp)) {
                const gi = _gl[gkey] || { label: gkey, color: '#64748b' };
                html += `<div class="fits-card-group">
                    <div class="fits-card-group-title" style="color:${gi.color}">${gi.label}</div>`;
                items.forEach(item => {
                    html += `<div class="fits-card-item">
                        <span class="fits-card-label">${item.label}</span>
                        <span class="fits-card-value">${item.valor}</span>
                    </div>`;
                });
                html += `</div>`;
            }
            html += '</div>';
            fitsInfoBody.innerHTML = html;
        } else {
            fitsInfoBody.innerHTML = '<p class="fits-not-fits">⚠️ No es un archivo FITS — sin cabecera disponible.</p>';
        }
    }

    // Indicador de discrepancia física si difiere el tipo del físico
    const tipoFisico = result.tipo_fisico;
    const discrepanciaEl = document.getElementById('physicalDiscrepancy');
    if (discrepanciaEl) {
        if (tipoFisico && tipoFisico !== result.tipo_clasificado) {
            discrepanciaEl.textContent =
                `(Físico estimó: ${result.subtipo_fisico || tipoFisico} — superado por voto ML+KNN)`;
            discrepanciaEl.style.display = 'block';
        } else {
            discrepanciaEl.style.display = 'none';
        }
    }

    // 2. Confidence (if available)
    if (result.confianza !== undefined) {
        const confidence = result.confianza;
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');
        const confidenceNote = document.getElementById('confidenceNote');

        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
        }, 100);

        confidenceText.textContent = `${confidence.toFixed(1)}%`;

        if (confidence >= 80) {
            confidenceNote.textContent = '✅ Alta confianza - Los métodos coinciden';
            confidenceNote.className = 'confidence-note high';
        } else if (confidence >= 50) {
            confidenceNote.textContent = '⚠️ Confianza media - Revisar alternativas';
            confidenceNote.className = 'confidence-note medium';
        } else {
            confidenceNote.textContent = '❌ Baja confianza - Revisar manualmente';
            confidenceNote.className = 'confidence-note low';
        }
    } else {
        document.querySelector('.confidence-section').style.display = 'none';
    }

    // 3. Alternatives (if available)
    if (result.alternativas && result.alternativas.length > 0) {
        document.getElementById('alternativesCard').style.display = 'block';
        const alternativesList = document.getElementById('alternativesList');
        alternativesList.innerHTML = '';

        alternativesList.innerHTML = result.alternativas.map((alt, idx) => {
            const tipo = alt.subtipo && alt.subtipo !== alt.tipo ? alt.subtipo : alt.tipo;
            const just = alt.justificacion || '';
            return `<div class="alt-chip rank-${idx + 1}">
                        <span class="alt-chip-tipo">${tipo}</span>
                        <span class="alt-chip-pct">${alt.confianza.toFixed(1)}%</span>
                        ${just ? `<span class="alt-chip-just">— ${just}</span>` : ''}
                    </div>`;
        }).join('');
    }

    // 4. Decision tree card — lineas_usadas, justificacion, advertencias (Fix 4 + Fix 6)
    const diag = result.diagnostics || {};
    const lineasUsadas = diag.lineas_usadas || [];
    const justificacion = diag.justificacion || '';
    const advertencias = diag.advertencias || [];
    const tipoClasificado = result.tipo_clasificado || '';
    const coversBlue = diag.covers_blue;

    if (lineasUsadas.length > 0 || justificacion) {
        document.getElementById('decisionTreeCard').style.display = 'block';
        const _expCard = document.getElementById('exportResultsCard');
        if (_expCard) _expCard.style.display = 'block';

        // Fix 6: banner criterio A
        const criterioABanner = document.getElementById('criterioABanner');
        if (tipoClasificado === 'A' || tipoClasificado === 'F') {
            if (coversBlue === true) {
                criterioABanner.style.display = 'block';
                criterioABanner.className = 'criterio-banner criterio-canonico';
                criterioABanner.innerHTML =
                    '✅ <b>Criterio canónico aplicado:</b> Ca II K 3933 Å vs H&epsilon; 3970 Å ' +
                    '(espectro cubre la región azul &lambda; &lt; 3950 Å)';
            } else if (coversBlue === false) {
                criterioABanner.style.display = 'block';
                criterioABanner.className = 'criterio-banner criterio-alternativo';
                criterioABanner.innerHTML =
                    '⚠️ <b>Criterio alternativo aplicado:</b> Ca II K 3933 Å fuera del rango ' +
                    'espectral. Se usaron Balmer (H&beta;, H&gamma;, H&delta;), Mg II 4481 y ' +
                    'Fe I como indicadores. <b>Confianza reducida a 75%.</b>';
            } else {
                criterioABanner.style.display = 'none';
            }
        } else {
            criterioABanner.style.display = 'none';
        }

        // Líneas usadas en la decisión
        // Separamos las líneas del árbol de decisión (DT) de las del clasificador físico
        const lineasUsadasGrid = document.getElementById('lineasUsadasGrid');
        lineasUsadasGrid.innerHTML = '';
        const dtLineasSet = new Set((diag.dt_lineas || []).map(l => l.toLowerCase()));

        // Primero las líneas DT (features del árbol), luego las del clasificador físico
        const dtLineas = (diag.dt_lineas || []).slice().sort();
        const fisLineas = lineasUsadas.filter(l => !dtLineasSet.has(l.toLowerCase())).sort();

        if (dtLineas.length > 0) {
            const dtHeader = document.createElement('div');
            dtHeader.style.cssText = 'width:100%;font-size:10px;color:#94a3b8;margin-bottom:4px;margin-top:2px;font-weight:600;letter-spacing:.5px;text-transform:uppercase';
            dtHeader.textContent = 'Features del árbol de decisión';
            lineasUsadasGrid.appendChild(dtHeader);
            dtLineas.forEach(linea => {
                const el = document.createElement('span');
                el.className = 'linea-usada-badge linea-dt-badge';
                el.textContent = linea;
                lineasUsadasGrid.appendChild(el);
            });
        }
        if (fisLineas.length > 0) {
            const fisHeader = document.createElement('div');
            fisHeader.style.cssText = 'width:100%;font-size:10px;color:#94a3b8;margin-bottom:4px;margin-top:8px;font-weight:600;letter-spacing:.5px;text-transform:uppercase';
            fisHeader.textContent = 'Clasificador físico';
            lineasUsadasGrid.appendChild(fisHeader);
            fisLineas.forEach(linea => {
                const el = document.createElement('span');
                el.className = 'linea-usada-badge';
                el.textContent = linea;
                lineasUsadasGrid.appendChild(el);
            });
        }

        // Pasos del árbol (justificación separada por " | ")
        const justificacionList = document.getElementById('justificacionList');
        justificacionList.innerHTML = '';
        const pasos = justificacion.split(' | ').filter(p => p.trim());
        pasos.forEach(paso => {
            const li = document.createElement('li');
            li.textContent = paso;
            justificacionList.appendChild(li);
        });

        // Advertencias
        const advertenciasSection = document.getElementById('advertenciasSection');
        const advertenciasList = document.getElementById('advertenciasList');
        if (advertencias.length > 0) {
            advertenciasSection.style.display = 'block';
            advertenciasList.innerHTML = '';
            advertencias.forEach(adv => {
                const li = document.createElement('li');
                li.textContent = adv;
                advertenciasList.appendChild(li);
            });
        } else {
            advertenciasSection.style.display = 'none';
        }

    }

    // 5. Lines detected — Fix 1: mostrar TODAS (sin límite de 12)
    document.getElementById('linesCount').textContent = result.n_lineas;
    document.getElementById('wavelengthRange').textContent =
        `${result.rango_lambda[0]}-${result.rango_lambda[1]}`;

    const linesGrid = document.getElementById('linesGrid');
    linesGrid.innerHTML = '';

    // Conjunto de líneas usadas en la decisión para resaltarlas
    const lineasUsadasSet = new Set((diag.lineas_usadas || []).map(l => l.replace(/ /g, '_')));

    result.lineas_detectadas.forEach(line => {   // Fix 1: todas las líneas, sin slice
        const lineDiv = document.createElement('div');
        const nombreClave = line.nombre.replace(/ /g, '_');
        const esDecisoria = lineasUsadasSet.has(nombreClave);
        lineDiv.className = 'line-item' + (esDecisoria ? ' line-item-key' : '');
        lineDiv.innerHTML = `
            <div class="line-name">${line.nombre}${esDecisoria ? ' <span class="key-badge">clave</span>' : ''}</div>
            <div class="line-details">
                ${line.longitud_onda.toFixed(2)} Å<br>
                EW: <span class="line-ew">${line.ancho_equivalente.toFixed(3)} Å</span>
            </div>
        `;
        // Click en tarjeta de línea → zoom en espectro del clasificador + switch a sub-tab espectro
        lineDiv.style.cursor = 'pointer';
        const _lambda = line.longitud_onda;
        const _nombre = line.nombre;
        lineDiv.addEventListener('click', () => {
            clasifZomarLinea(_lambda, _nombre);
            document.getElementById('clasifSVG')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
        linesGrid.appendChild(lineDiv);
    });

    // Spectrum display is handled entirely by clasifSVG (SVG-based)

    // 7. Method details card removed from UI

    // Download button
    document.getElementById('downloadPlotBtn').onclick = () => {
        window.open(`/plot/${encodeURIComponent(result.filename)}_plot.png`, '_blank');
    };

    // Poblar la pestaña Preámbulo & Líneas
    populatePreambulo(result);

    // ── Espectro SVG en sub-tab Clasificador ─────────────────────────────────
    _clasifZoom = null;  // resetear zoom al cargar nuevo resultado
    _clasifSpecData        = result.spectrum_data || null;
    _clasifTodasLineas     = result.todas_lineas || result.lineas_detectadas || [];
    _clasifLineasUsadasSet = new Set((diag.lineas_usadas || []).map(l => l.replace(/ /g, '_')));

    // Resetear barra de zoom y hint
    const _czBar  = document.getElementById('clasifZoomBar');
    const _czHint = document.getElementById('clasifZoomHint');
    if (_czBar)  _czBar.style.display  = 'none';
    if (_czHint) _czHint.style.display = 'block';
    document.querySelectorAll('#claLv1,#claLv2').forEach(b => b.classList.remove('active'));

    // Sin subtabs — todo se muestra en tarjetas apiladas

    if (_clasifSpecData) {
        requestAnimationFrame(() => {
            try {
                dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
                dibujarMiniSpec(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
            } catch(err) {
                console.error('dibujarClasifEspectro:', err);
            }
        });
    }

    // Barras de intensidad
    renderClasifLinesBars(result.lineas_detectadas, _clasifLineasUsadasSet);
}

// ═══════════════════════════════════════════════════════════
// PESTAÑA PREÁMBULO & LÍNEAS
// ═══════════════════════════════════════════════════════════

let _todasLineasData      = [];    // cache para filtrado sin re-render
let _preambuloZoom        = null;  // null = vista completa | {lmin, lmax, lambda, label}
let _preambuloSpecData    = null;  // cache para redibujar con zoom
let _preambuloTodasLineas = [];    // cache para redibujar con zoom

// ── Estado del espectro en pestaña Clasificador ───────────────────────────────
let _clasifSpecData       = null;  // spectrum_data del resultado actual
let _clasifTodasLineas    = [];    // todas las líneas
let _clasifLineasUsadasSet = new Set(); // nombres de líneas clave
let _clasifZoom           = null;  // null = vista completa | {lambda, label, level, lmin, lmax}

// ═══════════════════════════════════════════════════════════
// CLASIFICADOR — SUB-TABS
// ═══════════════════════════════════════════════════════════

function switchResultSubtab(tab) {
    document.querySelectorAll('.result-subtab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subtab === tab);
    });
    document.querySelectorAll('.result-subtab-pane').forEach(pane => {
        pane.style.display = pane.id === `subtab-${tab}` ? '' : 'none';
    });
    // Redibujar SVG al cambiar a la pestaña espectro
    if ((tab === 'espectro' || tab === 'spectrum') && _clasifSpecData) {
        requestAnimationFrame(() => {
            dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
        });
    }
}

// ═══════════════════════════════════════════════════════════
// CLASIFICADOR — ESPECTRO SVG
// ═══════════════════════════════════════════════════════════

function dibujarClasifEspectro(specData, todasLineas, lineasUsadasSet) {
    const svg = document.getElementById('clasifSVG');
    if (!svg) return;

    _clasifSpecData        = specData;
    _clasifTodasLineas     = todasLineas;
    _clasifLineasUsadasSet = lineasUsadasSet;

    const W      = 900;
    const X_LEFT = 55;
    const Y_TOP  = 210;
    const Y_BOT  = 360;
    const Y_TICK = 366;
    const Y_TICK2= 375;
    const Y_LABEL= 388;

    // Rango λ
    let LMIN, LMAX;
    if (_clasifZoom) {
        LMIN = _clasifZoom.lmin;
        LMAX = _clasifZoom.lmax;
    } else {
        const margin = (specData.wmax - specData.wmin) * 0.01;
        LMIN = specData.wmin - margin;
        LMAX = specData.wmax + margin;
        todasLineas.forEach(lin => {
            if (lin.longitud_onda < 6000 && lin.longitud_onda - 60 < LMIN) {
                LMIN = lin.longitud_onda - 60;
            }
        });
    }

    const lambdaToX = l => X_LEFT + ((l - LMIN) / (LMAX - LMIN)) * (W - X_LEFT);

    const wArr = specData.wavelength;
    const fArr = specData.flux;
    let fMin = Infinity, fMax = -Infinity;
    const ptsVisible = [];
    for (let i = 0; i < wArr.length; i++) {
        if (wArr[i] >= LMIN && wArr[i] <= LMAX && Number.isFinite(fArr[i])) {
            if (fArr[i] < fMin) fMin = fArr[i];
            if (fArr[i] > fMax) fMax = fArr[i];
            ptsVisible.push([wArr[i], fArr[i]]);
        }
    }
    if (ptsVisible.length < 2 || !Number.isFinite(fMin) || fMax <= fMin) {
        fMin = 0; fMax = 1.5;
    }
    const scaleF = f => Y_TOP + (1 - (f - fMin) / (fMax - fMin)) * (Y_BOT - Y_TOP);

    let s = '';

    // Defs
    s += `<defs>
      <linearGradient id="cgSpecGrad" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%"   stop-color="#7b00ff"/>
        <stop offset="15%"  stop-color="#4466ff"/>
        <stop offset="30%"  stop-color="#44aaff"/>
        <stop offset="45%"  stop-color="#44ffaa"/>
        <stop offset="60%"  stop-color="#aaff44"/>
        <stop offset="75%"  stop-color="#ffdd00"/>
        <stop offset="88%"  stop-color="#ff6600"/>
        <stop offset="100%" stop-color="#cc0000"/>
      </linearGradient>
      <clipPath id="cgClip">
        <rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"/>
      </clipPath>
    </defs>`;

    // Franja cromática de fondo
    s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"
            fill="url(#cgSpecGrad)" opacity="0.10" rx="4"/>`;

    // Ejes
    s += `<line x1="${X_LEFT}" y1="${Y_BOT}" x2="${W}" y2="${Y_BOT}" stroke="#333" stroke-width="1"/>`;
    s += `<line x1="${X_LEFT}" y1="${Y_TOP}" x2="${X_LEFT}" y2="${Y_BOT}" stroke="#555" stroke-width="1.5"/>`;

    // Eje Y (3 ticks)
    const yRange = fMax - fMin;
    let yTickStep = yRange <= 0.3 ? 0.05 : yRange <= 0.6 ? 0.1 : yRange <= 1.2 ? 0.2 : yRange <= 3.0 ? 0.5 : 1.0;
    const yTickStart = Math.ceil(fMin / yTickStep) * yTickStep;
    for (let ti = 0; ti < 20; ti++) {
        const fVal = Math.round((yTickStart + ti * yTickStep) * 1e6) / 1e6;
        if (fVal > fMax + 1e-9) break;
        const y = scaleF(fVal);
        if (y < Y_TOP - 5 || y > Y_BOT + 5) continue;
        s += `<line x1="${X_LEFT}" y1="${y.toFixed(1)}" x2="${W}" y2="${y.toFixed(1)}"
                stroke="#2a3040" stroke-width="0.8" opacity="0.5"/>`;
        s += `<line x1="${(X_LEFT-5).toFixed(1)}" y1="${y.toFixed(1)}" x2="${X_LEFT}" y2="${y.toFixed(1)}"
                stroke="#667" stroke-width="1.2"/>`;
        s += `<text x="${(X_LEFT-7).toFixed(1)}" y="${(y+4).toFixed(1)}" text-anchor="end"
                font-size="10" fill="#778" font-family="monospace">${fVal.toFixed(1)}</text>`;
    }

    // Espectro
    if (ptsVisible.length >= 2) {
        const step = Math.max(1, Math.floor(ptsVisible.length / 800));
        const pts  = ptsVisible
            .filter((_, i) => i % step === 0)
            .map(([w, f]) => `${lambdaToX(w).toFixed(1)},${scaleF(f).toFixed(1)}`)
            .join(' ');
        const ptsFill = `${X_LEFT},${Y_BOT} ` + pts + ` ${W},${Y_BOT}`;
        s += `<polygon points="${ptsFill}" fill="#7ecfff" opacity="0.07" clip-path="url(#cgClip)"/>`;
        s += `<polyline points="${pts}" fill="none" stroke="#7ecfff"
                stroke-width="2" opacity="0.92" clip-path="url(#cgClip)"/>`;
        const yContRef = scaleF(1.0);
        if (yContRef >= Y_TOP && yContRef <= Y_BOT) {
            s += `<line x1="${X_LEFT}" y1="${yContRef.toFixed(1)}" x2="${W}" y2="${yContRef.toFixed(1)}"
                    stroke="#ffffff" stroke-width="0.8" stroke-dasharray="6,10" opacity="0.22"/>`;
        }
    }

    // Eje λ
    const rango = LMAX - LMIN;
    const tickStep = rango > 1500 ? 500 : rango > 500 ? 200 : rango > 200 ? 50 : 20;
    const tickStart = Math.ceil(LMIN / tickStep) * tickStep;
    for (let l = tickStart; l <= LMAX; l += tickStep) {
        const x = lambdaToX(l);
        if (x < X_LEFT || x > W) continue;
        s += `<line x1="${x.toFixed(1)}" y1="${Y_TICK}" x2="${x.toFixed(1)}" y2="${Y_TICK2}" stroke="#444" stroke-width="1.5"/>`;
        s += `<text x="${x.toFixed(1)}" y="${Y_LABEL}" text-anchor="middle" font-size="12" fill="#778">${l}</text>`;
    }
    s += `<text x="${((X_LEFT + W) / 2).toFixed(1)}" y="${Y_LABEL + 14}" text-anchor="middle"
            font-size="11" fill="#556">λ (Å)</text>`;

    // Región sin cobertura
    if (specData.wmin > LMIN) {
        const xCovStart = lambdaToX(specData.wmin);
        s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${(xCovStart - X_LEFT).toFixed(1)}"
                height="${Y_BOT - Y_TOP}" fill="#0d1117" opacity="0.55"/>`;
        s += `<line x1="${xCovStart.toFixed(1)}" y1="${Y_TOP}"
                x2="${xCovStart.toFixed(1)}" y2="${Y_BOT}"
                stroke="#475569" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.70"/>`;
    }

    // Líneas espectrales — escalonamiento anti-solapamiento
    const lineasOrdenadas = [...todasLineas]
        .filter(l => l.ancho_equivalente > 0.05)
        .sort((a, b) => a.longitud_onda - b.longitud_onda);

    const _xPos    = lineasOrdenadas.map(lin => lambdaToX(lin.longitud_onda));
    const _slotOf  = new Array(lineasOrdenadas.length).fill(0);
    const _assigned= new Array(lineasOrdenadas.length).fill(false);

    // Pasada 1: líneas clave primero → ocupan slots bajos (más cerca del espectro)
    for (let _i = 0; _i < lineasOrdenadas.length; _i++) {
        const _esKey = lineasUsadasSet.has(lineasOrdenadas[_i].nombre.replace(/ /g, '_'));
        if (!_esKey) continue;
        const _used = new Set();
        for (let _j = 0; _j < lineasOrdenadas.length; _j++) {
            if (_j === _i || !_assigned[_j]) continue;
            if (Math.abs(_xPos[_j] - _xPos[_i]) < 55) _used.add(_slotOf[_j]);
        }
        let _s = 0; while (_used.has(_s)) _s++;
        _slotOf[_i]  = Math.min(_s, 8);
        _assigned[_i] = true;
    }
    // Pasada 2: líneas secundarias → se ubican en slots libres (más arriba)
    for (let _i = 0; _i < lineasOrdenadas.length; _i++) {
        if (_assigned[_i]) continue;
        const _used = new Set();
        for (let _j = 0; _j < lineasOrdenadas.length; _j++) {
            if (_j === _i || !_assigned[_j]) continue;
            if (Math.abs(_xPos[_j] - _xPos[_i]) < 55) _used.add(_slotOf[_j]);
        }
        let _s = 0; while (_used.has(_s)) _s++;
        _slotOf[_i]  = Math.min(_s, 8);
        _assigned[_i] = true;
    }

    for (let _idx = 0; _idx < lineasOrdenadas.length; _idx++) {
        const lin  = lineasOrdenadas[_idx];
        const x    = _xPos[_idx];
        if (x < X_LEFT - 8 || x > W + 8) continue;

        const xc       = x.toFixed(1);
        const slot     = _slotOf[_idx];
        const slotOff  = slot * 18;  // 18px por nivel
        const yName    = Y_TOP - 30 - slotOff;
        const yEW      = Y_TOP - 22 - slotOff;
        const yConn    = Y_TOP - 15 - slotOff;
        const nombreClave = lin.nombre.replace(/ /g, '_');
        const esKey    = lineasUsadasSet.has(nombreClave);
        const color    = colorLinea(lin.nombre);
        const isZoomed = _clasifZoom && Math.abs(_clasifZoom.lambda - lin.longitud_onda) < 5;
        const safeName = lin.nombre.replace(/'/g, "\\'");

        // Marco dorado para líneas clave
        if (esKey) {
            s += `<rect x="${(x-4).toFixed(1)}" y="${Y_TOP}" width="8"
                    height="${Y_BOT - Y_TOP}" fill="${color}" opacity="0.25" rx="1"/>`;
        }
        s += `<line x1="${xc}" y1="${Y_TOP}" x2="${xc}" y2="${Y_BOT}"
                stroke="${color}" stroke-width="${esKey ? '2' : '1.2'}"
                opacity="${esKey ? '0.95' : '0.60'}"
                stroke-dasharray="${esKey ? '10,5' : '5,6'}"/>`;

        // Etiqueta — clicable
        const ewLabel = `EW=${lin.ancho_equivalente.toFixed(2)}`;
        s += `<g onclick="clasifZomarLinea(${lin.longitud_onda}, '${safeName}')" style="cursor:pointer">`;
        if (esKey) {
            s += `<rect x="${(x-32).toFixed(1)}" y="${(yName-13).toFixed(1)}" width="64" height="22"
                    fill="${color}" opacity="0.15" rx="3" stroke="${color}" stroke-width="0.8"/>`;
        }
        s += `<text x="${xc}" y="${yName}" text-anchor="middle" font-size="9.5" fill="${color}"
                font-family="monospace" font-weight="${esKey ? 'bold' : 'normal'}"
                text-decoration="${isZoomed ? 'underline' : 'none'}">${lin.nombre}</text>`;
        s += `<text x="${xc}" y="${yEW}" text-anchor="middle" font-size="8" fill="#778"
                font-family="monospace">${ewLabel}</text>`;
        s += `</g>`;
        s += `<line x1="${xc}" y1="${yConn}" x2="${xc}" y2="${Y_TOP}"
                stroke="${color}" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.25"/>`;
    }

    // Indicador de zoom activo dentro del SVG
    if (_clasifZoom) {
        s += `<g onclick="clasifResetZoom()" style="cursor:pointer">
                <rect x="${X_LEFT+3}" y="3" width="130" height="18" rx="3"
                    fill="#1e3a5f" stroke="#3b82f6" stroke-width="1" opacity="0.90"/>
                <text x="${X_LEFT+68}" y="15" text-anchor="middle" font-size="10"
                    fill="#93c5fd" font-family="monospace">← Vista completa</text>
              </g>`;
    }

    // n. líneas
    const nDetect = todasLineas.filter(l => l.ancho_equivalente > 0.05).length;
    s += `<text x="${W-4}" y="14" text-anchor="end" font-size="9.5" fill="#334155"
            font-family="monospace" opacity="0.70">${nDetect} líneas detectadas</text>`;

    svg.innerHTML = s;
}

// ── Zoom — nivel 1 = ±80 Å, nivel 2 = ±25 Å ─────────────────────────────────
const _CLASIF_ZOOM_WINDOWS = [80, 25];

function clasifZomarLinea(lambda, label) {
    const ventana = _CLASIF_ZOOM_WINDOWS[0];
    _clasifZoom = { lambda, label, level: 1, lmin: lambda - ventana, lmax: lambda + ventana };
    _actualizarClasifZoomBar();
    if (_clasifSpecData) dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
}

function clasifSetZoomLevel(level) {
    if (!_clasifZoom) return;
    const ventana = _CLASIF_ZOOM_WINDOWS[level - 1];
    _clasifZoom.level = level;
    _clasifZoom.lmin  = _clasifZoom.lambda - ventana;
    _clasifZoom.lmax  = _clasifZoom.lambda + ventana;
    _actualizarClasifZoomBar();
    if (_clasifSpecData) dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
}

function clasifResetZoom() {
    _clasifZoom = null;
    const bar  = document.getElementById('clasifZoomBar');
    const hint = document.getElementById('clasifZoomHint');
    if (bar)  bar.style.display  = 'none';
    if (hint) hint.style.display = 'block';
    document.querySelectorAll('#claLv1,#claLv2').forEach(b => b.classList.remove('active'));
    if (_clasifSpecData) dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet);
}

function _actualizarClasifZoomBar() {
    const bar  = document.getElementById('clasifZoomBar');
    const lbl  = document.getElementById('clasifZoomLabel');
    const hint = document.getElementById('clasifZoomHint');
    if (bar)  bar.style.display  = 'flex';
    if (hint) hint.style.display = 'none';
    if (lbl && _clasifZoom) {
        lbl.textContent =
            `🔍 ${_clasifZoom.label} (${_clasifZoom.lambda} Å)  ·  ` +
            `${Math.round(_clasifZoom.lmin)}–${Math.round(_clasifZoom.lmax)} Å`;
    }
    // Resaltar botón del nivel activo
    const lv = _clasifZoom ? _clasifZoom.level : 0;
    const b1 = document.getElementById('claLv1');
    const b2 = document.getElementById('claLv2');
    if (b1) b1.classList.toggle('active', lv === 1);
    if (b2) b2.classList.toggle('active', lv === 2);
}

// ── Barras de intensidad (EW) para el clasificador ───────────────────────────
function renderClasifLinesBars(lineasDetectadas, lineasUsadasSet) {
    const container = document.getElementById('clasifLinesBars');
    if (!container) return;

    // Filtrar líneas detectadas (EW > 0.05), ordenar por EW desc
    const lista = (lineasDetectadas || [])
        .filter(l => l.ancho_equivalente > 0.05)
        .sort((a, b) => b.ancho_equivalente - a.ancho_equivalente);

    if (lista.length === 0) {
        container.innerHTML = '<p style="color:#64748b;font-size:0.85rem;text-align:center;">Sin líneas detectadas.</p>';
        return;
    }

    const maxEW = lista[0].ancho_equivalente || 1;

    container.innerHTML = lista.map(l => {
        const pct       = Math.min(100, Math.round((l.ancho_equivalente / maxEW) * 100));
        const nombreKey = l.nombre.replace(/ /g, '_');
        const esKey     = lineasUsadasSet.has(nombreKey);
        const color     = colorLinea(l.nombre);
        const safeName  = l.nombre.replace(/'/g, "\\'");
        return `<div class="clasif-bar-row${esKey ? ' bar-key' : ''}"
                     onclick="clasifZomarLinea(${l.longitud_onda}, '${safeName}'); document.getElementById('clasifSVG')?.scrollIntoView({behavior:'smooth',block:'center'})"
                     title="${l.nombre} — ${l.longitud_onda.toFixed(1)} Å — EW ${l.ancho_equivalente.toFixed(3)} Å">
            <div class="clasif-bar-name">
                ${esKey ? '<span class="bar-key-icon">🔑</span>' : ''}${l.nombre}
            </div>
            <div class="clasif-bar-track">
                <div class="clasif-bar-fill" style="width:${pct}%;background:${color};"></div>
            </div>
            <div class="clasif-bar-ew">${l.ancho_equivalente.toFixed(3)}</div>
        </div>`;
    }).join('');
}

// ── Mini espectro — solo líneas clave ────────────────────────────────────────

function dibujarMiniSpec(specData, todasLineas, lineasUsadasSet) {
    const svg = document.getElementById('miniSpecSVG');
    if (!svg || !specData) return;

    const wl   = specData.wavelength || [];
    const flux = specData.flux || [];
    if (wl.length < 2) { svg.innerHTML = ''; return; }

    const W = 860, H = 190;
    const PAD_L = 42, PAD_R = 16, PAD_T = 28, PAD_B = 28;
    const PW = W - PAD_L - PAD_R;
    const PH = H - PAD_T - PAD_B;

    const wMin = wl[0], wMax = wl[wl.length - 1];

    // Rango visible del flux
    let fMin = Infinity, fMax = -Infinity;
    for (let i = 0; i < flux.length; i++) {
        if (Number.isFinite(flux[i])) {
            if (flux[i] < fMin) fMin = flux[i];
            if (flux[i] > fMax) fMax = flux[i];
        }
    }
    if (!Number.isFinite(fMin) || fMax <= fMin) { fMin = 0; fMax = 1.5; }

    const tx = w => PAD_L + ((w - wMin) / (wMax - wMin)) * PW;
    const ty = f => PAD_T + (1 - (f - fMin) / (fMax - fMin)) * PH;

    // Polilínea — subsample para no generar SVG enorme
    const step = Math.max(1, Math.floor(wl.length / 1200));
    const pts = wl
        .filter((_, i) => i % step === 0 && Number.isFinite(flux[i]))
        .map((w, i) => `${tx(w).toFixed(1)},${ty(flux[i * step]).toFixed(1)}`)
        .join(' ');

    // Solo líneas usadas en la clasificación
    const keyLines = (todasLineas || []).filter(l => {
        const k = l.nombre.replace(/ /g, '_');
        return lineasUsadasSet.has(k) && l.longitud_onda >= wMin && l.longitud_onda <= wMax;
    });

    // Stagger etiquetas para evitar solapamiento
    keyLines.sort((a, b) => a.longitud_onda - b.longitud_onda);
    const linesSVG = keyLines.map((l, idx) => {
        const x   = tx(l.longitud_onda);
        const col = colorLinea(l.nombre);
        const yLbl = idx % 2 === 0 ? PAD_T - 6 : PAD_T - 16;
        return `<line x1="${x.toFixed(1)}" y1="${PAD_T}" x2="${x.toFixed(1)}" y2="${(PAD_T + PH).toFixed(1)}"
                      stroke="${col}" stroke-width="1.5" opacity="0.8" stroke-dasharray="4,3"/>
                <text x="${x.toFixed(1)}" y="${yLbl}" fill="${col}" font-size="9"
                      text-anchor="middle" font-family="monospace">${l.nombre}</text>`;
    }).join('');

    // Eje X
    const nTicks = 7;
    const ticksSVG = Array.from({length: nTicks}, (_, i) => {
        const w = wMin + (i / (nTicks - 1)) * (wMax - wMin);
        const x = tx(w);
        return `<line x1="${x.toFixed(1)}" y1="${(PAD_T + PH).toFixed(1)}"
                      x2="${x.toFixed(1)}" y2="${(PAD_T + PH + 4).toFixed(1)}"
                      stroke="#334155" stroke-width="1"/>
                <text x="${x.toFixed(1)}" y="${(PAD_T + PH + 15).toFixed(1)}"
                      fill="#64748b" font-size="9" text-anchor="middle">${Math.round(w)}</text>`;
    }).join('');

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = `
        <rect x="0" y="0" width="${W}" height="${H}" fill="#0a1628" rx="6"/>
        <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${PAD_T + PH}" stroke="#334155" stroke-width="1"/>
        <line x1="${PAD_L}" y1="${PAD_T + PH}" x2="${W - PAD_R}" y2="${PAD_T + PH}" stroke="#334155" stroke-width="1"/>
        <polyline points="${pts}" fill="none" stroke="#60a5fa" stroke-width="1.5" opacity="0.92"/>
        ${linesSVG}
        ${ticksSVG}
        <text x="${W / 2}" y="${H - 2}" fill="#475569" font-size="9"
              text-anchor="middle" font-family="monospace">λ (Å)</text>
    `;
}

// ── Exportar árbol de decisión ────────────────────────────────────────────────

function exportarArbol(formato) {
    const nombre = (window._preambuloFilename || 'espectro').replace(/\.[^.]+$/, '');
    const pasos = Array.from(document.querySelectorAll('#justificacionList li')).map(li => li.textContent.trim());
    const badges = Array.from(document.querySelectorAll('#lineasUsadasGrid .linea-usada-badge')).map(b => b.textContent.trim());
    const advs = Array.from(document.querySelectorAll('#advertenciasList li')).map(li => li.textContent.trim());
    const keyLines = (_clasifTodasLineas || []).filter(l => _clasifLineasUsadasSet.has(l.nombre.replace(/ /g, '_')));

    let contenido, mime, ext;
    const sep = '─'.repeat(60);

    if (formato === 'txt') {
        contenido = `SpectroClass v3.1 — Árbol de Decisión Espectroscópico\n`;
        contenido += `Archivo: ${nombre}\n${sep}\n\n`;
        contenido += `LÍNEAS DETERMINANTES\n${badges.join(', ')}\n\n`;
        contenido += `PASOS DEL ÁRBOL\n`;
        pasos.forEach((p, i) => { contenido += `${i + 1}. ${p}\n`; });
        if (advs.length) {
            contenido += `\nADVERTENCIAS\n`;
            advs.forEach(a => { contenido += `• ${a}\n`; });
        }
        mime = 'text/plain;charset=utf-8;'; ext = 'txt';
    } else {
        contenido = 'Linea,Lambda_A,EW_A,Profundidad\n';
        contenido += keyLines.map(l =>
            `"${l.nombre}",${l.longitud_onda.toFixed(3)},${l.ancho_equivalente.toFixed(4)},${l.profundidad.toFixed(4)}`
        ).join('\n');
        mime = 'text/csv;charset=utf-8;'; ext = 'csv';
    }

    const blob = new Blob([contenido], { type: mime });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `${nombre}_arbol.${ext}`; a.click();
    URL.revokeObjectURL(url);
}

function populatePreambulo(result) {
    document.getElementById('preambuloPlaceholder').style.display = 'none';
    document.getElementById('preambuloContent').style.display     = 'block';

    // ── 1. Información general ────────────────────────────────────────
    const diag0 = result.diagnostics || {};
    const _mk   = diag0.mk_full || '';
    const _lum  = diag0.luminosity_class || '';
    const _tipo = result.subtipo || result.tipo_clasificado || '';

    // Campos que provienen del archivo (se muestran en blanco si no existen)
    const fromFile = [
        { label: 'Archivo',        value: result.filename || '' },
        { label: 'Formato',        value: result.file_format || 'TXT' },
        { label: 'Objeto',         value: result.objeto || '' },        // '' → en blanco
        { label: 'Tipo original',  value: result.tipo_original || '' }, // '' → en blanco
        { label: 'Rango λ',        value: result.rango_lambda
            ? `${result.rango_lambda[0]} – ${result.rango_lambda[1]} Å` : '' },
        { label: 'Nº de puntos',   value: result.n_puntos
            ? result.n_puntos.toLocaleString() : '' },
        { label: 'Líneas detect.', value: result.n_lineas != null ? String(result.n_lineas) : '' },
        { label: 'Timestamp',      value: result.timestamp || '' },
    ];

    // Campos calculados por SpectroClass (siempre presentes)
    const calculado = [
        { label: 'Tipo espectral',    value: _tipo },
        { label: 'Clase luminosidad', value: _lum  },
        { label: 'Tipo MK completo',  value: _mk   },
    ];

    const grid = document.getElementById('preambuloInfoGrid');

    const renderFile = fromFile.map(({ label, value }) =>
        `<div class="preambulo-info-item">
            <span class="preambulo-info-label">${label}</span>
            <span class="preambulo-info-value">${value}</span>
         </div>`
    ).join('');

    const renderCalc = calculado.map(({ label, value }) =>
        `<div class="preambulo-info-item preambulo-info-calc">
            <span class="preambulo-info-label">
                ${label}
                <span class="calc-badge" title="Calculado por SpectroClass">⚙ calculado</span>
            </span>
            <span class="preambulo-info-value">${value}</span>
         </div>`
    ).join('');

    grid.innerHTML =
        `<div class="preambulo-info-section-title">📁 Del archivo</div>` +
        renderFile +
        `<div class="preambulo-info-section-title preambulo-info-section-calc">⚙️ Calculado por SpectroClass</div>` +
        renderCalc;

    // ── 2. Cabecera FITS ──────────────────────────────────────────────
    const fitsSection = document.getElementById('fitsSectionWrapper');
    const fitsBody    = document.getElementById('fitsHeaderBody');

    // Almacenar todas las filas para el filtro
    window._allFitsRows = [];

    if (result.fits_header && result.fits_header.length > 0) {
        fitsSection.style.display = 'block';
        window._allFitsRows = result.fits_header;
        fitsBody.innerHTML = result.fits_header.map(card =>
            `<tr>
                <td class="fits-key">${card.clave}</td>
                <td class="fits-val">${card.valor}</td>
                <td class="fits-com">${card.comentario}</td>
             </tr>`
        ).join('');
        const countEl = document.getElementById('fitsFilterCount');
        if (countEl) countEl.textContent = `${result.fits_header.length} tarjetas`;
        const fi = document.getElementById('fitsFilterInput');
        if (fi) fi.value = '';
    } else {
        fitsSection.style.display = 'none';
    }

    // Resumen de campos clave (fits_summary)
    const summaryGrid = document.getElementById('fitsSummaryGrid');
    if (summaryGrid) {
        if (result.fits_summary && result.fits_summary.length > 0) {
            const _grupLabels = {
                obs:  { label: '📡 Observación',  color: '#3b82f6' },
                spec: { label: '📊 Espectro',      color: '#10b981' },
                est:  { label: '⭐ Estelar',        color: '#f59e0b' },
                inst: { label: '🔬 Instrumento',   color: '#8b5cf6' },
                astr: { label: '🌐 Astrometría',   color: '#06b6d4' },
            };
            // Agrupar por grupo
            const groups = {};
            result.fits_summary.forEach(item => {
                if (!groups[item.grupo]) groups[item.grupo] = [];
                groups[item.grupo].push(item);
            });
            let html = '';
            for (const [gkey, items] of Object.entries(groups)) {
                const ginfo = _grupLabels[gkey] || { label: gkey, color: '#64748b' };
                html += `<div class="fits-summary-group">
                    <div class="fits-summary-group-title" style="color:${ginfo.color}">${ginfo.label}</div>
                    <div class="fits-summary-items">`;
                items.forEach(item => {
                    html += `<div class="fits-summary-item">
                        <span class="fits-summary-key">${item.label}</span>
                        <span class="fits-summary-val">${item.valor}</span>
                    </div>`;
                });
                html += `</div></div>`;
            }
            summaryGrid.innerHTML = html;
            summaryGrid.style.display = 'grid';
        } else {
            summaryGrid.style.display = 'none';
        }
    }

    // ── 3. Espectro normalizado SVG ───────────────────────────────────
    // Usar requestAnimationFrame para asegurar que el SVG es visible antes de dibujar
    if (result.spectrum_data) {
        const _specData   = result.spectrum_data;
        const _lineas     = result.todas_lineas || [];
        requestAnimationFrame(() => {
            try {
                dibujarPreambuloEspectro(_specData, _lineas);
            } catch(err) {
                console.error('dibujarPreambuloEspectro:', err);
            }
        });
    }

    // ── 4. Todas las líneas ───────────────────────────────────────────
    _todasLineasData = result.todas_lineas || result.lineas_detectadas || [];
    window._preambuloFilename = result.filename || 'espectro';
    document.getElementById('soloDetectadas').checked = false;
    document.getElementById('linesSortSelect').value  = 'ew';
    document.getElementById('linesFilterInput').value = '';
    filtrarLineas();
}

// ── Color por tipo de ion ─────────────────────────────────────────────────────
function colorLinea(nombre) {
    const n = nombre.toLowerCase();
    if (/^h (alpha|beta|gamma|delta|epsilon)$/.test(n)) return '#60a5fa'; // Balmer
    if (n.startsWith('na'))   return '#f87171'; // Na antes que N
    if (n.startsWith('n v') || n.startsWith('n iv') || n.startsWith('n iii') || n.startsWith('n ii') || n === 'n v') return '#e879f9';
    if (n.startsWith('he i '))  return '#a78bfa';
    if (n.startsWith('he ii')) return '#7c3aed';
    if (n.startsWith('ca ii')) return '#fb923c';
    if (n.startsWith('ca i')) return '#f97316';
    if (n.startsWith('fe ii')) return '#fbbf24';
    if (n.startsWith('fe i')) return '#d97706';
    if (n.startsWith('mg'))   return '#34d399';
    if (n.startsWith('tio'))  return '#e879f9';
    if (n.startsWith('ti'))   return '#c084fc';
    if (n.startsWith('si'))   return '#6ee7b7';
    if (n.startsWith('cr'))   return '#f472b6';
    if (n.startsWith('o ') || n.startsWith('o v') || n.startsWith('o ii') || n.startsWith('o iii')) return '#86efac';
    if (n.startsWith('c ') || n.startsWith('c ii') || n.startsWith('c iv')) return '#fde68a';
    if (n.startsWith('al'))   return '#94a3b8';
    return '#94a3b8';
}

// ── Dibujar espectro completo en la pestaña Preámbulo ────────────────────────
function dibujarPreambuloEspectro(specData, todasLineas) {
    const svg = document.getElementById('preambuloSVG');
    if (!svg) return;

    // Guardar en caché para redibujar al hacer zoom/reset
    _preambuloSpecData    = specData;
    _preambuloTodasLineas = todasLineas;

    const W      = 1400;
    const X_LEFT = 65;
    const Y_TOP  = 210;   // espacio para 9 niveles de etiquetas × 18px = 162px
    const Y_BOT  = 700;
    const Y_TICK = 710;
    const Y_TICK2= 728;
    const Y_LABEL= 750;

    // Rango λ — usar zoom si está activo, si no la vista completa
    let LMIN, LMAX;
    if (_preambuloZoom) {
        LMIN = _preambuloZoom.lmin;
        LMAX = _preambuloZoom.lmax;
    } else {
        const margin = (specData.wmax - specData.wmin) * 0.01;
        LMIN = specData.wmin - margin;
        LMAX = specData.wmax + margin;

        // Expandir izquierda para líneas diagnóstico < 6000 Å fuera de cobertura
        todasLineas.forEach(lin => {
            if (lin.longitud_onda < 6000 && lin.longitud_onda - 80 < LMIN) {
                LMIN = lin.longitud_onda - 80;
            }
        });
    }

    const lambdaToX = l => X_LEFT + ((l - LMIN) / (LMAX - LMIN)) * (W - X_LEFT);

    // Pre-calcular rango de flujo (filtrando NaN/Inf)
    const wArr = specData.wavelength;
    const fArr = specData.flux;
    let fMin = Infinity, fMax = -Infinity;
    const ptsVisible = [];
    for (let i = 0; i < wArr.length; i++) {
        if (wArr[i] >= LMIN && wArr[i] <= LMAX && Number.isFinite(fArr[i])) {
            if (fArr[i] < fMin) fMin = fArr[i];
            if (fArr[i] > fMax) fMax = fArr[i];
            ptsVisible.push([wArr[i], fArr[i]]);
        }
    }
    if (ptsVisible.length < 2 || !Number.isFinite(fMin) || !Number.isFinite(fMax) || fMax <= fMin) {
        fMin = 0; fMax = 1.5;
    }
    const scaleF = f => Y_TOP + (1 - (f - fMin) / (fMax - fMin)) * (Y_BOT - Y_TOP);

    let s = '';

    // Defs: gradiente + clipPath
    s += `<defs>
      <linearGradient id="pgSpecGrad" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%"   stop-color="#7b00ff"/>
        <stop offset="15%"  stop-color="#4466ff"/>
        <stop offset="30%"  stop-color="#44aaff"/>
        <stop offset="45%"  stop-color="#44ffaa"/>
        <stop offset="60%"  stop-color="#aaff44"/>
        <stop offset="75%"  stop-color="#ffdd00"/>
        <stop offset="88%"  stop-color="#ff6600"/>
        <stop offset="100%" stop-color="#cc0000"/>
      </linearGradient>
      <clipPath id="pgClip">
        <rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"/>
      </clipPath>
    </defs>`;

    // Franja cromática de fondo
    s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"
            fill="url(#pgSpecGrad)" opacity="0.10" rx="4"/>`;

    // Eje X base
    s += `<line x1="${X_LEFT}" y1="${Y_BOT}" x2="${W}" y2="${Y_BOT}" stroke="#333" stroke-width="1"/>`;

    // Eje Y
    s += `<line x1="${X_LEFT}" y1="${Y_TOP}" x2="${X_LEFT}" y2="${Y_BOT}" stroke="#555" stroke-width="1.5"/>`;

    const yRange = fMax - fMin;
    let yTickStep = yRange <= 0.3 ? 0.05 : yRange <= 0.6 ? 0.1 : yRange <= 1.2 ? 0.2 : yRange <= 3.0 ? 0.5 : 1.0;
    const yTickStart = Math.ceil(fMin / yTickStep) * yTickStep;
    const yTickCount = Math.ceil((fMax - yTickStart) / yTickStep) + 2;
    for (let ti = 0; ti < yTickCount; ti++) {
        const fVal = Math.round((yTickStart + ti * yTickStep) * 1e6) / 1e6;
        if (fVal > fMax + 1e-9) break;
        const y = scaleF(fVal);
        if (y < Y_TOP - 5 || y > Y_BOT + 5) continue;
        s += `<line x1="${X_LEFT}" y1="${y.toFixed(1)}" x2="${W}" y2="${y.toFixed(1)}"
                stroke="#2a3040" stroke-width="1" opacity="0.6"/>`;
        s += `<line x1="${(X_LEFT - 8).toFixed(1)}" y1="${y.toFixed(1)}"
                x2="${X_LEFT}" y2="${y.toFixed(1)}" stroke="#667" stroke-width="1.5"/>`;
        s += `<text x="${(X_LEFT - 10).toFixed(1)}" y="${(y + 5).toFixed(1)}"
                text-anchor="end" font-size="14" fill="#778" font-family="monospace">${fVal.toFixed(2)}</text>`;
    }
    const yMid = ((Y_TOP + Y_BOT) / 2).toFixed(1);
    s += `<text x="14" y="${yMid}" text-anchor="middle" font-size="14" fill="#556"
            font-family="monospace" transform="rotate(-90, 14, ${yMid})">Flujo</text>`;

    // Espectro real
    if (ptsVisible.length >= 2) {
        const step = Math.max(1, Math.floor(ptsVisible.length / 1200));
        const pts  = ptsVisible
            .filter((_, i) => i % step === 0)
            .map(([w, f]) => `${lambdaToX(w).toFixed(1)},${scaleF(f).toFixed(1)}`)
            .join(' ');
        const ptsFill = `${X_LEFT},${Y_BOT} ` + pts + ` ${W},${Y_BOT}`;
        s += `<polygon points="${ptsFill}" fill="#7ecfff" opacity="0.07" clip-path="url(#pgClip)"/>`;
        s += `<polyline points="${pts}" fill="none" stroke="#7ecfff"
                stroke-width="2.5" opacity="0.92" clip-path="url(#pgClip)"/>`;
        // Línea de continuo = 1
        const yContRef = scaleF(1.0);
        if (yContRef >= Y_TOP && yContRef <= Y_BOT) {
            s += `<line x1="${X_LEFT}" y1="${yContRef.toFixed(1)}" x2="${W}" y2="${yContRef.toFixed(1)}"
                    stroke="#ffffff" stroke-width="1" stroke-dasharray="8,12" opacity="0.28"/>`;
            s += `<text x="${X_LEFT + 8}" y="${(yContRef - 5).toFixed(1)}" font-size="15"
                    fill="#ffffff" opacity="0.40" font-family="monospace">cont=1</text>`;
        }
    }

    // Eje λ (ticks)
    const rango = LMAX - LMIN;
    const tickStep = rango > 1500 ? 500 : rango > 500 ? 200 : rango > 200 ? 50 : 20;
    const tickStart = Math.ceil(LMIN / tickStep) * tickStep;
    for (let l = tickStart; l <= LMAX; l += tickStep) {
        const x = lambdaToX(l);
        if (x < X_LEFT || x > W) continue;
        s += `<line x1="${x.toFixed(1)}" y1="${Y_TICK}" x2="${x.toFixed(1)}" y2="${Y_TICK2}"
                stroke="#444" stroke-width="2"/>`;
        s += `<text x="${x.toFixed(1)}" y="${Y_LABEL}" text-anchor="middle"
                font-size="18" fill="#778">${l}</text>`;
    }
    s += `<text x="${((X_LEFT + W) / 2).toFixed(1)}" y="${Y_LABEL + 22}" text-anchor="middle"
            font-size="16" fill="#556">Longitud de onda (Å)</text>`;

    // Región "sin cobertura" si la vista se expandió
    if (specData.wmin > LMIN) {
        const xCovStart = lambdaToX(specData.wmin);
        const xCovWidth = (xCovStart - X_LEFT).toFixed(1);
        const yMidPlot  = ((Y_TOP + Y_BOT) / 2).toFixed(1);
        s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${xCovWidth}"
                height="${Y_BOT - Y_TOP}" fill="#0d1117" opacity="0.55"/>`;
        s += `<text x="${((X_LEFT + xCovStart) / 2).toFixed(1)}" y="${(+yMidPlot - 12).toFixed(1)}"
                text-anchor="middle" font-size="14" fill="#475569"
                font-family="monospace" opacity="0.80">sin cobertura</text>`;
        s += `<text x="${((X_LEFT + xCovStart) / 2).toFixed(1)}" y="${(+yMidPlot + 10).toFixed(1)}"
                text-anchor="middle" font-size="12" fill="#334155"
                font-family="monospace" opacity="0.70">λ &lt; ${specData.wmin.toFixed(0)} Å</text>`;
        s += `<line x1="${xCovStart.toFixed(1)}" y1="${Y_TOP}"
                x2="${xCovStart.toFixed(1)}" y2="${Y_BOT}"
                stroke="#475569" stroke-width="2" stroke-dasharray="6,4" opacity="0.75"/>`;
    }

    // Líneas espectrales — escalonamiento anti-solapamiento
    // Filtrar solo líneas detectadas para la visualización (EW > 0.05), pero mostrar las no detectadas con opacidad baja
    const lineasOrdenadas = [...todasLineas].sort((a, b) => a.longitud_onda - b.longitud_onda);

    const _xPos    = lineasOrdenadas.map(lin => lambdaToX(lin.longitud_onda));
    const _slotOf  = new Array(lineasOrdenadas.length).fill(0);
    const _assigned= new Array(lineasOrdenadas.length).fill(false);

    // Pasada 1: líneas detectadas primero → slots bajos (más cerca del espectro)
    for (let _i = 0; _i < lineasOrdenadas.length; _i++) {
        if (lineasOrdenadas[_i].ancho_equivalente <= 0.05) continue;
        const _used = new Set();
        for (let _j = 0; _j < lineasOrdenadas.length; _j++) {
            if (_j === _i || !_assigned[_j]) continue;
            if (Math.abs(_xPos[_j] - _xPos[_i]) < 75) _used.add(_slotOf[_j]);
        }
        let _s = 0; while (_used.has(_s)) _s++;
        _slotOf[_i]   = Math.min(_s, 8);
        _assigned[_i] = true;
    }
    // Pasada 2: líneas no detectadas → slots libres más arriba
    for (let _i = 0; _i < lineasOrdenadas.length; _i++) {
        if (_assigned[_i]) continue;
        const _used = new Set();
        for (let _j = 0; _j < lineasOrdenadas.length; _j++) {
            if (_j === _i || !_assigned[_j]) continue;
            if (Math.abs(_xPos[_j] - _xPos[_i]) < 75) _used.add(_slotOf[_j]);
        }
        let _s = 0; while (_used.has(_s)) _s++;
        _slotOf[_i]   = Math.min(_s, 8);
        _assigned[_i] = true;
    }

    for (let _idx = 0; _idx < lineasOrdenadas.length; _idx++) {
        const lin     = lineasOrdenadas[_idx];
        const x       = _xPos[_idx];
        if (x < X_LEFT - 10 || x > W + 10) continue;

        const xc      = x.toFixed(1);
        const slot    = _slotOf[_idx];
        const slotOff = slot * 18;
        const yName   = Y_TOP - 38 - slotOff;
        const yEW     = Y_TOP - 30 - slotOff;
        const yConn   = Y_TOP - 22 - slotOff;

        const detectada = lin.ancho_equivalente > 0.05;
        const inRange   = lin.longitud_onda >= specData.wmin && lin.longitud_onda <= specData.wmax;
        const color     = inRange ? colorLinea(lin.nombre) : '#64748b';
        const opacity   = detectada && inRange ? '0.92' : (inRange ? '0.35' : '0.30');
        const fillOp    = detectada && inRange ? '0.20' : '0.06';
        const dashArray = inRange ? (detectada ? '10,6' : '4,6') : '3,5';

        // Solo dibujar líneas con EW > 0 o que estén en rango (para no saturar con líneas ausentes)
        // Mostrar todas las detectadas y las fuera de rango notables
        if (!detectada && inRange) {
            // No detectada en rango: omitir para no saturar
            continue;
        }

        s += `<rect x="${(x - 3).toFixed(1)}" y="${Y_TOP}" width="6"
                height="${Y_BOT - Y_TOP}" fill="${color}" opacity="${fillOp}" rx="1"/>`;
        s += `<line x1="${xc}" y1="${Y_TOP}" x2="${xc}" y2="${Y_BOT}"
                stroke="${color}" stroke-width="${detectada ? '1.5' : '1'}"
                opacity="${opacity}" stroke-dasharray="${dashArray}"/>`;

        // Marca de profundidad en el espectro real
        if (detectada && inRange && lin.profundidad > 0.02) {
            const fluxMin = Math.max(fMin, 1.0 - lin.profundidad);
            const yFlux   = scaleF(fluxMin);
            if (yFlux >= Y_TOP && yFlux <= Y_BOT) {
                s += `<line x1="${(x-22).toFixed(1)}" y1="${yFlux.toFixed(1)}"
                        x2="${(x+22).toFixed(1)}" y2="${yFlux.toFixed(1)}"
                        stroke="${color}" stroke-width="3" opacity="0.88"/>`;
                s += `<line x1="${(x-22).toFixed(1)}" y1="${(yFlux-7).toFixed(1)}"
                        x2="${(x-22).toFixed(1)}" y2="${(yFlux+7).toFixed(1)}"
                        stroke="${color}" stroke-width="1.8" opacity="0.70"/>`;
                s += `<line x1="${(x+22).toFixed(1)}" y1="${(yFlux-7).toFixed(1)}"
                        x2="${(x+22).toFixed(1)}" y2="${(yFlux+7).toFixed(1)}"
                        stroke="${color}" stroke-width="1.8" opacity="0.70"/>`;
            }
        }

        // Etiquetas — clicables para hacer zoom
        const label    = lin.nombre;
        const ewLabel  = detectada ? `EW=${lin.ancho_equivalente.toFixed(2)} Å` : 'fuera de rango';
        const ewColor  = detectada ? '#e2e8f0' : '#475569';
        const safeName = label.replace(/'/g, "\\'");
        const isZoomed = _preambuloZoom && Math.abs(_preambuloZoom.lambda - lin.longitud_onda) < 5;

        // Zona clicable alrededor de la etiqueta (área amplia para facilitar el clic)
        s += `<g onclick="preambuloZomarLinea(${lin.longitud_onda}, '${safeName}')"
                style="cursor:pointer" title="Ampliar ${label}">`;
        s += `<rect x="${(x - 38).toFixed(1)}" y="${(yName - 14).toFixed(1)}" width="76" height="32"
                fill="${isZoomed ? color : 'transparent'}" opacity="${isZoomed ? '0.18' : '1'}" rx="3"/>`;
        s += `<text x="${xc}" y="${yName}" text-anchor="middle"
                font-size="12" fill="${color}" font-family="monospace"
                font-weight="bold"
                opacity="${detectada ? '1.0' : '0.55'}"
                text-decoration="${isZoomed ? 'underline' : 'none'}">${label}</text>`;
        s += `<text x="${xc}" y="${yEW}" text-anchor="middle"
                font-size="10" fill="${ewColor}" font-family="monospace"
                opacity="0.80">${ewLabel}</text>`;
        s += `</g>`;
        s += `<line x1="${xc}" y1="${yConn}" x2="${xc}" y2="${Y_TOP}"
                stroke="${color}" stroke-width="1" stroke-dasharray="3,3" opacity="0.30"/>`;
    }

    // Indicador de n. líneas detectadas + hint de zoom (esquina superior derecha)
    const nDetect = todasLineas.filter(l => l.ancho_equivalente > 0.05).length;
    s += `<text x="${W - 8}" y="14" text-anchor="end" font-size="11" fill="#334155"
            font-family="monospace" opacity="0.75">${nDetect} líneas detectadas</text>`;

    // Si hay zoom activo: botón de reset dentro del SVG (esquina superior izquierda)
    if (_preambuloZoom) {
        s += `<g onclick="preambuloResetZoom()" style="cursor:pointer" title="Volver a vista completa">
                <rect x="${X_LEFT + 4}" y="4" width="148" height="22" rx="4"
                    fill="#1e3a5f" stroke="#3b82f6" stroke-width="1" opacity="0.90"/>
                <text x="${X_LEFT + 78}" y="18" text-anchor="middle" font-size="12"
                    fill="#93c5fd" font-family="monospace">← Vista completa</text>
              </g>`;
    }

    svg.innerHTML = s;
}

// ── Filtro para la tabla completa de cabecera FITS ────────────────────────
function filtrarFitsHeader() {
    const texto = (document.getElementById('fitsFilterInput')?.value || '').toLowerCase().trim();
    const body  = document.getElementById('fitsHeaderBody');
    const countEl = document.getElementById('fitsFilterCount');
    if (!body) return;
    const rows = window._allFitsRows || [];
    const filtered = texto
        ? rows.filter(r =>
            r.clave.toLowerCase().includes(texto) ||
            r.valor.toLowerCase().includes(texto) ||
            r.comentario.toLowerCase().includes(texto))
        : rows;
    body.innerHTML = filtered.map(card =>
        `<tr>
            <td class="fits-key">${card.clave}</td>
            <td class="fits-val">${card.valor}</td>
            <td class="fits-com">${card.comentario}</td>
         </tr>`
    ).join('');
    if (countEl) countEl.textContent = texto
        ? `${filtered.length} / ${rows.length} tarjetas`
        : `${rows.length} tarjetas`;
}

function filtrarLineas() {
    const texto   = (document.getElementById('linesFilterInput')?.value || '').toLowerCase();
    const orden   = document.getElementById('linesSortSelect')?.value || 'ew';
    const soloD   = document.getElementById('soloDetectadas')?.checked;

    let lista = _todasLineasData.filter(l => {
        if (soloD && l.ancho_equivalente <= 0.05) return false;
        if (texto) {
            const hayMatch = l.nombre.toLowerCase().includes(texto) ||
                             String(l.longitud_onda).includes(texto);
            if (!hayMatch) return false;
        }
        return true;
    });

    if (orden === 'ew')     lista.sort((a, b) => b.ancho_equivalente - a.ancho_equivalente);
    if (orden === 'lambda') lista.sort((a, b) => a.longitud_onda - b.longitud_onda);
    if (orden === 'depth')  lista.sort((a, b) => b.profundidad - a.profundidad);
    if (orden === 'nombre') lista.sort((a, b) => a.nombre.localeCompare(b.nombre));

    const MAX_BAR = lista.length > 0 ? Math.max(...lista.map(l => l.ancho_equivalente)) : 1;

    const cuerpo = document.getElementById('allLinesBody');
    if (!cuerpo) return;

    cuerpo.innerHTML = lista.map(l => {
        const pct     = Math.round((l.ancho_equivalente / MAX_BAR) * 100);
        const detect  = l.ancho_equivalente > 0.05;
        const rowCls  = detect ? 'line-row-detected' : 'line-row-faint';
        const barColor= detect ? '#3b82f6' : '#334155';
        const intensLabel = l.ancho_equivalente > 3 ? 'Muy fuerte'
                          : l.ancho_equivalente > 1 ? 'Fuerte'
                          : l.ancho_equivalente > 0.3 ? 'Moderada'
                          : l.ancho_equivalente > 0.05 ? 'Débil'
                          : 'No detectada';
        const safeName = l.nombre.replace(/'/g, "\\'");
        const isActive = _preambuloZoom && Math.abs(_preambuloZoom.lambda - l.longitud_onda) < 5;
        return `<tr class="${rowCls}${isActive ? ' ll-row-zoomed' : ''}">
            <td class="ll-nombre">${l.nombre}</td>
            <td class="ll-lambda">${l.longitud_onda.toFixed(1)}</td>
            <td class="ll-ew">${l.ancho_equivalente.toFixed(3)}</td>
            <td class="ll-depth">${l.profundidad.toFixed(3)}</td>
            <td class="ll-bar">
                <div class="ew-bar-wrap">
                    <div class="ew-bar-fill" style="width:${pct}%;background:${barColor}"></div>
                    <span class="ew-bar-label">${intensLabel}</span>
                </div>
            </td>
            <td class="ll-zoom">
                <button class="btn-ll-zoom${isActive ? ' active' : ''}"
                    onclick="${isActive
                        ? 'preambuloResetZoom()'
                        : `preambuloZomarLinea(${l.longitud_onda}, '${safeName}'); document.getElementById('preambuloSVG')?.scrollIntoView({behavior:'smooth', block:'center'})`}"
                    title="${isActive ? 'Volver a vista completa del espectro' : `Ver ${l.nombre} (${l.longitud_onda.toFixed(1)} Å) en el espectro de arriba`}">
                    ${isActive ? '✕' : '🔍'}
                </button>
            </td>
        </tr>`;
    }).join('');

    document.getElementById('linesCountLabel').textContent =
        `Mostrando ${lista.length} de ${_todasLineasData.length} líneas`;
}

// ── Exportar tabla de líneas ─────────────────────────────────────────────────

function exportarLineas(formato) {
    const datos = _todasLineasData || [];
    if (!datos.length) { alert('No hay datos para exportar.'); return; }

    const nombre = (window._preambuloFilename || 'lineas_espectrales').replace(/\.[^.]+$/, '');
    let contenido, mime, ext;

    if (formato === 'csv') {
        const cabecera = 'Linea,Lambda_A,EW_A,Profundidad,Intensidad\n';
        const filas = datos.map(l => {
            const intens = l.ancho_equivalente > 3 ? 'Muy fuerte'
                         : l.ancho_equivalente > 1 ? 'Fuerte'
                         : l.ancho_equivalente > 0.3 ? 'Moderada'
                         : l.ancho_equivalente > 0.05 ? 'Debil'
                         : 'No detectada';
            return `"${l.nombre}",${l.longitud_onda.toFixed(3)},${l.ancho_equivalente.toFixed(4)},${l.profundidad.toFixed(4)},"${intens}"`;
        }).join('\n');
        contenido = cabecera + filas;
        mime = 'text/csv;charset=utf-8;';
        ext  = 'csv';
    } else {
        const sep = '─'.repeat(60);
        const cab = `${'Línea'.padEnd(20)} ${'λ (Å)'.padStart(9)} ${'EW (Å)'.padStart(9)} ${'Prof.'.padStart(8)}  Intensidad`;
        const filas = datos.map(l => {
            const intens = l.ancho_equivalente > 3 ? 'Muy fuerte'
                         : l.ancho_equivalente > 1 ? 'Fuerte'
                         : l.ancho_equivalente > 0.3 ? 'Moderada'
                         : l.ancho_equivalente > 0.05 ? 'Débil'
                         : 'No detectada';
            return `${l.nombre.padEnd(20)} ${l.longitud_onda.toFixed(2).padStart(9)} ${l.ancho_equivalente.toFixed(4).padStart(9)} ${l.profundidad.toFixed(4).padStart(8)}  ${intens}`;
        }).join('\n');
        contenido = `SpectroClass v3.1 — Líneas Espectrales\nArchivo: ${nombre}\n${sep}\n${cab}\n${sep}\n${filas}\n${sep}\n`;
        mime = 'text/plain;charset=utf-8;';
        ext  = 'txt';
    }

    const blob = new Blob([contenido], { type: mime });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `${nombre}_lineas.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
}

// ── Zoom en el espectro del Análisis Detallado ───────────────────────────────

const _PREAMBULO_ZOOM_WINDOWS = [80, 25];

function preambuloZomarLinea(lambda, label) {
    const ventana = _PREAMBULO_ZOOM_WINDOWS[0];
    _preambuloZoom = { lmin: lambda - ventana, lmax: lambda + ventana, lambda, label, level: 1 };
    _actualizarPreambuloZoomBar();
    if (_preambuloSpecData) dibujarPreambuloEspectro(_preambuloSpecData, _preambuloTodasLineas);
    filtrarLineas();
}

function preambuloSetZoomLevel(level) {
    if (!_preambuloZoom) return;
    const ventana = _PREAMBULO_ZOOM_WINDOWS[level - 1];
    _preambuloZoom.level = level;
    _preambuloZoom.lmin  = _preambuloZoom.lambda - ventana;
    _preambuloZoom.lmax  = _preambuloZoom.lambda + ventana;
    _actualizarPreambuloZoomBar();
    if (_preambuloSpecData) dibujarPreambuloEspectro(_preambuloSpecData, _preambuloTodasLineas);
}

function preambuloResetZoom() {
    _preambuloZoom = null;
    const bar  = document.getElementById('preambuloZoomBar');
    const hint = document.getElementById('preambuloZoomHint');
    if (bar)  bar.style.display  = 'none';
    if (hint) hint.style.display = 'block';
    document.querySelectorAll('#preLv1,#preLv2').forEach(b => b.classList.remove('active'));
    if (_preambuloSpecData) dibujarPreambuloEspectro(_preambuloSpecData, _preambuloTodasLineas);
    filtrarLineas();
}

function _actualizarPreambuloZoomBar() {
    const bar  = document.getElementById('preambuloZoomBar');
    const lbl  = document.getElementById('preambuloZoomLabel');
    const hint = document.getElementById('preambuloZoomHint');
    if (bar)  bar.style.display  = 'flex';
    if (hint) hint.style.display = 'none';
    if (lbl && _preambuloZoom) {
        lbl.textContent = `🔍 ${_preambuloZoom.label} (${_preambuloZoom.lambda} Å)  ·  ${Math.round(_preambuloZoom.lmin)}–${Math.round(_preambuloZoom.lmax)} Å`;
    }
    const lv = _preambuloZoom ? _preambuloZoom.level : 0;
    const b1 = document.getElementById('preLv1');
    const b2 = document.getElementById('preLv2');
    if (b1) b1.classList.toggle('active', lv === 1);
    if (b2) b2.classList.toggle('active', lv === 2);
}

// ── Carga de espectro desde la pestaña Análisis Detallado ────────────────────

function initPreambuloUpload() {
    const input    = document.getElementById('fileInputPreambulo');
    const dropZone = document.getElementById('preambuloDropZone');
    if (!input || !dropZone) return;

    // Click en el input → analizar
    input.addEventListener('change', e => {
        if (e.target.files[0]) analizarDesdePreambulo(e.target.files[0]);
        e.target.value = '';
    });

    // Drag & drop sobre la barra completa
    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', e => {
        if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) analizarDesdePreambulo(file);
    });
}

function analizarDesdePreambulo(file) {
    const processing = document.getElementById('preambuloProcessing');
    const errEl      = document.getElementById('preambuloError');
    const label      = document.getElementById('preambuloFileLabel');

    // Estado: cargando
    processing.style.display = 'flex';
    errEl.style.display      = 'none';
    if (label) label.textContent = file.name;

    const formData = new FormData();
    formData.append('files[]', file);

    fetch('/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            processing.style.display = 'none';

            if (data.error) {
                errEl.textContent   = '❌ ' + data.error;
                errEl.style.display = 'block';
                if (label) label.textContent = 'Arrastra un espectro aquí o';
                return;
            }

            const results = data.results || [data];
            if (!results.length) return;

            const result = results[0];

            // Actualizar etiqueta con nombre del archivo cargado
            if (label) label.textContent = '✅ ' + (result.filename || file.name);

            // Poblar la pestaña Análisis Detallado
            populatePreambulo(result);

            // Poblar la pestaña Clasificador (tolerando errores)
            try { displaySingleResult(result); } catch(e) { console.error('displaySingleResult:', e); }

            // Mantener foco en Análisis Detallado
            document.querySelector('[data-tab="preambulo"]')?.click();
        })
        .catch(err => {
            processing.style.display = 'none';
            errEl.textContent   = '❌ Error de conexión: ' + err.message;
            errEl.style.display = 'block';
            if (label) label.textContent = 'Arrastra un espectro aquí o';
        });
}

function setupToggleButtons() {
    const toggles = [
        { btn: 'toggleAlternatives', body: 'alternativesBody',  label: 'Ver detalles' },
        { btn: 'toggleEspectro',     body: 'espectroBody',       label: 'Ver espectro' },
        { btn: 'toggleLines',        body: 'linesBody',          label: 'Ver líneas'   },
        { btn: 'toggleLinesBars',    body: 'linesBarsBody',      label: 'Ver barras'   },
        { btn: 'toggleDecisionTree', body: 'decisionTreeBody',   label: 'Ver árbol'    },
        { btn: 'toggleExport',       body: 'exportBody',         label: 'Ver opciones' },
        { btn: 'toggleMethods',      body: 'methodsBody',        label: 'Ver detalles' }
    ];

    toggles.forEach(({ btn, body, label }) => {
        const btnEl  = document.getElementById(btn);
        const bodyEl = document.getElementById(body);
        if (!btnEl || !bodyEl) return;

        btnEl.addEventListener('click', () => {
            const isHidden = bodyEl.style.display === 'none';
            bodyEl.style.display = isHidden ? 'block' : 'none';
            btnEl.textContent = isHidden ? `▲ Ocultar` : `▼ ${label}`;

            // Redibujar espectro principal al abrirlo
            if (btn === 'toggleEspectro' && isHidden && _clasifSpecData) {
                requestAnimationFrame(() =>
                    dibujarClasifEspectro(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet)
                );
            }
            // Redibujar mini espectro al abrir el árbol de decisión
            if (btn === 'toggleDecisionTree' && isHidden && _clasifSpecData) {
                requestAnimationFrame(() =>
                    dibujarMiniSpec(_clasifSpecData, _clasifTodasLineas, _clasifLineasUsadasSet)
                );
            }
        });
    });
}

// ===========================
// BATCH PROCESSING
// ===========================

function setupBatchProcessing() {
    const dropZone = document.getElementById('dropZoneBatch');
    const fileInput = document.getElementById('fileInputBatch');
    const batchActions = document.getElementById('batchActions');
    const processBatchBtn = document.getElementById('processBatchBtn');
    const clearBatchBtn = document.getElementById('clearBatchBtn');
    const exportBatchCSV = document.getElementById('exportBatchCSV');
    const btnSelectBatch = document.getElementById('btnSelectBatch');

    // Botón "Seleccionar Archivos" - detener propagación
    btnSelectBatch.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // Click en zona de arrastrar (pero no en el botón)
    dropZone.addEventListener('click', (e) => {
        if (e.target === btnSelectBatch || e.target.closest('.btn-upload')) return;
        fileInput.click();
    });

    // Drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleBatchFiles(e.dataTransfer.files);
    });

    // File input
    fileInput.addEventListener('change', (e) => {
        handleBatchFiles(e.target.files);
        e.target.value = '';  // Resetear para permitir seleccionar los mismos archivos
    });

    // Process button
    processBatchBtn.addEventListener('click', processBatch);

    // Clear button
    clearBatchBtn.addEventListener('click', () => {
        batchFiles = [];
        updateBatchFilesList();
    });

    // Export button
    exportBatchCSV.addEventListener('click', exportBatchToCSV);
}

function handleBatchFiles(files) {
    const validExts = ['txt', 'fits', 'fit'];

    for (let file of files) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (validExts.includes(ext)) {
            if (!batchFiles.find(f => f.name === file.name)) {
                batchFiles.push(file);
            }
        } else {
            alert(`Formato no permitido: ${file.name}`);
        }
    }

    updateBatchFilesList();
}

function updateBatchFilesList() {
    const filesList = document.getElementById('batchFilesList');
    const batchActions = document.getElementById('batchActions');

    if (batchFiles.length === 0) {
        filesList.innerHTML = '';
        batchActions.style.display = 'none';
        return;
    }

    batchActions.style.display = 'flex';

    filesList.innerHTML = `
        <h3>${batchFiles.length} archivo(s) seleccionado(s)</h3>
        ${batchFiles.map((file, idx) => `
            <div class="batch-file-item">
                <span>📄 ${file.name}</span>
                <small style="margin-left: auto;">${(file.size / 1024).toFixed(2)} KB</small>
                <button onclick="removeBatchFile(${idx})" style="margin-left: 1rem;">✕</button>
            </div>
        `).join('')}
    `;
}

function removeBatchFile(index) {
    batchFiles.splice(index, 1);
    updateBatchFilesList();
}

async function processBatch() {
    if (batchFiles.length === 0) return;

    const batchProgress     = document.getElementById('batchProgress');
    const batchProgressFill = document.getElementById('batchProgressFill');
    const batchProgressText = document.getElementById('batchProgressText');
    const batchProgressTitle= document.getElementById('batchProgressTitle');
    const batchLog          = document.getElementById('batchLog');
    const batchResultsDiv   = document.getElementById('batchResults');

    batchProgress.style.display = 'block';
    batchResultsDiv.style.display = 'none';
    batchLog.innerHTML = '';

    const total      = batchFiles.length;
    const allResults = [];
    const allErrors  = [];

    // Misma función de coloreado que usa el entrenamiento
    function lineClass(msg) {
        if (msg.startsWith('[OK]') || msg.includes('✓'))               return 'tl-ok';
        if (msg.startsWith('[ERROR]') || msg.includes('✗'))            return 'tl-error';
        if (msg.startsWith('[!]') || msg.includes('⚠'))                return 'tl-warn';
        if (msg.startsWith('===') || msg.startsWith('PROCESAMIENTO') ||
            msg.startsWith('PASO') || msg.startsWith('──'))            return 'tl-head';
        if (msg.includes('Clasificado:') || msg.includes('líneas'))    return 'tl-pct';
        return '';
    }

    function appendLog(msg) {
        const el = document.createElement('div');
        el.className = 'tl-line ' + lineClass(msg);
        el.textContent = msg;
        batchLog.appendChild(el);
        batchLog.scrollTop = batchLog.scrollHeight;
    }

    appendLog('='.repeat(60));
    appendLog('PROCESAMIENTO POR LOTES - CLASIFICACIÓN ESPECTRAL');
    appendLog(`Total de archivos: ${total}`);
    appendLog('='.repeat(60));

    for (let i = 0; i < total; i++) {
        const file = batchFiles[i];
        const pct  = Math.round((i / total) * 100);

        batchProgressFill.style.width = pct + '%';
        batchProgressText.textContent = `${i} / ${total} procesados`;
        batchProgressTitle.textContent = `Procesando espectros... [${i + 1}/${total}]`;

        appendLog('');
        appendLog(`── [${i + 1}/${total}]: ${file.name}`);
        appendLog(`PASO 1: Cargando espectro...`);

        const formData = new FormData();
        formData.append('files[]', file);
        formData.append('physical_weight', globalWeights.physical.toString());
        formData.append('dt_weight',       globalWeights.decision_tree.toString());
        formData.append('template_weight', globalWeights.template.toString());
        formData.append('knn_weight',      globalWeights.knn.toString());
        formData.append('cnn_1d_weight',   globalWeights.cnn_1d.toString());
        formData.append('cnn_2d_weight',   globalWeights.cnn_2d.toString());

        try {
            appendLog(`PASO 2: Normalizando al continuo (sigma-clipping)...`);
            appendLog(`PASO 3: Midiendo anchos equivalentes (EW)...`);
            appendLog(`PASO 4: Clasificando espectro (pesos: Fís=${globalWeights.physical.toFixed(2)} Árbol=${globalWeights.decision_tree.toFixed(2)} Tmpl=${globalWeights.template.toFixed(2)} KNN=${globalWeights.knn.toFixed(2)} CNN1D=${globalWeights.cnn_1d.toFixed(2)} CNN2D=${globalWeights.cnn_2d.toFixed(2)})...`);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();

            if (data.results && data.results.length > 0) {
                const r    = data.results[0];
                const tipo = `${r.tipo_clasificado} ${r.subtipo || ''}`.trim();
                const conf = r.confianza ? ` | Confianza: ${r.confianza.toFixed(1)}%` : '';
                const lin  = r.n_lineas  ? ` | Líneas: ${r.n_lineas}` : '';
                appendLog(`[OK] Clasificado: ${tipo}${conf}${lin}`);
                allResults.push(r);
            } else if (data.errors && data.errors.length > 0) {
                appendLog(`[ERROR] ${data.errors[0].error}`);
                allErrors.push({ file: file.name, error: data.errors[0].error });
            }

        } catch (err) {
            appendLog(`[ERROR] ${err.message}`);
            allErrors.push({ file: file.name, error: err.message });
        }
    }

    // Completado
    batchProgressFill.style.width = '100%';
    batchProgressText.textContent = `${total} / ${total} procesados`;
    batchProgressTitle.textContent = 'Procesamiento completado';

    appendLog('');
    appendLog('='.repeat(60));
    appendLog(`PROCESAMIENTO COMPLETADO`);
    appendLog(`[OK] Exitosos : ${allResults.length}`);
    if (allErrors.length > 0) appendLog(`[!]  Errores  : ${allErrors.length}`);
    appendLog('='.repeat(60));

    batchResults = {
        results:   allResults,
        errors:    allErrors,
        n_success: allResults.length,
        n_errors:  allErrors.length
    };

    setTimeout(() => {
        batchProgress.style.display = 'none';
        displayBatchResults(batchResults);
    }, 1000);
}

function displayBatchResults(data) {
    const batchResultsDiv = document.getElementById('batchResults');
    const successCount = document.getElementById('successCount');
    const errorCount = document.getElementById('errorCount');
    const resultsGrid = document.getElementById('batchResultsGrid');

    batchResultsDiv.style.display = 'block';
    batchResultsDiv.scrollIntoView({ behavior: 'smooth' });

    successCount.textContent = data.n_success;
    errorCount.textContent = data.n_errors;

    // Results grid
    resultsGrid.innerHTML = '';

    if (data.results && data.results.length > 0) {
        data.results.forEach(result => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <div class="result-header">
                    <h4>${result.filename}</h4>
                </div>
                <div class="result-body">
                    <p><strong>Tipo:</strong> ${result.tipo_clasificado} (${result.subtipo})</p>
                    ${result.confianza ? `<p><strong>Confianza:</strong> ${result.confianza.toFixed(1)}%</p>` : ''}
                    <p><strong>Líneas:</strong> ${result.n_lineas}</p>
                </div>
            `;
            resultsGrid.appendChild(resultCard);
        });
    }

    if (data.errors && data.errors.length > 0) {
        data.errors.forEach(error => {
            const errorCard = document.createElement('div');
            errorCard.className = 'result-card';
            errorCard.style.borderLeft = '4px solid var(--danger)';
            errorCard.innerHTML = `
                <div class="result-header" style="background: var(--danger);">
                    <h4>❌ ${error.file}</h4>
                </div>
                <div class="result-body">
                    <p style="color: var(--danger);">${error.error}</p>
                </div>
            `;
            resultsGrid.appendChild(errorCard);
        });
    }
}

function exportBatchToCSV() {
    if (!batchResults) return;

    fetch('/export_csv', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(batchResults)
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `clasificacion_${new Date().getTime()}.csv`;
        a.click();
    })
    .catch(error => {
        alert(`Error exportando: ${error.message}`);
    });
}

// ===========================
// SCRIPT EXECUTION (TOOLS TAB)
// ===========================

async function runScript(scriptName) {
    const outputArea = document.getElementById('scriptOutputArea');
    const scriptNameEl = document.getElementById('scriptName');
    const scriptStatus = document.getElementById('scriptStatus');
    const scriptOutput = document.getElementById('scriptOutput');

    // Show output area
    outputArea.style.display = 'block';
    outputArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Update UI
    scriptNameEl.textContent = scriptName + '.bat';
    scriptStatus.textContent = 'Ejecutando...';
    scriptStatus.className = 'status-badge running';
    scriptOutput.textContent = 'Ejecutando script, por favor espera...\n\nEsto puede tardar varios minutos para operaciones como entrenar modelos.';

    // Disable the button that was clicked
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Ejecutando...';

    try {
        const response = await fetch('/run_script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ script: scriptName })
        });

        const data = await response.json();

        if (data.success) {
            scriptStatus.textContent = '✅ Completado';
            scriptStatus.className = 'status-badge success';
            scriptOutput.textContent = data.output || 'Script ejecutado correctamente (sin salida).';
        } else {
            scriptStatus.textContent = '❌ Error';
            scriptStatus.className = 'status-badge error';
            scriptOutput.textContent = `ERROR:\n${data.error || 'Error desconocido'}\n\nSALIDA:\n${data.output || '(ninguna)'}`;
        }

    } catch (error) {
        scriptStatus.textContent = '❌ Error';
        scriptStatus.className = 'status-badge error';
        scriptOutput.textContent = `Error de conexión: ${error.message}\n\nAsegúrate de que el servidor está corriendo.`;
    } finally {
        // Re-enable button
        btn.disabled = false;
        btn.textContent = 'Ejecutar';
    }
}

// ===========================
// SUBTABS NAVIGATION (Herramientas)
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    setupSubtabs();
    setupModelTypeChange();
    setupFileInputs();
});

function setupSubtabs() {
    const subtabBtns = document.querySelectorAll('.subtab-btn');
    const subtabContents = document.querySelectorAll('.subtab-content');

    subtabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const subtabName = btn.dataset.subtab;

            // Remove active from all
            subtabBtns.forEach(b => b.classList.remove('active'));
            subtabContents.forEach(c => c.classList.remove('active'));

            // Add active to clicked
            btn.classList.add('active');
            document.getElementById(`subtab-${subtabName}`).classList.add('active');
        });
    });
}

function setupModelTypeChange() {
    const modelTypeSelect = document.getElementById('modelType');
    const nEstimatorsGroup = document.getElementById('nEstimatorsGroup');

    if (modelTypeSelect && nEstimatorsGroup) {
        modelTypeSelect.addEventListener('change', () => {
            const showEstimators = ['random_forest', 'gradient_boosting'].includes(modelTypeSelect.value);
            nEstimatorsGroup.style.display = showEstimators ? 'block' : 'none';
        });
    }
}

function setupFileInputs() {
    const testFileInput = document.getElementById('testSpectrumFileInput');
    const testFilePath  = document.getElementById('testSpectrumFile');

    if (testFileInput && testFilePath) {
        testFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            testFilePath.value = `Subiendo ${file.name}...`;
            testFilePath.disabled = true;

            try {
                const formData = new FormData();
                formData.append('file', file);

                const resp = await fetch('/upload_single', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();

                if (data.success) {
                    testFilePath.value = data.server_path;
                } else {
                    testFilePath.value = '';
                    alert(`Error al subir archivo: ${data.error}`);
                }
            } catch (err) {
                testFilePath.value = '';
                alert(`Error de conexion: ${err.message}`);
            } finally {
                testFilePath.disabled = false;
                testFileInput.value = '';
            }
        });
    }
}

// ===========================
// ENTRENAR MODELOS
// ===========================

async function trainModel() {
    const btn          = document.getElementById('btnTrainModel');
    const progressDiv  = document.getElementById('trainProgress');
    const progressFill = document.getElementById('trainProgressFill');
    const progressText = document.getElementById('trainProgressText');
    const trainLog     = document.getElementById('trainLog');
    const resultsDiv   = document.getElementById('trainResults');

    const config = {
        catalog_path: document.getElementById('catalogPath').value,
        model_type:   document.getElementById('modelType').value,
        max_depth:    parseInt(document.getElementById('maxDepth').value),
        test_size:    parseInt(document.getElementById('testSize').value),
        n_estimators: parseInt(document.getElementById('nEstimators').value),
        output_path:  document.getElementById('outputPath').value
    };

    btn.disabled = true;
    btn.textContent = 'Entrenando...';
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    progressFill.style.background = 'linear-gradient(90deg, var(--primary) 0%, var(--success) 100%)';
    progressFill.style.width = '3%';
    progressText.textContent = 'Iniciando...';
    trainLog.innerHTML = '';

    // Clasificar línea para aplicar color en la consola
    function lineClass(msg) {
        if (msg.includes('[OK]') || msg.includes('exitosamente'))      return 'tl-ok';
        if (msg.includes('[ERROR]') || msg.includes('Error'))          return 'tl-error';
        if (msg.includes('[!]'))                                        return 'tl-warn';
        if (msg.startsWith('=') || msg.startsWith('ENTRENAMIENTO') ||
            msg.startsWith('VALIDAC') || msg.startsWith('RANDOM') ||
            msg.startsWith('GRADIENT') || msg.startsWith('CARGA'))     return 'tl-head';
        if (msg.includes('Procesados:') || msg.includes('Accuracy'))   return 'tl-pct';
        return '';
    }

    function appendLog(msg) {
        const el = document.createElement('div');
        el.className = 'tl-line ' + lineClass(msg);
        el.textContent = msg;
        trainLog.appendChild(el);
        trainLog.scrollTop = trainLog.scrollHeight;
    }

    const startTime = Date.now();

    try {
        const response = await fetch('/train_model_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop();           // último fragmento incompleto

            for (const part of parts) {
                if (!part.startsWith('data: ')) continue;
                let event;
                try { event = JSON.parse(part.slice(6)); } catch { continue; }

                if (event.type === 'line') {
                    progressFill.style.width = `${event.pct}%`;
                    progressText.textContent = event.msg.replace(/^\[.*?\]\s*/, '').substring(0, 60) || progressText.textContent;
                    appendLog(event.msg);

                } else if (event.type === 'done') {
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

                    if (event.success) {
                        progressFill.style.width = '100%';
                        progressText.textContent = 'Entrenamiento completado';
                        appendLog(`[OK] Listo en ${elapsed}s — Accuracy: ${event.accuracy}%`);

                        setTimeout(() => {
                            progressDiv.style.display = 'none';
                            resultsDiv.style.display = 'block';
                            // Mostrar accuracy en la sección de resultados
                            document.getElementById('trainAccuracy').textContent = event.accuracy;
                            document.getElementById('trainSamples').textContent  = event.n_samples || '-';
                            document.getElementById('trainTime').textContent     = `${elapsed}s`;
                            document.getElementById('trainDetailsOutput').textContent = 'Entrenamiento completado correctamente.';
                            // Barra de accuracy
                            const bar = document.getElementById('trainAccuracyBar');
                            if (bar) { bar.style.width = Math.min(event.accuracy, 100) + '%'; bar.style.background = event.accuracy >= 80 ? '#22c55e' : event.accuracy >= 60 ? '#f59e0b' : '#ef4444'; }
                            // Actualizar banner del modelo actual
                            _updateCurrentModelBanner({ accuracy: event.accuracy, n_samples: event.n_samples, timestamp: new Date().toLocaleDateString() });
                        }, 800);
                    } else {
                        progressFill.style.width = '100%';
                        progressFill.style.background = 'var(--danger)';
                        progressText.textContent = 'Error en entrenamiento';
                        appendLog(`[ERROR] ${event.msg || 'Error desconocido'}`);
                    }
                }
            }
        }

    } catch (error) {
        progressFill.style.background = 'var(--danger)';
        progressText.textContent = 'Error de conexion';
        appendLog(`[ERROR] ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Iniciar Entrenamiento';
    }
}

function resetTrainConfig() {
    document.getElementById('catalogPath').value = 'data/elodie/';
    document.getElementById('modelType').value = 'decision_tree';
    document.getElementById('maxDepth').value = '9';
    document.getElementById('testSize').value = '20';
    document.getElementById('nEstimators').value = '100';
    document.getElementById('outputPath').value = 'models/';
    document.getElementById('nEstimatorsGroup').style.display = 'none';
}

// ─── Banner "Modelo Actual" en la pestaña de entrenamiento ───────────────────

function _updateCurrentModelBanner(info) {
    const banner = document.getElementById('currentModelBanner');
    if (!banner) return;
    const acc = parseFloat(info.accuracy) || 0;
    const accEl    = document.getElementById('cmbAccuracy');
    const sampEl   = document.getElementById('cmbSamples');
    const tsEl     = document.getElementById('cmbTimestamp');
    const barEl    = document.getElementById('cmbBarFill');
    const typeEl   = document.getElementById('cmbModelType');

    if (accEl)  accEl.textContent  = acc.toFixed(1) + '%';
    if (sampEl) sampEl.textContent = info.n_samples || info.n_train + info.n_test || '—';
    if (tsEl)   tsEl.textContent   = info.timestamp || '—';
    if (typeEl) typeEl.textContent = info.model_type || 'Árbol de Decisión';
    if (barEl) {
        barEl.style.width      = Math.min(acc, 100) + '%';
        barEl.style.background = acc >= 85 ? '#22c55e' : acc >= 70 ? '#f59e0b' : '#ef4444';
    }
    banner.style.display = 'flex';
}

async function loadCurrentModelBanner() {
    try {
        const resp = await fetch('/get_metrics');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.accuracy_test !== undefined) {
            _updateCurrentModelBanner({
                accuracy:    data.accuracy_test,   // ya viene en % desde la API
                n_samples:   (data.n_train || 0) + (data.n_test || 0),
                timestamp:   data.timestamp || '—',
                model_type:  data.model_type || 'Árbol de Decisión'
            });
        }
    } catch (_) {}
}

// Llamar al cargar la página y al activar la pestaña de herramientas
document.addEventListener('DOMContentLoaded', () => { loadCurrentModelBanner(); });


function browseCatalog() {
    // Abrir listado de catálogos
    listCatalogs();
}

// Toggle y cargar catálogos para la sección de entrenamiento
async function toggleTrainCatalogList() {
    const listDiv = document.getElementById('trainCatalogsList');
    const contentDiv = document.getElementById('trainCatalogsContent');

    if (listDiv.style.display === 'none') {
        listDiv.style.display = 'block';
        contentDiv.innerHTML = '<div class="loading-hint">⏳ Cargando catálogos...</div>';

        try {
            const response = await fetch('/list_catalogs');
            const data = await response.json();

            if (data.success && data.catalogs.length > 0) {
                contentDiv.innerHTML = '';
                data.catalogs.forEach(catalog => {
                    const typesStr = Object.entries(catalog.types || {})
                        .map(([t, c]) => `<span class="type-badge-small">${t}:${c}</span>`)
                        .join(' ');

                    const item = document.createElement('div');
                    item.className = 'catalog-inline-item';
                    item.innerHTML = `
                        <div class="catalog-inline-info">
                            <strong>📂 ${catalog.name}</strong>
                            <span class="catalog-count">${catalog.n_files} archivos</span>
                        </div>
                        <div class="catalog-inline-types">${typesStr}</div>
                        <button class="btn-small btn-use" onclick="selectTrainCatalog('${catalog.path}')">Usar</button>
                    `;
                    contentDiv.appendChild(item);
                });
            } else {
                contentDiv.innerHTML = '<div class="no-catalogs">No se encontraron catálogos. Verifica que existan carpetas con archivos *_tipo*.txt</div>';
            }
        } catch (error) {
            contentDiv.innerHTML = `<div class="error-hint">❌ Error: ${error.message}</div>`;
        }
    } else {
        listDiv.style.display = 'none';
    }
}

// Seleccionar un catálogo para entrenamiento
function selectTrainCatalog(path) {
    document.getElementById('catalogPath').value = path;
    document.getElementById('trainCatalogsList').style.display = 'none';
}

// ===========================
// FILE BROWSER - CATALOG FILES
// ===========================

let catalogFilesCache = [];

async function loadCatalogFiles() {
    const catalog = document.getElementById('testCatalogSelect').value;
    const fileList = document.getElementById('catalogFileList');

    fileList.innerHTML = '<div class="file-list-loading">⏳ Cargando archivos...</div>';

    try {
        const response = await fetch(`/list_catalog_files/${catalog}?limit=500`);
        const data = await response.json();

        if (data.success) {
            catalogFilesCache = data.files;
            renderFileList(data.files);
        } else {
            fileList.innerHTML = `<div class="file-list-hint">❌ Error: ${data.error}</div>`;
        }
    } catch (error) {
        fileList.innerHTML = `<div class="file-list-hint">❌ Error de conexión</div>`;
    }
}

function filterCatalogFiles() {
    const searchText = document.getElementById('testFileSearch').value.toLowerCase();
    const typeFilter = document.getElementById('testTypeFilter').value;

    let filtered = catalogFilesCache;

    if (searchText) {
        filtered = filtered.filter(f => f.filename.toLowerCase().includes(searchText));
    }

    if (typeFilter) {
        filtered = filtered.filter(f => f.tipo && f.tipo.startsWith(typeFilter));
    }

    renderFileList(filtered);
}

function renderFileList(files) {
    const fileList = document.getElementById('catalogFileList');

    if (files.length === 0) {
        fileList.innerHTML = '<div class="file-list-hint">No se encontraron archivos</div>';
        return;
    }

    let html = '';
    const maxShow = 50;  // Mostrar máximo 50 para no sobrecargar

    files.slice(0, maxShow).forEach(f => {
        html += `
            <div class="file-list-item" onclick="selectCatalogFile('${f.filename}', '${f.path}')">
                <span class="file-name">${f.filename}</span>
                <span class="file-type">${f.tipo || '?'}</span>
            </div>
        `;
    });

    if (files.length > maxShow) {
        html += `<div class="file-list-count">Mostrando ${maxShow} de ${files.length} archivos. Use el buscador para filtrar.</div>`;
    } else {
        html += `<div class="file-list-count">${files.length} archivo(s) encontrado(s)</div>`;
    }

    fileList.innerHTML = html;
}

function selectCatalogFile(filename, path) {
    // Marcar como seleccionado en la lista
    document.querySelectorAll('.file-list-item').forEach(item => {
        item.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    // Poner el nombre del archivo en el campo de texto
    document.getElementById('testSpectrumFile').value = filename;
}

// Cargar archivos al iniciar si estamos en la pestaña
document.addEventListener('DOMContentLoaded', function() {
    // Delay para asegurar que el DOM esté listo
    setTimeout(() => {
        if (document.getElementById('testCatalogSelect')) {
            loadCatalogFiles();
        }
    }, 500);
});

// ===========================
// TEST ESPECTROS
// ===========================

async function testSpectrum() {
    const btn = document.getElementById('btnTestSpectrum');
    const resultsDiv = document.getElementById('testResults');

    const spectrumPath = document.getElementById('testSpectrumFile').value;
    const method = document.getElementById('testModelSelect').value;
    const detailLevel = document.getElementById('testShowDetails').value;
    const savePlot = document.getElementById('testSavePlot').checked;

    if (!spectrumPath) {
        alert('Por favor, especifica la ruta del espectro');
        return;
    }

    btn.disabled = true;
    btn.textContent = '🔬 Analizando...';
    resultsDiv.style.display = 'none';

    try {
        const response = await fetch('/test_spectrum_advanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                spectrum_path: spectrumPath,
                method: method,
                detail_level: detailLevel,
                save_plot: savePlot
            })
        });

        const data = await response.json();

        if (data.success) {
            displayTestResults(data);
        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔬 Ejecutar Test';
    }
}

function displayTestResults(data) {
    const resultsDiv = document.getElementById('testResults');
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });

    // Tipo clasificado
    document.getElementById('testResultType').textContent = data.tipo_clasificado;
    document.getElementById('testResultSubtype').textContent = data.subtipo;

    // Confianza
    const confValue = data.confianza || 0;
    document.getElementById('testConfidenceValue').textContent = `${confValue.toFixed(0)}%`;

    // Actualizar círculo de confianza
    const circle = document.getElementById('testConfidenceCircle');
    circle.style.background = `conic-gradient(var(--success) ${confValue}%, var(--light) ${confValue}%)`;

    // Comparación con original
    if (data.original_type && data.original_type !== 'N/A') {
        document.getElementById('testComparison').style.display = 'block';
        document.getElementById('testOriginalType').textContent = data.original_type;
        document.getElementById('testClassifiedType').textContent = data.tipo_clasificado;

        const matchEl = document.getElementById('testMatchResult');
        if (data.is_match) {
            matchEl.textContent = '✅ COINCIDE';
            matchEl.className = 'comparison-value match';
        } else {
            matchEl.textContent = '❌ NO COINCIDE';
            matchEl.className = 'comparison-value no-match';
        }
    } else {
        document.getElementById('testComparison').style.display = 'none';
    }

    const detailLevel = document.getElementById('testShowDetails').value;

    // Líneas detectadas
    const linesTable = document.getElementById('testLinesTable');
    linesTable.innerHTML = '';

    const detailsPanel = document.getElementById('testDetailsPanel');
    detailsPanel.style.display = (detailLevel === 'basic') ? 'none' : 'block';

    if (detailLevel !== 'basic' && data.lineas_detectadas && data.lineas_detectadas.length > 0) {
        data.lineas_detectadas.forEach(line => {
            const row = document.createElement('div');
            row.className = 'line-row';

            const ewStr   = line.ancho_equivalente !== undefined ? line.ancho_equivalente.toFixed(3) + ' Å' : '-';
            const depStr  = line.profundidad        !== undefined ? '(prof. ' + line.profundidad.toFixed(3) + ')' : '';
            const lambStr = line.longitud_onda      !== undefined ? line.longitud_onda + ' Å' : '';

            if (detailLevel === 'debug') {
                row.innerHTML = `
                    <span class="line-name">${line.nombre} <small style="color:var(--gray)">${lambStr}</small></span>
                    <span class="line-value">${ewStr} <span class="line-depth">${depStr}</span></span>
                `;
            } else {
                row.innerHTML = `
                    <span class="line-name">${line.nombre}</span>
                    <span class="line-value">${ewStr}</span>
                `;
            }
            linesTable.appendChild(row);
        });
    }

    // Panel de diagnósticos internos (solo debug)
    const debugPanel   = document.getElementById('testDebugPanel');
    const debugContent = document.getElementById('testDebugContent');

    if (detailLevel === 'debug' && data.diagnostics && Object.keys(data.diagnostics).length > 0) {
        debugPanel.style.display = 'block';
        debugContent.innerHTML = '';

        function renderDebug(obj, prefix) {
            Object.entries(obj).forEach(([k, v]) => {
                const fullKey = prefix ? `${prefix}.${k}` : k;
                if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
                    renderDebug(v, fullKey);
                } else {
                    const valStr = Array.isArray(v) ? JSON.stringify(v).substring(0, 80) : String(v);
                    const row = document.createElement('div');
                    row.className = 'debug-row';
                    row.innerHTML = `<span class="debug-key">${fullKey}</span><span class="debug-val">${valStr}</span>`;
                    debugContent.appendChild(row);
                }
            });
        }
        renderDebug(data.diagnostics, '');
    } else {
        debugPanel.style.display = 'none';
    }

    // Gráfico
    if (data.plot_url) {
        document.getElementById('testPlotPanel').style.display = 'block';
        document.getElementById('testPlotImage').src = data.plot_url + '?t=' + Date.now();
    } else {
        document.getElementById('testPlotPanel').style.display = 'none';
    }
}

function clearTestResults() {
    document.getElementById('testResults').style.display = 'none';
    document.getElementById('testSpectrumFile').value = '';
}

// ===========================
// MÉTRICAS
// ===========================

async function loadMetrics() {
    const metricsPanel = document.getElementById('metricsPanel');

    try {
        const response = await fetch('/get_metrics');
        const data = await response.json();

        if (data.success) {
            metricsPanel.style.display = 'block';

            // Resumen
            document.getElementById('metricAccuracy').textContent = `${data.accuracy_test}%`;
            document.getElementById('metricSamples').textContent = data.n_train + data.n_test;
            document.getElementById('metricDepth').textContent = '-'; // TODO: cargar desde metadata

            // Accuracy por tipo
            const typeBars = document.getElementById('typeAccuracyBars');
            typeBars.innerHTML = '';

            const types = ['O', 'B', 'A', 'F', 'G', 'K', 'M'];
            types.forEach(tipo => {
                const typeData = data.accuracy_by_type[tipo];
                if (typeData) {
                    const bar = document.createElement('div');
                    bar.className = 'type-bar';
                    bar.innerHTML = `
                        <span class="type-label">${tipo}</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${typeData.accuracy}%"></div>
                        </div>
                        <span class="bar-value">${typeData.accuracy.toFixed(1)}%</span>
                    `;
                    typeBars.appendChild(bar);
                }
            });

            // Matriz de confusión
            if (data.has_confusion_matrix) {
                document.getElementById('confusionMatrixImg').src = '/confusion_matrix?t=' + Date.now();
            }

            // Top features
            const featuresList = document.getElementById('topFeaturesList');
            featuresList.innerHTML = '';
            data.top_features.forEach((feature, idx) => {
                const item = document.createElement('div');
                item.className = 'feature-item';
                item.innerHTML = `
                    <span>${idx + 1}. ${feature}</span>
                `;
                featuresList.appendChild(item);
            });

        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// ===========================
// UTILIDADES
// ===========================

async function listCatalogs() {
    const catalogsList = document.getElementById('catalogsList');
    const catalogsContent = document.getElementById('catalogsContent');

    try {
        const response = await fetch('/list_catalogs');
        const data = await response.json();

        if (data.success) {
            catalogsList.style.display = 'block';
            catalogsContent.innerHTML = '';

            if (data.catalogs.length === 0) {
                catalogsContent.innerHTML = '<p>No se encontraron catálogos con espectros etiquetados.</p>';
                return;
            }

            data.catalogs.forEach(catalog => {
                const item = document.createElement('div');
                item.className = 'catalog-item';

                const typesStr = Object.entries(catalog.types)
                    .map(([t, c]) => `${t}:${c}`)
                    .join(', ');

                item.innerHTML = `
                    <div>
                        <span class="catalog-name">📂 ${catalog.name}</span>
                        <span class="catalog-count">(${catalog.n_files} archivos)</span>
                    </div>
                    <div class="catalog-types">${typesStr}</div>
                    <div class="catalog-actions">
                        <button class="btn-small" onclick="useCatalog('${catalog.path}')">Usar</button>
                    </div>
                `;
                catalogsContent.appendChild(item);
            });

        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function useCatalog(path) {
    document.getElementById('catalogPath').value = path;

    // Cambiar a subtab de entrenar
    document.querySelectorAll('.subtab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));
    document.querySelector('[data-subtab="entrenar"]').classList.add('active');
    document.getElementById('subtab-entrenar').classList.add('active');
}

async function verifyModel() {
    try {
        const response = await fetch('/verify_model');
        const data = await response.json();

        let message = `Estado del modelo: ${data.status}\n\n`;
        message += `Archivo del modelo: ${data.checks.model_file ? '✅' : '❌'}\n`;
        message += `Metadata: ${data.checks.metadata ? '✅' : '❌'}\n`;
        message += `Matriz de confusión: ${data.checks.confusion_matrix ? '✅' : '❌'}\n`;
        message += `Reporte de validación: ${data.checks.validation_report ? '✅' : '❌'}\n`;
        message += `Multi-método disponible: ${data.checks.multi_method_available ? '✅' : '❌'}\n`;

        if (data.model_info.accuracy) {
            message += `\nAccuracy: ${data.model_info.accuracy}%\n`;
            message += `Muestras: ${data.model_info.n_samples}\n`;
            message += `Última actualización: ${data.model_info.timestamp}`;
        }

        alert(message);

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function clearCache() {
    if (!confirm('¿Estás seguro de que deseas eliminar todos los archivos temporales?')) {
        return;
    }

    try {
        const response = await fetch('/clear_cache', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert(data.message);
        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// ===========================
// SPECTRUM ZOOM (Simple - solo rueda del ratón)
// ===========================

let zoomState = {
    scale: 1,
    originX: 50,
    originY: 50
};

function initSpectrumZoom() {
    const wrapper = document.getElementById('spectrumWrapper');
    const img = document.getElementById('spectrumPlot');
    const controls = document.getElementById('zoomControls');

    if (!wrapper || !img || !controls) return;

    img.onload = function() {
        controls.style.display = 'flex';
        img.style.width = '100%';
        img.style.height = 'auto';
        resetZoom();

        // Zoom con rueda del ratón en la posición del cursor
        wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();

            const rect = img.getBoundingClientRect();
            // Posición del mouse relativa a la imagen (0-100%)
            const mouseX = ((e.clientX - rect.left) / rect.width) * 100;
            const mouseY = ((e.clientY - rect.top) / rect.height) * 100;

            // Zoom in o out
            const delta = e.deltaY > 0 ? 0.85 : 1.2;
            zoomState.scale *= delta;
            zoomState.scale = Math.max(1, Math.min(4, zoomState.scale));

            // Centrar zoom en la posición del cursor
            if (zoomState.scale > 1) {
                zoomState.originX = mouseX;
                zoomState.originY = mouseY;
            } else {
                zoomState.originX = 50;
                zoomState.originY = 50;
            }

            applyZoom(img);
        }, { passive: false });

        // Doble clic para resetear
        wrapper.addEventListener('dblclick', () => resetZoom());

        // Botones de control
        document.getElementById('zoomIn').addEventListener('click', () => {
            zoomState.scale = Math.min(4, zoomState.scale * 1.3);
            applyZoom(img);
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            zoomState.scale = Math.max(1, zoomState.scale * 0.7);
            if (zoomState.scale === 1) {
                zoomState.originX = 50;
                zoomState.originY = 50;
            }
            applyZoom(img);
        });

        document.getElementById('zoomReset').addEventListener('click', () => resetZoom());
    };

    function applyZoom(img) {
        img.style.transformOrigin = `${zoomState.originX}% ${zoomState.originY}%`;
        img.style.transform = `scale(${zoomState.scale})`;
        document.getElementById('zoomLevel').textContent = `${Math.round(zoomState.scale * 100)}%`;
    }

    function resetZoom() {
        zoomState.scale = 1;
        zoomState.originX = 50;
        zoomState.originY = 50;
        applyZoom(img);
    }
}

// ===========================
// REDES NEURONALES (KNN/CNN)
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    setupNeuralTab();
});

function setupNeuralTab() {
    // Cambiar parámetros según tipo de modelo seleccionado
    const modelCards = document.querySelectorAll('.model-card input[name="neuralType"]');
    modelCards.forEach(radio => {
        radio.addEventListener('change', updateNeuralParams);
    });

    // También actualizar la visualización de las tarjetas
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', function() {
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
        });
    });

    // Slider de dropout
    const dropoutSlider = document.getElementById('cnnDropout');
    if (dropoutSlider) {
        dropoutSlider.addEventListener('input', (e) => {
            document.getElementById('cnnDropoutValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Slider de peso en votación
    const weightSlider = document.getElementById('nnVotingWeight');
    if (weightSlider) {
        weightSlider.addEventListener('input', (e) => {
            const weight = parseFloat(e.target.value);
            document.getElementById('nnVotingWeightValue').textContent = weight.toFixed(2);
            updateWeightsDisplay();
            saveNeuralConfig();
        });
    }

    // Checkbox incluir neural → habilitar/deshabilitar selector y slider
    const nnCheckbox = document.getElementById('nnIncludeInVoting');
    if (nnCheckbox) {
        nnCheckbox.addEventListener('change', () => {
            const enabled = nnCheckbox.checked;
            document.getElementById('nnModelSelectorContainer').style.opacity = enabled ? '1' : '0.45';
            document.getElementById('nnModelSelect').disabled = !enabled;
            document.getElementById('nnVotingWeight').disabled = !enabled;
            saveNeuralConfig();
        });
    }

    // Selector de modelo → actualizar badge de accuracy y guardar
    const modelSelect = document.getElementById('nnModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            updateNeuralModelBadge();
            saveNeuralConfig();
        });
    }

    // Restaurar configuración guardada antes de cargar modelos
    restoreNeuralConfig();

    // Cargar modelos disponibles al init del tab neural
    populateNeuralModelSelect();
}

// ── Persistencia de configuración neural (localStorage) ──────────────────────
const NEURAL_CONFIG_KEY = 'spectral_neural_config';

function saveNeuralConfig() {
    const config = {
        model:          document.getElementById('nnModelSelect')?.value     ?? 'auto',
        weight:         document.getElementById('nnVotingWeight')?.value    ?? '0.40',
        includeNeural:  document.getElementById('nnIncludeInVoting')?.checked ?? true
    };
    localStorage.setItem(NEURAL_CONFIG_KEY, JSON.stringify(config));
}

function restoreNeuralConfig() {
    const raw = localStorage.getItem(NEURAL_CONFIG_KEY);
    if (!raw) return;

    try {
        const config = JSON.parse(raw);

        const checkbox = document.getElementById('nnIncludeInVoting');
        const slider   = document.getElementById('nnVotingWeight');
        const select   = document.getElementById('nnModelSelect');

        if (checkbox && config.includeNeural !== undefined) {
            checkbox.checked = config.includeNeural;
            // reflejar estado visual
            const enabled = config.includeNeural;
            document.getElementById('nnModelSelectorContainer').style.opacity = enabled ? '1' : '0.45';
            select.disabled  = !enabled;
            slider.disabled  = !enabled;
        }

        if (slider && config.weight !== undefined) {
            slider.value = config.weight;
            document.getElementById('nnVotingWeightValue').textContent =
                parseFloat(config.weight).toFixed(2);
            updateWeightsDisplay();
        }

        // El modelo se restaura en populateNeuralModelSelect() una vez que
        // se conocen los modelos disponibles
        if (config.model) {
            // guardarlo temporalmente para usarlo en populate
            window._pendingNeuralModel = config.model;
        }
    } catch (e) {
        // JSON corrupto → ignorar
        localStorage.removeItem(NEURAL_CONFIG_KEY);
    }
}
// ─────────────────────────────────────────────────────────────────────────────

// Cache de modelos disponibles con su accuracy
let _neuralModelsCache = {};

async function populateNeuralModelSelect() {
    const select = document.getElementById('nnModelSelect');
    const hint   = document.getElementById('nnModelHint');
    if (!select) return;

    try {
        const resp = await fetch('/neural_metrics');
        const data = await resp.json();

        if (!data.success) { hint.textContent = 'No se pudieron cargar los modelos.'; return; }

        _neuralModelsCache = data.models || {};
        const available = Object.keys(_neuralModelsCache);

        // Reconstruir opciones
        select.innerHTML = '<option value="auto">Auto (mejor disponible)</option>';
        const labels = { knn: 'KNN', cnn_1d: 'CNN 1D', cnn_2d: 'CNN 2D' };
        available.forEach(m => {
            const acc = _neuralModelsCache[m]?.accuracy;
            const label = `${labels[m] || m}${acc != null ? ' — ' + acc.toFixed(1) + '%' : ''}`;
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = label;
            select.appendChild(opt);
        });

        if (available.length === 0) {
            hint.textContent = 'No hay modelos entrenados aún. Entrena KNN o CNN primero.';
        } else {
            hint.textContent = `${available.length} modelo(s) disponible(s). "Auto" elige el de mayor accuracy.`;
        }

        // Restaurar modelo guardado si existe y está disponible en el select
        if (window._pendingNeuralModel) {
            const pending = window._pendingNeuralModel;
            const optExists = Array.from(select.options).some(o => o.value === pending);
            if (optExists) select.value = pending;
            window._pendingNeuralModel = null;
        }

        updateNeuralModelBadge();

    } catch (e) {
        if (hint) hint.textContent = 'Error cargando modelos.';
    }
}

function updateNeuralModelBadge() {
    const select = document.getElementById('nnModelSelect');
    const badge  = document.getElementById('nnModelAccuracyBadge');
    if (!select || !badge) return;

    const val = select.value;
    if (val === 'auto' || !_neuralModelsCache[val]) {
        badge.style.display = 'none';
        return;
    }
    const acc = _neuralModelsCache[val]?.accuracy;
    if (acc != null) {
        badge.textContent = acc.toFixed(1) + '%';
        badge.className = 'accuracy-badge ' + (acc >= 70 ? 'acc-good' : 'acc-low');
        badge.style.display = 'inline-block';
    } else {
        badge.style.display = 'none';
    }
}

function updateNeuralParams() {
    const type = document.querySelector('input[name="neuralType"]:checked').value;

    // Mostrar/ocultar paneles según tipo
    document.getElementById('knnParams').style.display = type === 'knn' ? 'block' : 'none';
    document.getElementById('cnnParams').style.display = type.startsWith('cnn') ? 'block' : 'none';
    document.getElementById('cnn2dParams').style.display = type === 'cnn_2d' ? 'block' : 'none';
}

function updateWeightsDisplay() {
    // Muestra los pesos actuales de globalWeights (6 métodos independientes)
    const w = globalWeights;
    const weightsDisplay = document.getElementById('currentWeightsDisplay');
    if (!weightsDisplay) return;

    const rows = [
        ['Físico',            w.physical,      '#e17055'],
        ['Árbol de Decisión', w.decision_tree, '#00b894'],
        ['Template',          w.template,      '#fdcb6e'],
        ['KNN',               w.knn,           '#0984e3'],
        ['CNN 1D',            w.cnn_1d,        '#74b9ff'],
        ['CNN 2D',            w.cnn_2d,        '#a29bfe'],
    ];

    weightsDisplay.innerHTML = `
        <h4>Pesos Actuales:</h4>
        <div class="weights-bars">
            ${rows.map(([label, val, color]) => `
            <div class="weight-bar">
                <span class="weight-label">${label}</span>
                <div class="weight-fill" style="width:${(val*100).toFixed(1)}%; background:${color}"></div>
                <span class="weight-value">${val.toFixed(2)}</span>
            </div>`).join('')}
        </div>`;
}

async function trainNeuralModel() {
    const type = document.querySelector('input[name="neuralType"]:checked').value;
    const btn = document.getElementById('btnTrainNeural');
    const progressDiv = document.getElementById('nnProgress');
    const progressFill = document.getElementById('nnProgressFill');
    const progressText = document.getElementById('nnProgressText');
    const progressTitle = document.getElementById('nnProgressTitle');
    const resultsDiv = document.getElementById('nnResults');

    // Construir configuración según tipo de modelo
    const config = {
        model_type: type,
        catalog_path: document.getElementById('nnCatalogPath').value,
        test_size: parseInt(document.getElementById('nnTestSize').value) / 100,
        output_path: document.getElementById('nnOutputPath').value
    };

    // Parámetros específicos según tipo
    if (type === 'knn') {
        config.n_neighbors = parseInt(document.getElementById('knnNeighbors').value);
        config.weights = document.getElementById('knnWeights').value;
        config.metric = document.getElementById('knnMetric').value;
    } else {
        // CNN params
        config.epochs = parseInt(document.getElementById('cnnEpochs').value);
        config.batch_size = parseInt(document.getElementById('cnnBatchSize').value);
        config.learning_rate = parseFloat(document.getElementById('cnnLearningRate').value);
        config.dropout_rate = parseFloat(document.getElementById('cnnDropout').value);
        config.dense_units = parseInt(document.getElementById('cnnDenseUnits').value);

        if (type === 'cnn_2d') {
            config.image_size = parseInt(document.getElementById('cnn2dImageSize').value);
            config.image_dir = document.getElementById('cnn2dImageDir').value;
            config.labels_csv = document.getElementById('cnn2dLabelsCSV').value;
        }
    }

    const typeNames = { 'knn': 'KNN', 'cnn_1d': 'CNN 1D', 'cnn_2d': 'CNN 2D' };
    const nnLog = document.getElementById('nnLog');

    // Inicializar UI
    btn.disabled = true;
    btn.textContent = 'Entrenando...';
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    progressTitle.textContent = `Entrenando modelo ${typeNames[type]}...`;
    progressFill.style.background = 'linear-gradient(90deg, var(--primary) 0%, var(--success) 100%)';
    progressFill.style.width = '3%';
    progressText.textContent = 'Iniciando...';
    nnLog.innerHTML = '';

    // Inicializar gráficas en vivo para CNN
    if (type === 'cnn_1d' || type === 'cnn_2d') {
        _initLiveCharts();
    }

    function lineClass(msg) {
        if (msg.includes('[OK]') || msg.includes('completado') || msg.includes('exitosamente')) return 'tl-ok';
        if (msg.includes('[ERROR]') || msg.includes('Error') || msg.includes('ERROR'))          return 'tl-error';
        if (msg.includes('[!]') || msg.includes('ADVERTENCIA'))                                  return 'tl-warn';
        if (msg.startsWith('=') || msg.includes('ENTRENANDO') || msg.includes('INICIANDO'))      return 'tl-head';
        if (msg.includes('Accuracy') || msg.includes('Procesados') || msg.includes('Epocas'))   return 'tl-pct';
        return '';
    }

    function appendLog(msg) {
        const el = document.createElement('div');
        el.className = 'tl-line ' + lineClass(msg);
        el.textContent = msg;
        nnLog.appendChild(el);
        nnLog.scrollTop = nnLog.scrollHeight;
    }

    const startTime = Date.now();

    try {
        const response = await fetch('/train_neural_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop();

            for (const part of parts) {
                if (!part.startsWith('data: ')) continue;
                let event;
                try { event = JSON.parse(part.slice(6)); } catch { continue; }

                if (event.type === 'epoch') {
                    // Actualizar gráfica en tiempo real
                    _updateLiveCharts(event);
                    appendLog(`Época ${event.epoch} — acc: ${(event.acc*100).toFixed(1)}%  val_acc: ${(event.val_acc*100).toFixed(1)}%  loss: ${event.loss}`);

                } else if (event.type === 'line') {
                    progressFill.style.width = `${event.pct}%`;
                    progressText.textContent = event.msg.replace(/^\[.*?\]\s*/, '').substring(0, 60) || progressText.textContent;
                    appendLog(event.msg);

                } else if (event.type === 'done') {
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

                    if (event.success) {
                        progressFill.style.width = '100%';
                        progressText.textContent = 'Entrenamiento completado';
                        appendLog(`[OK] Listo en ${elapsed}s — Accuracy: ${event.accuracy}%`);

                        setTimeout(() => {
                            progressDiv.style.display = 'none';
                            resultsDiv.style.display = 'block';
                            document.getElementById('nnAccuracy').textContent = `${event.accuracy}%`;
                            document.getElementById('nnSamples').textContent  = event.n_samples || '-';
                            document.getElementById('nnClasses').textContent  = event.n_classes || '-';
                            document.getElementById('nnTime').textContent     = `${elapsed}s`;
                            document.getElementById('nnResultsDetails').innerHTML =
                                `<p>Modelo ${typeNames[type]} guardado correctamente.</p>`;
                            resultsDiv.scrollIntoView({ behavior: 'smooth' });
                            // Cargar métricas finales y asegurar que se muestra el modelo correcto
                            loadTrainingHistory(type);
                        }, 800);
                    } else {
                        progressFill.style.width = '100%';
                        progressFill.style.background = 'var(--danger)';
                        progressText.textContent = 'Error en entrenamiento';
                        appendLog(`[ERROR] ${event.msg || 'Error desconocido'}`);
                    }
                }
            }
        }

    } catch (error) {
        progressFill.style.background = 'var(--danger)';
        progressText.textContent = 'Error de conexion';
        appendLog(`[ERROR] ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Entrenar Modelo';
    }
}

async function loadNeuralMetrics() {
    const modelsPanel = document.getElementById('availableModelsPanel');
    const modelsList = document.getElementById('availableModelsList');

    try {
        const response = await fetch('/neural_metrics');
        const data = await response.json();

        if (data.success) {
            modelsPanel.style.display = 'block';
            modelsList.innerHTML = '';

            const models = data.models || {};
            const modelTypes = Object.keys(models);

            if (modelTypes.length === 0) {
                modelsList.innerHTML = '<p>No hay modelos neuronales entrenados aún.</p>';
                return;
            }

            modelTypes.forEach(modelType => {
                const info = models[modelType];
                const card = document.createElement('div');
                card.className = 'model-info-card';
                card.innerHTML = `
                    <h4>${modelType.toUpperCase()}</h4>
                    <div class="model-info-grid">
                        <div class="info-item">
                            <span class="info-label">Tipo:</span>
                            <span class="info-value">${info.tipo || modelType}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Clases:</span>
                            <span class="info-value">${info.clases && info.clases.length ? info.clases.join(', ') : '-'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Accuracy:</span>
                            <span class="info-value">${info.accuracy != null ? info.accuracy.toFixed(1) + '%' : '-'}</span>
                        </div>
                        ${info.n_samples ? `
                        <div class="info-item">
                            <span class="info-label">Muestras:</span>
                            <span class="info-value">${info.n_samples}</span>
                        </div>` : ''}
                        ${info.n_neighbors ? `
                        <div class="info-item">
                            <span class="info-label">K (vecinos):</span>
                            <span class="info-value">${info.n_neighbors}</span>
                        </div>` : ''}
                        ${info.epochs_trained ? `
                        <div class="info-item">
                            <span class="info-label">Epocas entrenadas:</span>
                            <span class="info-value">${info.epochs_trained}</span>
                        </div>` : ''}
                        ${info.spectrum_length ? `
                        <div class="info-item">
                            <span class="info-label">Longitud espectro:</span>
                            <span class="info-value">${info.spectrum_length}</span>
                        </div>` : ''}
                    </div>
                `;
                modelsList.appendChild(card);
            });

            modelsPanel.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function verifyNeuralModels() {
    try {
        const response = await fetch('/verify_neural_models');
        const data = await response.json();

        let message = '📦 Estado de Modelos Neuronales:\n\n';

        if (data.knn) {
            message += `✅ KNN: ${data.knn.path}\n`;
            message += `   - Clases: ${data.knn.classes || 'N/A'}\n\n`;
        } else {
            message += `❌ KNN: No encontrado\n\n`;
        }

        if (data.cnn_1d) {
            message += `✅ CNN 1D: ${data.cnn_1d.path}\n`;
            message += `   - Clases: ${data.cnn_1d.classes || 'N/A'}\n\n`;
        } else {
            message += `❌ CNN 1D: No encontrado\n\n`;
        }

        if (data.cnn_2d) {
            message += `✅ CNN 2D: ${data.cnn_2d.path}\n`;
            message += `   - Clases: ${data.cnn_2d.classes || 'N/A'}\n\n`;
        } else {
            message += `❌ CNN 2D: No encontrado\n\n`;
        }

        message += `Directorio de modelos: ${data.models_dir}`;

        alert(message);

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Funciones auxiliares para navegación de archivos

// Toggle catálogo inline para Red Neuronal (igual que training tab)
async function toggleNNCatalogList() {
    const listDiv = document.getElementById('nnCatalogsList');
    const contentDiv = document.getElementById('nnCatalogsContent');

    if (listDiv.style.display === 'none') {
        listDiv.style.display = 'block';
        contentDiv.innerHTML = '<div class="loading-hint">⏳ Cargando catálogos...</div>';

        try {
            const response = await fetch('/list_catalogs');
            const data = await response.json();

            if (data.success && data.catalogs.length > 0) {
                contentDiv.innerHTML = '';
                data.catalogs.forEach(catalog => {
                    const typesStr = Object.entries(catalog.types || {})
                        .map(([t, c]) => `<span class="type-badge-small">${t}:${c}</span>`)
                        .join(' ');
                    const item = document.createElement('div');
                    item.className = 'catalog-inline-item';
                    item.innerHTML = `
                        <div class="catalog-inline-info">
                            <strong>📂 ${catalog.name}</strong>
                            <span class="catalog-count">${catalog.n_files} archivos</span>
                        </div>
                        <div class="catalog-inline-types">${typesStr}</div>
                        <button class="btn-small btn-use" onclick="selectNNCatalog('${catalog.path}')">Usar</button>
                    `;
                    contentDiv.appendChild(item);
                });
            } else {
                contentDiv.innerHTML = '<div class="no-catalogs">No se encontraron catálogos. Verifica que existan carpetas con archivos *_tipo*.txt</div>';
            }
        } catch (error) {
            contentDiv.innerHTML = `<div class="error-hint">❌ Error: ${error.message}</div>`;
        }
    } else {
        listDiv.style.display = 'none';
    }
}

function selectNNCatalog(path) {
    document.getElementById('nnCatalogPath').value = path;
    document.getElementById('nnCatalogsList').style.display = 'none';
}

// Selección de carpeta de imágenes para CNN 2D
function onImageDirSelected(input) {
    if (input.files && input.files.length > 0) {
        // webkitRelativePath es "carpeta/archivo.png" — extraer solo el primer segmento
        const firstPath = input.files[0].webkitRelativePath;
        const folderName = firstPath.split('/')[0];
        document.getElementById('cnn2dImageDir').value = folderName + '/';
    }
}

// Selección de archivo CSV de etiquetas para CNN 2D
function onLabelsCsvSelected(input) {
    if (input.files && input.files.length > 0) {
        document.getElementById('cnn2dLabelsCSV').value = input.files[0].name;
    }
}


// ===========================
// ÁRBOL DE DECISIÓN INTERACTIVO
// ===========================

// ---------- DATOS: NODOS DEL ÁRBOL ----------
// Cada nodo: { id, bloque, paso, pregunta, hint, lineas, opciones[] }
// lineas: array de { lambda, label, color }
// opciones: { texto, icono, clase, detalle, siguiente (id|'resultado'), resultado? }

const ARBOL_NODOS = {

    // ─── PASO 1: Clasificación gruesa ───
    'inicio': {
        bloque: 'Clasificación Gruesa',
        paso: 'Paso 1 / ≈6 preguntas · 26 nodos',
        pregunta: '¿El espectro tiene pocas líneas intensas y el continuo es casi plano o azulado?',
        hint: 'Observa toda la región 3900–5000 Å. Pocas líneas anchas y continuo que sube hacia el azul sugieren estrella temprana (O, B, A).',
        ayuda: {
            titulo: 'Complejidad del espectro',
            contenido: '<p><b>Pocas líneas fuertes</b> → O, B, A (T ≥ 10 000 K). El continuo de Planck domina; solo transiciones ionizadas producen picos contrastados.</p><p><b>Balmer domina</b> → A–F. El H alcanza máximo en ~8 500 K por la ecuación de Boltzmann + Saha (nivel n=2).</p><p><b>Muchas líneas + continuo modulado</b> → K, M. Moléculas (TiO, CaH) y metales neutros en atmósferas frías.</p>'
        },
        lineas: [
            { lambda: 3934, label: 'Ca II K', color: '#f39c12' },
            { lambda: 4101, label: 'Hδ',      color: '#fbbf24' },
            { lambda: 4340, label: 'Hγ',      color: '#fbbf24' },
            { lambda: 4861, label: 'Hβ',      color: '#fbbf24' },
            { lambda: 6563, label: 'Hα',      color: '#e74c3c' },
        ],
        opciones: [
            { texto: 'Sí — pocas líneas, continuo plano/azul', icono: '✅', clase: 'opcion-si',
              detalle: 'Tipo temprano (O, B, A)', siguiente: 'oba' },
            { texto: 'Balmer domina (Hα, Hβ, Hγ, Hδ muy fuertes)', icono: '🔵', clase: 'opcion-comp',
              detalle: 'Tipo intermedio (A, F)', siguiente: 'intermedias' },
            { texto: 'Muchas líneas + continuo no plano', icono: '🟠', clase: 'opcion-no',
              detalle: 'Tipo tardío (F, G, K, M)', siguiente: 'tardias' },
        ]
    },

    // ─── BLOQUE OBA ───
    'oba': {
        bloque: 'Bloque OBA',
        paso: 'Paso 2',
        pregunta: '¿Se observan líneas de He I en 4471, 4026, 4922 y 5016 Å?',
        hint: 'He I aparece como líneas de absorción estrechas en estrellas O y B. Si están ausentes o muy débiles, es tipo A.',
        ayuda: {
            titulo: 'Líneas de He I neutro',
            contenido: '<p>He I (4026, 4471, 4922 Å) aparece en T &gt; 10 000 K.</p><p><b>Fundamento:</b> Helio neutro requiere mucha energía para excitar transiciones; su presencia indica atmósfera muy caliente (O/B).</p><ul><li><b>He I visible:</b> Tipo O o B</li><li><b>He I ausente o débil:</b> Tipo A</li></ul>'
        },
        lineas: [
            { lambda: 4026, label: 'He I 4026', color: '#9b59b6' },
            { lambda: 4471, label: 'He I 4471', color: '#9b59b6' },
            { lambda: 4861, label: 'Hβ',         color: '#fbbf24' },
            { lambda: 4922, label: 'He I 4922', color: '#9b59b6' },
            { lambda: 5016, label: 'He I 5016', color: '#9b59b6' },
        ],
        opciones: [
            { texto: 'Sí — He I claramente visible', icono: '✅', clase: 'opcion-si',
              detalle: 'Tipo O o B', siguiente: 'ob' },
            { texto: 'No — He I ausente o muy débil', icono: '❌', clase: 'opcion-no',
              detalle: 'Tipo A', siguiente: 'tipoA' },
        ]
    },

    // ─── BLOQUE OB ───
    'ob': {
        bloque: 'Bloque OB',
        paso: 'Paso 3',
        pregunta: '¿Se observan líneas de He II en 4200, 4542 y 4686 Å?',
        hint: 'He II requiere temperatura >25 000 K para ionizarse. Solo aparece en estrellas tipo O. En B, He II está ausente.',
        ayuda: {
            titulo: 'He II ionizado — diagnóstico O vs B',
            contenido: '<p>He II (4200, 4542, 4686 Å) solo aparece a T &gt; 30 000 K. Su presencia define inequívocamente el tipo O.</p><ul><li><b>He II claro:</b> Tipo O (&gt;30 000 K)</li><li><b>He II ausente:</b> Tipo B (10 000–30 000 K)</li></ul>'
        },
        lineas: [
            { lambda: 4200, label: 'He II 4200', color: '#e74c3c' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4542, label: 'He II 4542', color: '#e74c3c' },
            { lambda: 4686, label: 'He II 4686', color: '#e74c3c' },
        ],
        opciones: [
            { texto: 'Sí — He II presente', icono: '✅', clase: 'opcion-si',
              detalle: 'Tipo O', siguiente: 'tipoO_sub' },
            { texto: 'No — He II ausente', icono: '❌', clase: 'opcion-no',
              detalle: 'Tipo B', siguiente: 'tipoB_ini' },
        ]
    },

    // ─── BLOQUE O: subtipo ───
    'tipoO_sub': {
        bloque: 'Tipo O — Subtipo',
        paso: 'Paso 4',
        pregunta: 'Compara la profundidad (absorción) de He I 4471 Å vs He II 4542 Å. ¿Cuál es más profunda?',
        hint: '📐 Truco visual: mide cuánto baja cada línea desde el continuo (=1). He II 4542 es el ref. O3–O6: He II claramente más profunda que He I. O7–O9: ambas casi iguales. En O4, He I es ≈ la mitad de He II.',
        ayuda: {
            titulo: 'He I 4471 vs He II 4542 — hoja de ruta de subtipos O',
            contenido: `<p>Este es el criterio MK fundamental para tipos O. Mide la <b>profundidad relativa</b> (cuánto baja cada línea desde el continuo).</p>
<table style="width:100%;border-collapse:collapse;font-size:0.9em">
<tr style="background:#1e293b"><th style="padding:4px 8px;text-align:left">Subtipo</th><th style="padding:4px 8px">He I/He II</th><th style="padding:4px 8px">T_eff</th><th style="padding:4px 8px;text-align:left">Descripción</th></tr>
<tr><td style="padding:4px 8px"><b>O3</b></td><td style="padding:4px 8px;text-align:center">&lt; 0.3</td><td style="padding:4px 8px;text-align:center">≥ 45 000 K</td><td style="padding:4px 8px">He I casi invisible. N V + O V presentes</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O4</b></td><td style="padding:4px 8px;text-align:center">0.3–0.5</td><td style="padding:4px 8px;text-align:center">~42 000 K</td><td style="padding:4px 8px">He I débil (≈ mitad He II). N V visible moderado</td></tr>
<tr><td style="padding:4px 8px"><b>O5</b></td><td style="padding:4px 8px;text-align:center">0.5–0.7</td><td style="padding:4px 8px;text-align:center">~38 000 K</td><td style="padding:4px 8px">He I débil pero claramente visible. Sin N V</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O6</b></td><td style="padding:4px 8px;text-align:center">0.7–0.85</td><td style="padding:4px 8px;text-align:center">~35 000 K</td><td style="padding:4px 8px">He I algo débil aún. Sin N V</td></tr>
<tr><td style="padding:4px 8px"><b>O7–O9</b></td><td style="padding:4px 8px;text-align:center">0.85–1.3</td><td style="padding:4px 8px;text-align:center">31–34 000 K</td><td style="padding:4px 8px">He I ≈ He II. N III como diagnóstico</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O9.5</b></td><td style="padding:4px 8px;text-align:center">&gt; 1.3</td><td style="padding:4px 8px;text-align:center">~31 000 K</td><td style="padding:4px 8px">He I supera levemente He II. Límite O–B</td></tr>
</table>`
        },
        lineas: [
            { lambda: 4200, label: 'He II 4200', color: '#e74c3c' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4542, label: 'He II 4542', color: '#e74c3c' },
            { lambda: 4604, label: 'N V 4604',   color: '#1abc9c' },
            { lambda: 4686, label: 'He II 4686', color: '#e74c3c' },
            { lambda: 4713, label: 'He I 4713',  color: '#7c3aed' },
        ],
        opciones: [
            { texto: 'He I ≪ He II — He I claramente más débil (profundidad < mitad de He II)',
              icono: '⬅️', clase: 'opcion-menor',
              detalle: 'O temprana: O3 – O6 (T > 35 000 K)', siguiente: 'tipoO_early' },
            { texto: 'He I ≈ He II — ambas líneas de profundidad similar (±20%)',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'O tardía: O7 – O9', siguiente: 'tipoO_late' },
            { texto: 'He I ≥ He II — He I tan profunda o levemente mayor',
              icono: '➡️', clase: 'opcion-mayor',
              detalle: 'O9 / O9.5 (límite hacia B)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O9–O9.5', criterio: 'He I 4471 ≥ He II 4542. O muy tardía, transición hacia B0. ~31 000–32 000 K.' }},
        ]
    },

    // ─── BLOQUE B inicial ───
    'tipoB_ini': {
        bloque: 'Tipo B — Clasificación',
        paso: 'Paso 4',
        pregunta: 'Compara He I 4471 Å vs Mg II 4481 Å. ¿Cuál predomina?',
        hint: 'Mg II 4481 Å aparece progresivamente a medida que baja la temperatura. En B tempranas HeI domina; en B tardías MgII se iguala o supera.',
        ayuda: {
            titulo: 'He I 4471 vs Mg II 4481 — temperatura en B',
            contenido: '<p>Mg II 4481 Å crece al bajar T porque los metales se pueden excitar a temperaturas menores.</p><ul><li><b>He I ≫ Mg II:</b> B temprana B0–B5 (&gt;14 000 K)</li><li><b>Mg II ≥ He I:</b> B tardía B6–B9 (10 000–14 000 K)</li></ul>'
        },
        lineas: [
            { lambda: 4471, label: 'He I 4471', color: '#9b59b6' },
            { lambda: 4481, label: 'Mg II 4481',color: '#27ae60' },
            { lambda: 4861, label: 'Hβ',         color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'He I ≫ Mg II (Mg II casi imperceptible)', icono: '💜', clase: 'opcion-mayor',
              detalle: 'B temprana (B0–B5)', siguiente: 'tipoB_temprana' },
            { texto: 'Mg II ≥ He I (Mg II comparable o mayor)', icono: '💚', clase: 'opcion-comp',
              detalle: 'B tardía (B6–B9)', siguiente: 'tipoB_tardia' },
        ]
    },

    // ─── BLOQUE B temprana ───
    'tipoB_temprana': {
        bloque: 'B Temprana — Silicio',
        paso: 'Paso 5',
        pregunta: 'Compara las líneas de silicio: Si IV 4089 Å, Si III 4553 Å y Si II 4128-4130 Å.',
        hint: 'La secuencia de ionización del Si es el criterio clave para B0–B5. Observa qué ión domina.',
        ayuda: {
            titulo: 'Ionización del Si en B temprana',
            contenido: '<p>La secuencia Si IV → Si III → Si II ocurre al bajar T y es el criterio diagnóstico clave para B0–B5.</p><ul><li><b>Si IV 4089 dom.:</b> B0 (~30 000 K)</li><li><b>Si III 4553 dom.:</b> B1–B3 (~18 000–25 000 K)</li><li><b>Si II 4128 dom.:</b> B4–B5 (~14 000 K)</li></ul>'
        },
        lineas: [
            { lambda: 4070, label: 'O II 4070',  color: '#0ea5e9' },
            { lambda: 4076, label: 'O II 4076',  color: '#0ea5e9' },
            { lambda: 4089, label: 'Si IV 4089', color: '#e74c3c' },
            { lambda: 4116, label: 'Si IV 4116', color: '#e74c3c' },
            { lambda: 4121, label: 'He I 4121',  color: '#7c3aed' },
            { lambda: 4128, label: 'Si II 4128', color: '#f39c12' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4553, label: 'Si III 4553',color: '#a3e635' },
        ],
        opciones: [
            { texto: 'Si IV ≈ Si III (ambos comparables)',     icono: '⚖️', clase: 'opcion-comp',
              detalle: 'B0',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B0', criterio: 'Si IV 4089 ≈ Si III 4553. Límite O9–B0, temperatura ~30 000 K.' }},
            { texto: 'Si III > Si IV y Si II (Si III domina)', icono: '🟢', clase: 'opcion-mayor',
              detalle: 'B1 – B3', siguiente: 'tipoB_OII' },
            { texto: 'Si III ≥ Si II (comenzando a igualarse)',icono: '🔶', clase: 'opcion-comp',
              detalle: 'B3',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B3', criterio: 'Si III ≈ Si II. Temperatura ~18 000 K, transición de ionización del Si.' }},
            { texto: 'Si III ≪ Si II (Si II claramente mayor)', icono: '🟠', clase: 'opcion-mmay',
              detalle: 'B4 – B5',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B4–B5', criterio: 'Si II 4128–4130 domina sobre Si III. ~14 000–17 000 K.' }},
        ]
    },

    // ─── BLOQUE B tardía ───
    'tipoB_tardia': {
        bloque: 'B Tardía — Magnesio',
        paso: 'Paso 5',
        pregunta: 'Compara la intensidad de Mg II 4481 Å respecto a He I 4471 Å.',
        hint: 'En B tardías la razón Mg II / He I aumenta sistemáticamente. Mide la profundidad de ambas líneas.',
        ayuda: {
            titulo: 'Mg II / He I en B tardía',
            contenido: '<p>En B tardía (B6–B9) la temperatura desciende y los metales neutros crecen progresivamente.</p><ul><li><b>Mg II ≈ He I:</b> B6–B7 (~13 000 K)</li><li><b>Mg II 2–3× He I:</b> B8 (~11 000 K, ej. Rigel B8Ia)</li><li><b>He I casi ausente:</b> B9–A0 (frontera B–A)</li></ul>'
        },
        lineas: [
            { lambda: 4471, label: 'He I 4471', color: '#9b59b6' },
            { lambda: 4481, label: 'Mg II 4481',color: '#27ae60' },
            { lambda: 4553, label: 'Si III 4553',color: '#a3e635' },
        ],
        opciones: [
            { texto: 'Mg II ≈ He I (intensidades similares)',   icono: '⚖️', clase: 'opcion-comp',
              detalle: 'B6 – B7',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B6–B7', criterio: 'Mg II 4481 ≈ He I 4471. ~12 000–14 000 K.' }},
            { texto: 'Mg II 2–3× más intenso que He I',         icono: '📈', clase: 'opcion-mayor',
              detalle: 'B8',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B8', criterio: 'Mg II ≈ 2–3× He I. ~11 000 K. Rigel es un ejemplo (B8Ia).' }},
            { texto: 'Mg II ≫ He I (He I muy débil o ausente)', icono: '⬆️', clase: 'opcion-mmay',
              detalle: 'B9 – B9.5', siguiente: 'tipoB95' },
        ]
    },

    // ─── BLOQUE Intermedias (Balmer domina) → A/F ───
    'intermedias': {
        bloque: 'Tipos A / F',
        paso: 'Paso 2',
        pregunta: 'Compara Ca II K (3934 Å) con H I 3968 Å (línea de Balmer más cercana).',
        hint: 'Ca II K crece fuertemente de A a F. En A tempranas es débil frente a Balmer; en F es ya comparable o mayor.',
        ayuda: {
            titulo: 'Ca II K vs Balmer — tipos A y F',
            contenido: `
<p><b>¿Por qué la línea "K"?</b><br>
El calcio ionizado (Ca II) produce un <em>doblete</em> muy conocido:
la línea <b>K</b> a 3933.7 Å y la línea <b>H</b> a 3968.5 Å.
La nomenclatura histórica (letras A–K asignadas por Fraunhofer) las bautizó así antes de que se conociera su origen atómico.</p>

<p><b>¿Por qué se usa K y no H?</b><br>
La línea H del calcio (3968.5 Å) se superpone casi exactamente con
<b>H&epsilon;</b> del hidrógeno (3970 Å). Esta mezcla dificulta medir
el calcio puro. La línea K, en cambio, está despejada y es el
<em>indicador limpio</em> del Ca II.</p>

<p><b>Ca II K como termómetro estelar:</b></p>
<ul>
  <li><b>A0–A2</b> (~10 000 K): Ca II K casi invisible — Balmer en su máximo absoluto (ej. Sirio A1V)</li>
  <li><b>A5–A9</b> (~8 000 K): Ca II K visible, ≈ ¼–½ de Hδ</li>
  <li><b>F0</b> (~7 500 K): Ca II K ≈ Balmer — punto de cruce</li>
  <li><b>F5+</b> (&lt;6 500 K): Ca II K domina sobre Balmer</li>
</ul>

<p><em>Si Ca II K está fuera del rango del espectro cargado, se mostrará
como línea discontinua gris con la etiqueta "fuera de rango" y
se usará el criterio alternativo (Balmer + Mg II + Fe I).</em></p>`
        },
        lineas: [
            { lambda: 3934, label: 'Ca II K', color: '#f39c12' },
            { lambda: 3968, label: 'Ca II H / Hε', color: '#e67e22' },
            { lambda: 4101, label: 'Hδ',       color: '#fbbf24' },
            { lambda: 4340, label: 'Hγ',       color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Ca II K ≪ H I (K apenas visible)',         icono: '⬅️', clase: 'opcion-menor',
              detalle: 'A temprana (A0–A3)', siguiente: 'tipoA' },
            { texto: 'Ca II K ≈ ½ H I (K moderada)',             icono: '⚖️', clase: 'opcion-comp',
              detalle: 'A intermedia (A5–A9)', siguiente: 'tipoA_media' },
            { texto: 'Ca II K ≈ H I (K tan fuerte como Balmer)', icono: '➡️', clase: 'opcion-mayor',
              detalle: 'F0', siguiente: 'tipoF' },
        ]
    },

    // ─── BLOQUE Intermedias SIN cobertura azul (λ < 4000 Å) ───
    'intermedias_alt': {
        bloque: 'Tipos A / F — criterio alternativo',
        paso: 'Paso 2 (sin Ca II K)',
        pregunta: 'Tu espectro no cubre Ca II K / Hε. Compara la intensidad de Hδ (4102 Å) con Mg II 4481 Å y Fe I 4383 Å.',
        hint: 'En A, Balmer (Hδ, Hγ, Hβ) domina claramente y los metales son muy débiles. En F los metales se vuelven comparables. Mg II 4481 aparece en A tardía.',
        ayuda: {
            titulo: 'Criterio alternativo A/F — Balmer + Mg II + Fe I',
            contenido: `
<p>Cuando el espectro no cubre la región del Ca II K (3934 Å) se usan indicadores dentro del rango 4000–5000 Å:</p>
<ul>
  <li><b>Hδ (4102 Å) / Hγ (4340 Å) / Hβ (4861 Å)</b> — La serie de Balmer es el termómetro principal. Máxima en A2, decrece hacia G.</li>
  <li><b>Mg II 4481 Å</b> — Prácticamente ausente en A0–A3, visible en A5+, moderado en F. Es el mejor sustituto del Ca II K para estrellas A tardías.</li>
  <li><b>Fe I 4383 Å</b> — Aparece en F y crece hacia G/K. Junto con Mg II ayuda a separar F de A.</li>
</ul>
<p><b>Regla rápida:</b> Si Hδ es claramente más alto que Mg II → tipo A. Si son comparables → A9/F0. Si Mg II y Fe I son visibles al mismo nivel que Balmer → tipo F.</p>
<p><em>Confianza reducida (~75%) respecto al criterio canónico con Ca II K.</em></p>`
        },
        lineas: [
            { lambda: 4101, label: 'Hδ',    color: '#fbbf24' },
            { lambda: 4340, label: 'Hγ',    color: '#fbbf24' },
            { lambda: 4383, label: 'Fe I',  color: '#94a3b8' },
            { lambda: 4481, label: 'Mg II', color: '#60a5fa' },
            { lambda: 4861, label: 'Hβ',    color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Hδ/Hγ/Hβ muy intensas, Mg II y Fe I apenas visibles o ausentes',
              icono: '⬅️', clase: 'opcion-menor',
              detalle: 'A tardía (A5–A9)', siguiente: 'tipoA_media' },
            { texto: 'Balmer moderado, Mg II visible (~¼ de Hδ), Fe I débil',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'Límite A9 / F0–F2', siguiente: 'tipoF' },
            { texto: 'Balmer moderado-débil, Mg II y Fe I claramente visibles',
              icono: '➡️', clase: 'opcion-mayor',
              detalle: 'F (F2–F8)', siguiente: 'tipoF' },
        ]
    },

    // ─── TIPO A temprana/inicial ───
    'tipoA': {
        bloque: 'Tipo A',
        paso: 'Paso 3',
        pregunta: '¿Cuánto mide la línea Ca II K (3934 Å) respecto a Balmer (Hδ 4101, Hγ 4340)?',
        hint: 'En A0 Ca II es casi invisible. En A5 es ~½ de Hδ. Balmer es máxima en A2 y decrece hacia F.',
        ayuda: {
            titulo: 'Ca II K en subtipos A',
            contenido: '<p>El Ca II K crece de A0 a A9 mientras Balmer decrece. A2 es el máximo absoluto de la serie Balmer.</p><ul><li><b>Ca II mínimo, Balmer máximo:</b> A0–A2 (ej. Sirio A1V, ~10 000 K)</li><li><b>Ca II débil visible (~¼ Hδ):</b> A3–A5 (~8 500 K)</li></ul>'
        },
        lineas: [
            { lambda: 3934, label: 'Ca II K', color: '#f39c12' },
            { lambda: 4101, label: 'Hδ',       color: '#fbbf24' },
            { lambda: 4340, label: 'Hγ',       color: '#fbbf24' },
            { lambda: 4861, label: 'Hβ',       color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Ca II K casi ausente, Balmer máximo',  icono: '🔵', clase: 'opcion-menor',
              detalle: 'A0 – A2',
              siguiente: 'resultado',
              resultado: { tipo: 'A', subtipo: 'A0–A2', criterio: 'Balmer en máximo, Ca II K despreciable. Sirio (A1V) es el ejemplo clásico. ~9 000–10 000 K.' }},
            { texto: 'Ca II K débil (~¼ de Hδ)',             icono: '🔹', clase: 'opcion-comp',
              detalle: 'A3 – A5',
              siguiente: 'resultado',
              resultado: { tipo: 'A', subtipo: 'A3–A5', criterio: 'Balmer aún fuerte, Ca II K visible pero débil. ~7 500–9 000 K.' }},
        ]
    },

    // ─── TIPO A temprana SIN cobertura azul ───
    'tipoA_alt': {
        bloque: 'Tipo A — criterio alternativo',
        paso: 'Paso 3 (sin Ca II K)',
        pregunta: 'Tu espectro no cubre Ca II K. Evalúa la anchura y profundidad de Hδ (4101 Å) y Hγ (4340 Å). ¿Hay alguna traza de Mg II 4481 Å?',
        hint: 'En A0–A2 Hδ/Hγ son extremadamente anchas y profundas, y Mg II está ausente. En A3–A5 siguen siendo muy fuertes pero Mg II empieza a ser perceptible.',
        ayuda: {
            titulo: 'Subtipos A sin Ca II K — Balmer y Mg II 4481',
            contenido: `
<p>Sin acceso a Ca II K (3934 Å), el criterio de subtipo A usa la anchura de las líneas de Balmer y la aparición de Mg II 4481 Å:</p>
<ul>
  <li><b>A0–A2</b> (~10 000 K): Hδ y Hγ en su <em>anchura máxima</em> (muy largas y profundas). Mg II 4481 <b>ausente</b> por completo.</li>
  <li><b>A3–A5</b> (~8 500–9 000 K): Hδ/Hγ siguen siendo muy fuertes pero algo menos anchas. Mg II 4481 puede aparecer como una traza muy débil.</li>
</ul>
<p><em>Confianza reducida (~70%) sin Ca II K como referencia.</em></p>`
        },
        lineas: [
            { lambda: 4101, label: 'Hδ',    color: '#fbbf24' },
            { lambda: 4340, label: 'Hγ',    color: '#fbbf24' },
            { lambda: 4481, label: 'Mg II', color: '#60a5fa' },
            { lambda: 4861, label: 'Hβ',    color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Hδ/Hγ extremadamente anchas y profundas, Mg II ausente',
              icono: '🔵', clase: 'opcion-menor',
              detalle: 'A0 – A2',
              siguiente: 'resultado',
              resultado: { tipo: 'A', subtipo: 'A0–A2', criterio: 'Balmer en máximo absoluto, Mg II indetectable. ~9 000–10 000 K. (Criterio alternativo sin Ca II K)' }},
            { texto: 'Hδ/Hγ muy fuertes pero algo menos anchas, Mg II apenas perceptible',
              icono: '🔹', clase: 'opcion-comp',
              detalle: 'A3 – A5',
              siguiente: 'resultado',
              resultado: { tipo: 'A', subtipo: 'A3–A5', criterio: 'Balmer aún dominante, Mg II tenue emergente. ~8 500–9 000 K. (Criterio alternativo sin Ca II K)' }},
        ]
    },

    // ─── TIPO A media ───
    'tipoA_media': {
        bloque: 'Tipo A Media',
        paso: 'Paso 3',
        pregunta: '¿Hay líneas metálicas tenues visibles (Fe I 4383, Ca I 4226, Mg I)?',
        hint: 'En A5–A7 los metales están aún ausentes o son invisibles. En A8–A9 Fe I y Ca I empiezan a aparecer como trazas débiles. Ca II K puede o no estar en rango.',
        ayuda: {
            titulo: 'Aparición de metales en A5–A9',
            contenido: '<p>En A5–A9 comienzan las primeras trazas de Fe I, Ca I junto a un Ca II ya notable (si el espectro lo cubre). Indica descenso de temperatura bajo ~8 500 K.</p><ul><li><b>Sin metales (solo Balmer dominante):</b> A5–A7 (~8 000 K)</li><li><b>Fe I / Ca I tenues emergentes:</b> A8–F0 (~7 500 K)</li></ul>'
        },
        lineas: [
            { lambda: 3934, label: 'Ca II K',   color: '#f39c12' },
            { lambda: 4226, label: 'Ca I 4226', color: '#e67e22' },
            { lambda: 4340, label: 'Hγ',         color: '#fbbf24' },
            { lambda: 4383, label: 'Fe I 4383', color: '#95a5a6' },
            { lambda: 4861, label: 'Hβ',         color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Sin metales visibles — solo Ca II K (si visible) o Balmer puro',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'A5 – A7',
              siguiente: 'resultado',
              resultado: { tipo: 'A', subtipo: 'A5–A7', criterio: 'Ca II K ≈ ½ Hδ, Fe I / Ca I aún ausentes. ~8 000 K.' }},
            { texto: 'Metales tenues visibles (Fe I, Ca I) — con o sin Ca II K',
              icono: '🔶', clase: 'opcion-mayor',
              detalle: 'A8 – A9 / F0',
              siguiente: 'tipoF' },
        ]
    },

    // ─── BLOQUE F ───
    'tipoF': {
        bloque: 'Tipo F',
        paso: 'Paso 3',
        pregunta: 'Compara Hδ (4101) y Hγ (4340) con Fe I 4046 Å y Ca I 4226 Å.',
        hint: 'En F0–F4 el Balmer domina claramente. En F5+ Ca I 4226 se vuelve la línea más prominente entre Hδ y Hγ. En F9 los metales igualan al Balmer.',
        ayuda: {
            titulo: 'Subtipo F — Balmer vs Ca I y Fe I',
            contenido: `<p>La clasificación F usa tres etapas:</p>
<ul>
  <li><b>F0–F4</b> (~7 000–7 500 K): Hδ y Hγ claramente más intensas que Fe I 4046 y Ca I 4226. Ca II K muy fuerte pero no indica temperatura pasado F3.</li>
  <li><b>F5–F9</b> (~6 000–6 500 K): Ca I 4226 se convierte en la línea más prominente entre Hδ y Hγ. Ratios clave: <em>Fe I 4046 / Hδ</em> y <em>Ca I 4226 / Hδ</em> aumentan con la temperatura.</li>
  <li><b>F9–G0</b> (~5 800 K): Fe I 4383 y Ca I 4226 igualan o superan a Hγ; G-band CH 4300 empieza a aparecer.</li>
</ul>
<p><b>Nota metalicidad:</b> estos ratios pueden ser engañosos en estrellas pobres o ricas en metales.</p>`
        },
        lineas: [
            { lambda: 4030, label: 'Mn I 4030',  color: '#64748b' },
            { lambda: 4046, label: 'Fe I 4046',  color: '#95a5a6' },
            { lambda: 4101, label: 'Hδ',          color: '#fbbf24' },
            { lambda: 4226, label: 'Ca I 4226',  color: '#e67e22' },
            { lambda: 4271, label: 'Fe I 4271',  color: '#94a3b8' },
            { lambda: 4340, label: 'Hγ',          color: '#fbbf24' },
            { lambda: 4383, label: 'Fe I 4383',  color: '#95a5a6' },
        ],
        opciones: [
            { texto: 'Hδ/Hγ claramente más intensas que Fe I 4046 y Ca I 4226',
              icono: '🔵', clase: 'opcion-mayor',
              detalle: 'F0 – F4 (~7 000–7 500 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F0–F4', criterio: 'Balmer > metales. Ca II K fuerte, Ca I 4226 y Fe I 4046 visibles pero menores que Hδ. ~7 000–7 500 K.' }},
            { texto: 'Ca I 4226 tan fuerte o más que Hδ; Fe I 4046 notable',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'F5 – F9 → afinar subtipo', siguiente: 'tipoF_sub' },
            { texto: 'Metálicas equiparables o superiores al Balmer; G-band incipiente',
              icono: '🟠', clase: 'opcion-mmay',
              detalle: 'F9 – G0', siguiente: 'tardias' },
        ]
    },

    // ─── BLOQUE F tardía (F5–F9) ───
    'tipoF_sub': {
        bloque: 'Tipo F Tardía — F5 a F9',
        paso: 'Paso 4',
        pregunta: 'Compara Ca I 4226 con Hδ (4101). ¿Y está Sr II 4077 visible respecto a Fe I 4046?',
        hint: 'Ca I 4226 crece de F5 a F9. Sr II 4077 aparece en F6+ y es indicador de luminosidad. En F9 los metales casi igualan a Hβ.',
        ayuda: {
            titulo: 'F5–F9 — Ca I 4226, Sr II 4077 y Fe I',
            contenido: `<p>Para afinar dentro de F tardía, el documento MK recomienda:</p>
<ul>
  <li><b>Temperatura</b> — Ratios <em>Ca I 4226 / Hδ</em> y <em>Fe I 4046 / Hδ</em>:
    <ul>
      <li>F5–F6: Ca I 4226 ≈ ½ Hδ, Fe I 4046 débil</li>
      <li>F7–F8: Ca I 4226 ≈ Hδ, Fe I 4383 ≈ ½ Hγ</li>
      <li>F9: Ca I 4226 > Hδ, Fe I notable, G-band CH 4300 insinuada</li>
    </ul>
  </li>
  <li><b>Luminosidad</b> — <em>Sr II 4077 / Fe I 4046</em>: el estroncio ionizado crece en gigantes y supergigantes. Si Sr II 4077 es claramente más intenso que Fe I 4046, la estrella es muy probablemente gigante (III) o supergigante (I).</li>
  <li><b>Mn I 4030</b>: blend ancho de Mn I presente en F. Aumenta con la temperatura hacia G.</li>
</ul>`
        },
        lineas: [
            { lambda: 4030, label: 'Mn I 4030',  color: '#64748b' },
            { lambda: 4046, label: 'Fe I 4046',  color: '#95a5a6' },
            { lambda: 4077, label: 'Sr II 4077', color: '#a78bfa' },
            { lambda: 4101, label: 'Hδ',          color: '#fbbf24' },
            { lambda: 4226, label: 'Ca I 4226',  color: '#e67e22' },
            { lambda: 4300, label: 'G-band CH',  color: '#fb923c' },
            { lambda: 4383, label: 'Fe I 4383',  color: '#95a5a6' },
        ],
        opciones: [
            { texto: 'Ca I 4226 ≈ ½ Hδ, Fe I 4046 débil, sin G-band',
              icono: '🔵', clase: 'opcion-menor',
              detalle: 'F5 – F6',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F5–F6', criterio: 'Ca I 4226 ≈ ½ Hδ. Fe I 4046 y Fe I 4383 visibles pero débiles. Sr II puede ser indicador de luminosidad. ~6 300–6 500 K.' }},
            { texto: 'Ca I 4226 ≈ Hδ, Fe I 4383 notable, G-band débil o ausente',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'F7 – F8',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F7–F8', criterio: 'Ca I 4226 ≈ Hδ. Fe I 4383 ≈ ½ Hγ. G-band CH 4300 puede ser débil. ~6 000–6 200 K.' }},
            { texto: 'Ca I 4226 > Hδ, Fe I notable, G-band CH 4300 insinuada',
              icono: '🟠', clase: 'opcion-mmay',
              detalle: 'F9',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F9', criterio: 'Ca I 4226 > Hδ. Fe I 4383 ≈ ½ Hγ. G-band CH 4300 empezando. Transición F9/G0. ~5 900 K.' }},
        ]
    },

    // ─── BLOQUE Tardías (GKM) ───
    'tardias': {
        bloque: 'Tipos G / K / M',
        paso: 'Paso 2',
        pregunta: 'Compara Hδ 4101 Å con Ca I 4226 Å, Fe I 4144 Å y Fe I/Cr I 4260 Å.',
        hint: 'G0: Hδ > Ca I 4226 ≈ Fe I 4046, Fe I 4144 débil. G5: Hδ ≈ Ca I 4226, Fe I 4144 crece. K: Hδ débil, metales dominan. TiO → M.',
        ayuda: {
            titulo: 'Clasificación G / K / M: Balmer vs Metales',
            contenido: '<p>En tipos tardíos el Balmer se debilita progresivamente y los metales dominan. Fe I 4144 es clave para separar G0 de G5:</p><ul><li><b>Hδ &gt; Ca I 4226; Fe I 4144 débil:</b> G0–G2 (~5 800–6 000 K)</li><li><b>Hδ ≈ Ca I 4226; Fe I 4144 ≈ ½ Ca I:</b> G5 (~5 500 K)</li><li><b>Hδ &lt; Ca I 4226; Fe I 4144 y Cr I fuertes:</b> K0+ (&lt;5 200 K)</li><li><b>Bandas TiO visibles:</b> M (&lt;3 700 K)</li></ul>'
        },
        lineas: [
            { lambda: 4046, label: 'Fe I 4046',  color: '#94a3b8' },
            { lambda: 4101, label: 'Hδ',          color: '#fbbf24' },
            { lambda: 4144, label: 'Fe I 4144',  color: '#94a3b8' },
            { lambda: 4226, label: 'Ca I 4226',  color: '#e67e22' },
            { lambda: 4254, label: 'Cr I 4254',  color: '#e74c3c' },
            { lambda: 4260, label: 'Fe I 4260',  color: '#95a5a6' },
            { lambda: 4308, label: 'G-band',     color: '#f39c12' },
        ],
        opciones: [
            { texto: 'Hδ > Ca I 4226; Fe I 4144 débil (Balmer domina)',          icono: '🟡', clase: 'opcion-mayor',
              detalle: 'G0 – G2', siguiente: 'tipoG' },
            { texto: 'Hδ ≈ Ca I 4226; Fe I 4144 ≈ ½ Ca I; Cr I tenue',         icono: '⚖️', clase: 'opcion-comp',
              detalle: 'G5', siguiente: 'tipoG5' },
            { texto: 'Hδ < Ca I 4226; Fe I 4144 y Cr I claramente dominan',     icono: '🟠', clase: 'opcion-menor',
              detalle: 'K0 o más tardío', siguiente: 'tipoK' },
        ]
    },

    // ─── TIPO G ───
    'tipoG': {
        bloque: 'Tipo G — Subtipo',
        paso: 'Paso 3',
        pregunta: 'Compara Hδ 4101 con Fe I 4144 Å; y Cr I 4254 con Fe I 4260 Å. ¿Está Y II 4376 visible?',
        hint: 'G0: Hδ > Fe I 4144, Fe I 4260 > Cr I 4254. G2: Hδ ≈ Fe I 4144. G5: Hδ < Fe I 4144, Cr I≈Fe I. Y II 4376 ≥ Fe I 4383 → gigante.',
        ayuda: {
            titulo: 'Fe I 4144, Cr I 4254 y Y II 4376 — subtipos G',
            contenido: '<p>Dos criterios de temperatura independientes para subtipos G:</p><ul><li><b>Hδ vs Fe I 4144:</b> G0: Hδ &gt; Fe I 4144; G2: Hδ ≈ Fe I 4144; G5: Hδ &lt; Fe I 4144.</li><li><b>Cr I 4254 vs Fe I 4260:</b> Fe I &gt; Cr I en G0–G2; Cr I ≈ Fe I en G5.</li></ul><p><b>Luminosidad (Y II 4376 / Fe I 4383):</b> Si Y II 4376 ≥ Fe I 4383 → probable gigante o supergigante (clase III o superior). En enanas (V) Y II es débil.</p>'
        },
        lineas: [
            { lambda: 4101, label: 'Hδ',          color: '#fbbf24' },
            { lambda: 4144, label: 'Fe I 4144',  color: '#94a3b8' },
            { lambda: 4254, label: 'Cr I 4254',  color: '#e74c3c' },
            { lambda: 4260, label: 'Fe I 4260',  color: '#95a5a6' },
            { lambda: 4308, label: 'G-band',     color: '#f39c12' },
            { lambda: 4376, label: 'Y II 4376',  color: '#06b6d4' },
            { lambda: 4383, label: 'Fe I 4383',  color: '#95a5a6' },
            { lambda: 5173, label: 'Mg b',       color: '#27ae60' },
        ],
        opciones: [
            { texto: 'Hδ > Fe I 4144; Cr I 4254 < Fe I 4260 (Balmer aún domina)', icono: '⚙️', clase: 'opcion-menor',
              detalle: 'G0 – G2', siguiente: 'tipoG_Mg' },
            { texto: 'Hδ ≈ Fe I 4144; Cr I 4254 ≈ Fe I 4260',                     icono: '⚖️', clase: 'opcion-comp',
              detalle: 'G5',
              siguiente: 'resultado',
              resultado: { tipo: 'G', subtipo: 'G5', criterio: 'Hδ ≈ Fe I 4144, Cr I ≈ Fe I 4260, G-band fuerte. ~5 500 K. (Si Y II 4376 ≥ Fe I 4383 → gigante G5III.)' }},
        ]
    },

    // ─── TIPO G5 (intermedio) ───
    'tipoG5': {
        bloque: 'G5 / K0',
        paso: 'Paso 3',
        pregunta: 'Compara Cr I 4254 Å vs Fe I 4260 Å y la intensidad de Ca I 4226 Å.',
        hint: 'En G5–K0 el Ca I crece rápidamente. Si Cr I empieza a superar a Fe I, estamos en K.',
        ayuda: {
            titulo: 'Límite G5 / K0 — Ca I y Cr I',
            contenido: '<p>En G5–K0 el Ca I 4226 Å crece rápidamente. Si Cr I empieza a superar a Fe I, la temperatura ha bajado a ~5 200 K (K0).</p><ul><li><b>Fe I &gt; Cr I, Ca I moderado:</b> G5 (~5 500 K)</li><li><b>Cr I ≥ Fe I, Ca I fuerte:</b> K0 (~5 200 K)</li></ul>'
        },
        lineas: [
            { lambda: 4226, label: 'Ca I 4226', color: '#e67e22' },
            { lambda: 4254, label: 'Cr I 4254', color: '#e74c3c' },
            { lambda: 4260, label: 'Fe I 4260', color: '#95a5a6' },
            { lambda: 4308, label: 'G-band',    color: '#f39c12' },
        ],
        opciones: [
            { texto: 'Cr I < Fe I, Ca I moderado',  icono: '🟡', clase: 'opcion-menor',
              detalle: 'G5',
              siguiente: 'resultado',
              resultado: { tipo: 'G', subtipo: 'G5', criterio: 'G-band fuerte, Fe I domina sobre Cr I, Ca I moderado. ~5 500 K.' }},
            { texto: 'Cr I ≥ Fe I, Ca I fuerte',    icono: '🟠', clase: 'opcion-mayor',
              detalle: 'K0',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K0', criterio: 'Cr I ≥ Fe I, Ca I fuerte, Balmer débil. ~5 200 K. Transición G–K.' }},
        ]
    },

    // ─── TIPO K ───
    'tipoK': {
        bloque: 'Tipo K — Subtipo',
        paso: 'Paso 3',
        pregunta: 'Compara el triplete Cr I (4254, 4275, 4290 Å) con Fe I 4260 Å y Ca I 4226 Å.',
        hint: 'K0: Cr I 4254 ≈ Fe I 4260, Ca I fuerte. K3–K5: Cr I 4254 ≥ Cr I 4275, todos > Fe I. K7+: TiO emergente.',
        ayuda: {
            titulo: 'Triplete Cr I — temperatura en tipo K',
            contenido: '<p>El triplete Cr I (4254, 4275 y 4290 Å) es el criterio de temperatura más preciso en espectros K, más sensible que Ca I solo:</p><ul><li><b>Cr I 4254 ≈ Fe I 4260, Ca I fuerte:</b> K0–K2 (~4 800–5 200 K, ej. Arcturus K1.5III)</li><li><b>Cr I 4254 ≈ Cr I 4275, ambos &gt; Fe I:</b> K3–K5 (~4 000–4 500 K)</li><li><b>Cr I triplete todo &gt; Fe I, Ca I dominante:</b> K5 (~4 000 K)</li><li><b>TiO tenue emergente:</b> K7–M0</li></ul><p><b>Luminosidad:</b> Y II 4376 / Fe I 4383 ≥ 1 → gigante (III). Ba II 4554 fuerte → supergigante.</p>'
        },
        lineas: [
            { lambda: 4226, label: 'Ca I 4226',  color: '#e67e22' },
            { lambda: 4254, label: 'Cr I 4254',  color: '#f43f5e' },
            { lambda: 4260, label: 'Fe I 4260',  color: '#95a5a6' },
            { lambda: 4275, label: 'Cr I 4275',  color: '#f43f5e' },
            { lambda: 4290, label: 'Cr I 4290',  color: '#f43f5e' },
            { lambda: 4376, label: 'Y II 4376',  color: '#06b6d4' },
            { lambda: 4383, label: 'Fe I 4383',  color: '#95a5a6' },
            { lambda: 4761, label: 'TiO 4761',   color: '#c0392b' },
        ],
        opciones: [
            { texto: 'Cr I 4254 ≈ Fe I 4260; Ca I fuerte (sin TiO)',             icono: '🟠', clase: 'opcion-comp',
              detalle: 'K0 – K2', siguiente: 'tipoK_Na' },
            { texto: 'Cr I 4254 ≈ Cr I 4275, ambos > Fe I 4260; Ca I dominante', icono: '🔶', clase: 'opcion-comp',
              detalle: 'K3 – K5',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K3–K5', criterio: 'Cr I 4254 y 4275 comparables entre sí y ambos > Fe I. Ca I dominante. Balmer casi ausente. ~4 000–4 500 K.' }},
            { texto: 'Triplete Cr I todo muy fuerte (≥ 2× Fe I), Ca I ≫ Fe I',   icono: '🔴', clase: 'opcion-mayor',
              detalle: 'K5',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K5', criterio: 'Cr I triplete (4254/4275/4290) todos ≥ 2× Fe I 4260. Ca I dominante, sin TiO. ~4 000 K.' }},
            { texto: 'Ca I ≫ Fe I + TiO 4761 tenue emergente',                   icono: '🟥', clase: 'opcion-mmay',
              detalle: 'K7 – M0', siguiente: 'tipoKM' },
        ]
    },

    // ─── BLOQUE KM ───
    'tipoKM': {
        bloque: 'Límite K/M — TiO',
        paso: 'Paso 4',
        pregunta: 'Evalúa las bandas moleculares de TiO (4761, 4955, 5167 Å) y el aspecto de Fe I 4957 Å.',
        hint: 'La progresión TiO es el criterio definitivo K7→M. Bandas de TiO absorben franjas enteras del continuo.',
        ayuda: {
            titulo: 'Bandas moleculares de TiO — tipos K tardío y M',
            contenido: '<p>TiO solo existe en T &lt; 4 000 K. Sus bandas absorben franjas enteras del continuo y son la firma inequívoca de tipo M.</p><ul><li><b>Fe I 4957 simétrica:</b> K5 (~4 000 K)</li><li><b>Fe I asimétrica (TiO incipiente):</b> M0 (~3 700 K)</li><li><b>TiO domina en 4761 + 5167 Å:</b> M2–M4 (~3 200–3 700 K)</li><li><b>TiO en todo el óptico:</b> M6–M7 (&lt;3 000 K)</li></ul>'
        },
        lineas: [
            { lambda: 4761, label: 'TiO 4761',  color: '#c0392b' },
            { lambda: 4770, label: 'MgH 4770',  color: '#10b981' },
            { lambda: 4957, label: 'Fe I 4957', color: '#95a5a6' },
            { lambda: 5167, label: 'TiO 5167',  color: '#c0392b' },
            { lambda: 6150, label: 'TiO 6150',  color: '#c0392b' },
            { lambda: 6950, label: 'TiO 6950',  color: '#c0392b' },
        ],
        opciones: [
            { texto: 'Fe I 4957 simétrica, TiO muy tenue o ausente', icono: '🟠', clase: 'opcion-comp',
              detalle: 'K5',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K5', criterio: 'Fe I 4957 simétrica, sin bandas TiO evidentes. ~4 000 K.' }},
            { texto: 'Fe I 4957 asimétrica (TiO empieza a mordella)', icono: '🔴', clase: 'opcion-mayor',
              detalle: 'M0',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M0', criterio: 'Fe I 4957 con ala roja asimétrica por TiO incipiente. ~3 700–3 800 K.' }},
            { texto: 'Fe I desaparece, TiO 4761 y 5167 dominan',      icono: '🔴', clase: 'opcion-mmay',
              detalle: 'M2 – M4', siguiente: 'tipoM_sub' },
            { texto: 'TiO dominante en todo el óptico (6150, 6950)',   icono: '🟥', clase: 'opcion-mmay',
              detalle: 'M5 – M7', siguiente: 'tipoM_late' },
        ]
    },

    // ─── TIPO O temprana: O3 / O4 / O5 / O6 ───
    'tipoO_early': {
        bloque: 'O Temprana — N V 4604',
        paso: 'Paso 5',
        pregunta: '¿Se detecta alguna absorción (o emisión débil) en 4604 Å (N V)? Busca cualquier señal, aunque sea pequeña.',
        hint: '🔎 Busca en 4604 Å una pequeña "muesca" hacia abajo (absorción). En O4 NO necesita ser tan intensa como He II — basta con ser visible. Si ves ALGO en 4604 Å → elige la primera opción. Solo si no hay NADA → O5/O6.',
        ayuda: {
            titulo: 'N V 4604 Å — clave para O3 y O4',
            contenido: `<p>N V (4604 y 4620 Å) solo se forma a T &gt; 40 000 K. Es el criterio decisivo para separar O3/O4 de O5/O6.</p>
<p><b>⚠️ Error frecuente:</b> El N V en O4 NO suele ser tan fuerte como He II. Puede verse como una muesca pequeña o moderada. Si hay cualquier absorción en 4604 Å → es O3/O4.</p>
<table style="width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px">
<tr style="background:#1e293b"><th style="padding:4px 8px;text-align:left">Subtipo</th><th style="padding:4px 8px;text-align:left">N V 4604</th><th style="padding:4px 8px;text-align:left">He I/He II</th></tr>
<tr><td style="padding:4px 8px"><b>O3</b></td><td style="padding:4px 8px">Muy fuerte + O V 5114</td><td style="padding:4px 8px">&lt; 0.3</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O4</b></td><td style="padding:4px 8px">Visible (moderado)</td><td style="padding:4px 8px">0.3–0.5</td></tr>
<tr><td style="padding:4px 8px"><b>O5</b></td><td style="padding:4px 8px">Ausente / imperceptible</td><td style="padding:4px 8px">0.5–0.7</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O6</b></td><td style="padding:4px 8px">Ausente</td><td style="padding:4px 8px">0.7–0.85</td></tr>
</table>
<p style="margin-top:8px"><em>Si tienes dudas entre O4 y O5: mira He I. En O4 He I es ≈ mitad de He II; en O5 es ~60–70% de He II.</em></p>`
        },
        lineas: [
            { lambda: 4200, label: 'He II 4200', color: '#e74c3c' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4542, label: 'He II 4542', color: '#e74c3c' },
            { lambda: 4604, label: 'N V 4604',   color: '#1abc9c' },
            { lambda: 4620, label: 'N V 4620',   color: '#1abc9c' },
            { lambda: 5114, label: 'O V 5114',   color: '#f39c12' },
        ],
        opciones: [
            { texto: 'N V 4604 visible — cualquier absorción, aunque sea pequeña o moderada',
              icono: '🟢', clase: 'opcion-si',
              detalle: 'O3 – O4 (T > 40 000 K) → siguiente paso distingue cuál', siguiente: 'tipoO_N34' },
            { texto: 'Sin N V en absoluto; He I ≈ 50–70% de He II (relación He I/He II < 0.70)',
              icono: '🟣', clase: 'opcion-comp',
              detalle: 'O5 (~38 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O5', criterio: 'He I/He II < 0.70, N V completamente ausente. ~38 000 K. Si hubo duda con O4, verificar que no hay ninguna señal en 4604 Å.' }},
            { texto: 'Sin N V; He I ≈ 70–85% de He II (He I algo débil pero claramente visible)',
              icono: '⬅️', clase: 'opcion-menor',
              detalle: 'O6 (~35 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O6', criterio: 'He I/He II 0.70–0.85, N V ausente. ~35 000 K.' }},
        ]
    },

    // ─── NUEVO: O tardía — N III refinamiento (O7 vs O9) ───
    'tipoO_late': {
        bloque: 'O Tardía — N III 4634',
        paso: 'Paso 5',
        pregunta: '¿Se detecta N III 4634/4641 Å? Puede ser absorción débil o incluso una pequeña emisión en esa región.',
        hint: '🔎 Busca en 4634–4641 Å. N III aparece en O7–O8 como la "fluorescencia de Bowen": puede verse en absorción o como leve relleno de la línea. En O9 ya no está. Compara también He I vs He II más cuidadosamente.',
        ayuda: {
            titulo: 'N III 4634/4641 Å — O7, O8, O9, O9.5',
            contenido: `<p>Llegaste aquí porque He I ≈ He II (profundidades similares). Ahora N III 4634/4641 Å afina el subtipo:</p>
<table style="width:100%;border-collapse:collapse;font-size:0.9em;margin-bottom:8px">
<tr style="background:#1e293b"><th style="padding:4px 8px;text-align:left">Subtipo</th><th style="padding:4px 8px;text-align:left">N III 4634</th><th style="padding:4px 8px;text-align:left">He I vs He II</th><th style="padding:4px 8px;text-align:left">T_eff</th></tr>
<tr><td style="padding:4px 8px"><b>O7</b></td><td style="padding:4px 8px">Visible o en emisión</td><td style="padding:4px 8px">He I ≈ 85–95% He II</td><td style="padding:4px 8px">~36 000 K</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O8</b></td><td style="padding:4px 8px">Débil pero detectable</td><td style="padding:4px 8px">He I ≈ He II (90–100%)</td><td style="padding:4px 8px">~34 000 K</td></tr>
<tr><td style="padding:4px 8px"><b>O9</b></td><td style="padding:4px 8px">Ausente</td><td style="padding:4px 8px">He I ≈ He II (95–110%)</td><td style="padding:4px 8px">~32 000 K</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px"><b>O9.5</b></td><td style="padding:4px 8px">Ausente</td><td style="padding:4px 8px">He I levemente > He II</td><td style="padding:4px 8px">~31 000 K</td></tr>
</table>
<p><em>Nota: en estrellas supergigantes (clase Ia) N III suele estar en emisión. En enanas (V) suele estar en absorción.</em></p>`
        },
        lineas: [
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4542, label: 'He II 4542', color: '#e74c3c' },
            { lambda: 4634, label: 'N III 4634', color: '#1abc9c' },
            { lambda: 4641, label: 'N III 4641', color: '#1abc9c' },
            { lambda: 4686, label: 'He II 4686', color: '#e74c3c' },
        ],
        opciones: [
            { texto: 'N III 4634/4641 visible (absorción débil o leve emisión en esa zona)',
              icono: '🟢', clase: 'opcion-si',
              detalle: 'O7 – O8 (~34 000–36 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O7–O8', criterio: 'He I ≈ He II. N III 4634/4641 Å presente (fluorescencia de Bowen). ~34 000–36 000 K. Ej: ξ Per (O7.5III), HD 36591 (O9III).' }},
            { texto: 'N III ausente; He I y He II prácticamente iguales en profundidad',
              icono: '⚖️', clase: 'opcion-comp',
              detalle: 'O9 (~32 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O9', criterio: 'He I ≈ He II (relación 0.95–1.1), N III ausente. ~32 000 K. Transición clásica O9–B0. Ej: ι Ori (O9III).' }},
            { texto: 'N III ausente; He I supera levemente a He II (He I un poco más profunda)',
              icono: '➡️', clase: 'opcion-mayor',
              detalle: 'O9.5 (límite O/B)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O9.5', criterio: 'He I ligeramente > He II, N III ausente, Si III 4553 tenue aparece. ~31 000 K. Límite O–B, catalogado como B0 en algunos sistemas.' }},
        ]
    },

    // ─── NUEVO: O3 vs O4 — O V 5114 Å ───
    'tipoO_N34': {
        bloque: 'O3 vs O4 — O V 5114',
        paso: 'Paso 6',
        pregunta: 'Busca absorción en O V 5114 Å. Si no la encuentras: compara N V 4604 vs N III 4634.',
        hint: '🔎 Desplázate a 5114 Å. En O3 hay una pequeña muesca allí. En O4 esa región es plana (sin O V). Si ya llegaste aquí con N V visible pero moderado y sin O V → es O4.',
        ayuda: {
            titulo: 'O3 vs O4 — O V 5114 Å y N V / N III',
            contenido: `<p>Si llegaste a este nodo es porque detectaste <b>alguna señal de N V 4604 Å</b>. Ahora hay que decidir entre O3 y O4:</p>
<table style="width:100%;border-collapse:collapse;font-size:0.9em;margin-bottom:8px">
<tr style="background:#1e293b"><th style="padding:4px 8px;text-align:left">Indicador</th><th style="padding:4px 8px;text-align:center">O3</th><th style="padding:4px 8px;text-align:center">O4</th></tr>
<tr><td style="padding:4px 8px">O V 5114 Å</td><td style="padding:4px 8px;text-align:center">✅ Presente</td><td style="padding:4px 8px;text-align:center">❌ Ausente</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px">N V 4604 Å</td><td style="padding:4px 8px;text-align:center">Muy fuerte</td><td style="padding:4px 8px;text-align:center">Moderado a fuerte</td></tr>
<tr><td style="padding:4px 8px">N III 4634 Å</td><td style="padding:4px 8px;text-align:center">Débil o ausente</td><td style="padding:4px 8px;text-align:center">Débil a moderado</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px">He I / He II</td><td style="padding:4px 8px;text-align:center">&lt; 0.3</td><td style="padding:4px 8px;text-align:center">0.3 – 0.5</td></tr>
<tr><td style="padding:4px 8px">T_eff aprox.</td><td style="padding:4px 8px;text-align:center">≥ 45 000 K</td><td style="padding:4px 8px;text-align:center">40 000–44 000 K</td></tr>
<tr style="background:#0f172a"><td style="padding:4px 8px">Ejemplos</td><td style="padding:4px 8px;text-align:center">HD 93129A</td><td style="padding:4px 8px;text-align:center">HD 46223, θ¹ Ori C</td></tr>
</table>
<p><b>La mayoría de los O4 NO tienen O V.</b> Si N V es visible pero moderado y O V ausente → O4 es la clasificación correcta.</p>`
        },
        lineas: [
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4542, label: 'He II 4542', color: '#e74c3c' },
            { lambda: 4604, label: 'N V 4604',   color: '#1abc9c' },
            { lambda: 4634, label: 'N III 4634', color: '#16a085' },
            { lambda: 4686, label: 'He II 4686', color: '#e74c3c' },
            { lambda: 5114, label: 'O V 5114',   color: '#f39c12' },
        ],
        opciones: [
            { texto: 'O V 5114 Å presente (pequeña muesca visible) + N V muy fuerte',
              icono: '🔴', clase: 'opcion-mayor',
              detalle: 'O3 (T ≥ 45 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O3', criterio: 'O V 5114 presente, N V muy fuerte >> N III. T ≥ 45 000 K. Ej: HD 93129A. Estrella entre las más calientes de la secuencia principal.' }},
            { texto: 'O V 5114 Å ausente (región plana) + N V visible (moderado a fuerte)',
              icono: '🟠', clase: 'opcion-comp',
              detalle: 'O4 (~40 000–44 000 K)',
              siguiente: 'resultado',
              resultado: { tipo: 'O', subtipo: 'O4', criterio: 'N V 4604 visible, O V 5114 ausente, He I ≈ 30–50% de He II. ~40 000–44 000 K. Ej: HD 46223, θ¹ Ori C.' }},
        ]
    },

    // ─── NUEVO: B1–B3 con O II / C II ───
    'tipoB_OII': {
        bloque: 'B Temprana — O II / C II',
        paso: 'Paso 6',
        pregunta: '¿Se detectan líneas O II en 4415–4417 Å y/o C II 4267 Å?',
        hint: 'O II 4415/4417 Å aparece en B2–B5. C II 4267 Å es más fuerte en B1–B2. La relación entre ellas afina el subtipo mejor que Si III solo.',
        ayuda: {
            titulo: 'O II 4415 y C II 4267 — refinamiento B1–B3',
            contenido: '<p>En B temprana (B1–B3), la región 4250–4420 Å contiene diagnósticos secundarios que afínan el subtipo tras identificar que Si III domina:</p><ul><li><b>C II 4267 fuerte, O II 4415 débil o ausente:</b> B1 (~22 000–24 000 K)</li><li><b>C II ≈ O II (intensidades similares):</b> B2 (~20 000 K)</li><li><b>O II 4415 > C II 4267 (O II domina):</b> B3 (~17 000–18 000 K, Si III en declive)</li></ul>'
        },
        lineas: [
            { lambda: 4128, label: 'Si II 4128', color: '#f39c12' },
            { lambda: 4267, label: 'C II 4267',  color: '#e74c3c' },
            { lambda: 4415, label: 'O II 4415',  color: '#27ae60' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4553, label: 'Si III 4553',color: '#a3e635' },
        ],
        opciones: [
            { texto: 'C II 4267 fuerte; O II 4415 débil o ausente', icono: '🔴', clase: 'opcion-mayor',
              detalle: 'B1',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B1', criterio: 'C II 4267 dominante sobre O II 4415. Si III visible. ~22 000–24 000 K.' }},
            { texto: 'C II ≈ O II (intensidades similares)',         icono: '⚖️', clase: 'opcion-comp',
              detalle: 'B2',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B2', criterio: 'C II 4267 ≈ O II 4415. Si III moderado. ~20 000 K. Regulo B2III es un ejemplo clásico.' }},
            { texto: 'O II 4415 > C II 4267 (O II domina)',          icono: '🟢', clase: 'opcion-menor',
              detalle: 'B3',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B3', criterio: 'O II 4415 > C II 4267. Si III → Si II transición. ~17 000–18 000 K.' }},
        ]
    },

    // ─── NUEVO: F tardía / G — Banda CH (G-band) ───
    'tipoFG_CH': {
        bloque: 'F Tardía / G — Banda CH',
        paso: 'Paso 4',
        pregunta: '¿Es visible la banda CH (G-band) alrededor de 4300 Å? ¿Qué intensidad tiene respecto a Fe I 4383?',
        hint: 'La G-band (4300 Å) es un conjunto de líneas CH moleculares. Aparece tenue en F5 y se refuerza progresivamente. Su ausencia confirma F temprana; si es clara, estamos en F9–G.',
        ayuda: {
            titulo: 'Banda CH (G-band) 4300 Å — transición F/G',
            contenido: '<p>La banda G es un blending de transiciones de la molécula CH. Aparece por primera vez ~F5 (~6 400 K) y es prominente en G:</p><ul><li><b>G-band ausente:</b> F5–F6 (~6 400–6 700 K)</li><li><b>G-band tenue (visible pero débil):</b> F7–F8 (~6 000–6 300 K)</li><li><b>G-band clara y bien definida:</b> F9–G0 (~5 700–6 000 K) — pasar a clasificación G/K</li><li><b>G-band muy fuerte:</b> G2+ (el Sol G2V la tiene prominente)</li></ul>'
        },
        lineas: [
            { lambda: 3934, label: 'Ca II K',    color: '#f39c12' },
            { lambda: 4271, label: 'Fe I 4271',  color: '#95a5a6' },
            { lambda: 4300, label: 'G-band CH',  color: '#e67e22' },
            { lambda: 4383, label: 'Fe I 4383',  color: '#95a5a6' },
            { lambda: 4861, label: 'Hβ',         color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'G-band ausente o imperceptible',        icono: '⬅️', clase: 'opcion-menor',
              detalle: 'F5 – F6',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F5–F6', criterio: 'Balmer ≈ metales, G-band ausente. Ca II K muy fuerte. ~6 400–6 700 K.' }},
            { texto: 'G-band tenue (visible pero débil)',     icono: '⚖️', clase: 'opcion-comp',
              detalle: 'F7 – F8',
              siguiente: 'resultado',
              resultado: { tipo: 'F', subtipo: 'F7–F8', criterio: 'G-band CH tenue, metales ≈ Balmer, Ca II K dominante. ~6 000–6 300 K.' }},
            { texto: 'G-band clara y bien definida',          icono: '🟠', clase: 'opcion-mayor',
              detalle: 'F9 – G0', siguiente: 'tardias' },
        ]
    },

    // ─── NUEVO: tipo G — Triplete Mg I b ───
    'tipoG_Mg': {
        bloque: 'Tipo G — Triplete Mg b',
        paso: 'Paso 4',
        pregunta: '¿Cuál es la intensidad del triplete Mg I b (5167 / 5172 / 5183 Å) respecto a Fe I 5270 Å?',
        hint: 'El triplete Mg b se fortalece de G0 a K0. En el Sol (G2V) es ya moderado. Fe I 5270 Å es la referencia de comparación.',
        ayuda: {
            titulo: 'Triplete Mg I b — subtipos G',
            contenido: '<p>El triplete Mg b (b1=5183.6, b2=5172.7, b3=5167.3 Å) es uno de los mejores indicadores de temperatura en el óptico verde. Se fortalece al bajar T:</p><ul><li><b>Mg b débil (≪ Fe I 5270):</b> G0–G1 (~5 900–6 000 K)</li><li><b>Mg b moderado (comparable a Fe I 5270):</b> G2 (~5 780 K, tipo solar)</li><li><b>Mg b fuerte (> Fe I 5270):</b> G5–G8 (~5 300–5 500 K)</li></ul><p>Nota: Mg b también es sensible a la gravedad; en supergigantes G es más débil que en enanas.</p>'
        },
        lineas: [
            { lambda: 4308, label: 'G-band',    color: '#f39c12' },
            { lambda: 4376, label: 'Y II 4376', color: '#06b6d4' },
            { lambda: 4383, label: 'Fe I 4383', color: '#95a5a6' },
            { lambda: 5167, label: 'Mg b3',     color: '#27ae60' },
            { lambda: 5173, label: 'Mg b2',     color: '#27ae60' },
            { lambda: 5184, label: 'Mg b1',     color: '#27ae60' },
            { lambda: 5270, label: 'Fe I 5270', color: '#95a5a6' },
        ],
        opciones: [
            { texto: 'Mg b débil (≪ Fe I 5270)',                     icono: '🟡', clase: 'opcion-menor',
              detalle: 'G0 – G1',
              siguiente: 'resultado',
              resultado: { tipo: 'G', subtipo: 'G0–G1', criterio: 'Hδ > Fe I 4144, Fe I > Cr I 4254, Mg b débil. G-band presente. ~5 900–6 000 K. (Si Y II 4376 ≥ Fe I 4383 → gigante G0–G1 III.)' }},
            { texto: 'Mg b moderado (comparable a Fe I 5270)',        icono: '⚖️', clase: 'opcion-comp',
              detalle: 'G2',
              siguiente: 'resultado',
              resultado: { tipo: 'G', subtipo: 'G2', criterio: 'Hδ ≈ Fe I 4144, Mg b ≈ Fe I 5270. Tipo solar (~5 780 K). El Sol es G2V. (Si Y II 4376 ≥ Fe I 4383 → G2III.)' }},
            { texto: 'Mg b fuerte (> Fe I 5270)',                     icono: '🟠', clase: 'opcion-mayor',
              detalle: 'G5 – G8', siguiente: 'tipoG5' },
        ]
    },

    // ─── NUEVO: tipo K — Na I D doblete ───
    'tipoK_Na': {
        bloque: 'Tipo K — Na I D',
        paso: 'Paso 4',
        pregunta: '¿Cuál es la intensidad del doblete Na I D (5890 / 5896 Å) respecto a Ca I 4226 Å?',
        hint: 'Na I D se refuerza al bajar la temperatura en K. Es también sensible a la gravedad: más fuerte en enanas que en gigantes del mismo subtipo.',
        ayuda: {
            titulo: 'Na I D 5890/5896 Å — gravedad y temperatura en K',
            contenido: '<p>El doblete Na I D es uno de los indicadores más intensos en espectros de tipo K. Sus fortalezas dependen de temperatura y log g:</p><ul><li><b>Na I D débil o moderado:</b> K0–K1 (~5 000–5 200 K; posiblemente gigante)</li><li><b>Na I D fuerte:</b> K2–K3 (~4 500–5 000 K)</li><li><b>Na I D muy fuerte (≈ Ca II K en intensidad):</b> K4–K5 (&lt;4 500 K, o enana K)</li></ul><p>Ba II 4554 Å es otro indicador de luminosidad en K: más fuerte en gigantes.</p>'
        },
        lineas: [
            { lambda: 4226, label: 'Ca I 4226', color: '#e67e22' },
            { lambda: 4254, label: 'Cr I 4254', color: '#f43f5e' },
            { lambda: 4275, label: 'Cr I 4275', color: '#f43f5e' },
            { lambda: 4290, label: 'Cr I 4290', color: '#f43f5e' },
            { lambda: 4376, label: 'Y II 4376', color: '#06b6d4' },
            { lambda: 4383, label: 'Fe I 4383', color: '#95a5a6' },
            { lambda: 4554, label: 'Ba II 4554',color: '#3498db' },
            { lambda: 5173, label: 'Mg I b',    color: '#27ae60' },
            { lambda: 5890, label: 'Na I D2',   color: '#f39c12' },
            { lambda: 5896, label: 'Na I D1',   color: '#f39c12' },
        ],
        opciones: [
            { texto: 'Na I D débil o moderado (< Ca I 4226)',        icono: '🟠', clase: 'opcion-menor',
              detalle: 'K0 – K1',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K0–K1', criterio: 'Ca I ≈ Fe I, Na I D moderado. ~5 000–5 200 K. Ej: Arcturus K1.5 III.' }},
            { texto: 'Na I D fuerte (comparable a Ca I)',             icono: '🔶', clase: 'opcion-comp',
              detalle: 'K2 – K3',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K2–K3', criterio: 'Na I D fuerte, Ca I > Fe I, Balmer casi ausente. ~4 500–5 000 K.' }},
            { texto: 'Na I D muy fuerte (≈ Ca II K en intensidad)', icono: '🔴', clase: 'opcion-mayor',
              detalle: 'K4 – K5',
              siguiente: 'resultado',
              resultado: { tipo: 'K', subtipo: 'K4–K5', criterio: 'Na I D muy fuerte, Ca I dominante. ~4 000–4 500 K. Enana K probable si Ba II 4554 es débil.' }},
        ]
    },

    // ─── NUEVO: subtipo M con VO + CaH ───
    'tipoM_sub': {
        bloque: 'Tipo M — VO y CaH',
        paso: 'Paso 5',
        pregunta: '¿Se detectan bandas VO (7434 Å) o CaH (6382 Å, 6750 Å) además de TiO?',
        hint: 'VO aparece en M4–M5. CaH es más fuerte en enanas M que en gigantes. La combinación TiO+VO+CaH define subtipos M4–M6.',
        ayuda: {
            titulo: 'VO y CaH — subtipos M intermedios',
            contenido: '<p>En M tardías las bandas VO (óxido de vanadio) complementan a las TiO y permiten afinar subtipos:</p><ul><li><b>Solo TiO, sin VO ni CaH:</b> M2–M3 (~3 400–3 700 K)</li><li><b>TiO + VO 7434 tenue:</b> M4–M5 (~3 100–3 400 K)</li><li><b>TiO + VO fuerte + CaH visible:</b> M5–M6 (~2 900–3 100 K) — continuar</li></ul><p><b>CaH 6382/6750 Å como indicador de clase de luminosidad:</b> En enanas M (clase V) CaH es notablemente más fuerte que en gigantes M (clase III) del mismo subtipo TiO.</p>'
        },
        lineas: [
            { lambda: 4955, label: 'TiO 4955',  color: '#c0392b' },
            { lambda: 5167, label: 'TiO 5167',  color: '#c0392b' },
            { lambda: 6158, label: 'TiO 6158',  color: '#c0392b' },
            { lambda: 6382, label: 'CaH 6382',  color: '#8e44ad' },
            { lambda: 6651, label: 'TiO 6651',  color: '#c0392b' },
            { lambda: 7434, label: 'VO 7434',   color: '#e67e22' },
        ],
        opciones: [
            { texto: 'Solo TiO, sin VO ni CaH detectables',            icono: '🔴', clase: 'opcion-comp',
              detalle: 'M2 – M3',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M2–M3', criterio: 'TiO claro en 4761 y 5167 Å, sin VO ni CaH. ~3 400–3 700 K.' }},
            { texto: 'TiO + VO 7434 tenue emergente',                   icono: '🟥', clase: 'opcion-mayor',
              detalle: 'M4 – M5',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M4–M5', criterio: 'TiO dominante + VO 7434 emergente. ~3 100–3 400 K.' }},
            { texto: 'TiO + VO fuerte + CaH visible (M tardía)',        icono: '🟥', clase: 'opcion-mmay',
              detalle: 'M5 – M6', siguiente: 'tipoM_late' },
        ]
    },

    // ─── NUEVO: M tardía con TiO + VO ───
    'tipoM_late': {
        bloque: 'M Tardía — TiO + VO',
        paso: 'Paso 6',
        pregunta: '¿Cuál es la intensidad de VO 7865 Å respecto a TiO 6651 Å? ¿TiO se debilita?',
        hint: 'En M6–M7 VO 7865 ya es notable. En M8+ puede superar a TiO. El continuo está completamente deformado por bandas moleculares.',
        ayuda: {
            titulo: 'Estrellas M tardías — frontera M/L',
            contenido: '<p>Las estrellas M5–M9 tienen atmósferas dominadas por moléculas diatómicas. El VO gana importancia progresivamente:</p><ul><li><b>TiO 6651 fuerte, VO 7865 tenue:</b> M5–M6 (~2 800–3 000 K)</li><li><b>TiO ≈ VO (ambos muy fuertes):</b> M7–M8 (~2 500–2 800 K)</li><li><b>TiO se debilita, VO domina:</b> M9–L0 (&lt;2 400 K, frontera enana marrón)</li></ul><p>En M tardías CaH 6750 Å ya no distingue bien enanas de gigantes porque las atmósferas son muy frías y complejas.</p>'
        },
        lineas: [
            { lambda: 5448, label: 'TiO 5448',  color: '#c0392b' },
            { lambda: 6158, label: 'TiO 6158',  color: '#c0392b' },
            { lambda: 6651, label: 'TiO 6651',  color: '#c0392b' },
            { lambda: 6750, label: 'CaH 6750',  color: '#8e44ad' },
            { lambda: 7434, label: 'VO 7434',   color: '#e67e22' },
            { lambda: 7865, label: 'VO 7865',   color: '#e67e22' },
        ],
        opciones: [
            { texto: 'TiO 6651 fuerte, VO 7865 tenue',     icono: '🟥', clase: 'opcion-comp',
              detalle: 'M5 – M6',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M5–M6', criterio: 'TiO domina en todo el rojo, VO tenue. CaH visible. ~2 800–3 000 K.' }},
            { texto: 'TiO ≈ VO (ambos muy intensos)',       icono: '🔴', clase: 'opcion-mayor',
              detalle: 'M7 – M8',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M7–M8', criterio: 'TiO y VO comparables en intensidad. Continuo completamente deformado. ~2 500–2 800 K.' }},
            { texto: 'TiO debilitado; VO domina o hay granos', icono: '⬛', clase: 'opcion-mmay',
              detalle: 'M9 – L0',
              siguiente: 'resultado',
              resultado: { tipo: 'M', subtipo: 'M9–L0', criterio: 'TiO debilitado por condensación, VO dominante. Frontera M/L, enana ultra-fría o enana marrón. < 2 400 K.' }},
        ]
    },

    // ─── BLOQUE B9 / B9.5 — Ti II 4468 ───
    'tipoB95': {
        bloque: 'B9 / B9.5 — Ti II',
        paso: 'Paso 6',
        pregunta: '¿Se observa Ti II 4468 Å justo a la izquierda de He I 4471 Å? ¿Tienen intensidad similar?',
        hint: 'Ti II 4468 aparece en B9.5 igualando a He I 4471. Su ausencia o debilidad indica B9. Es el criterio definitivo para este límite.',
        ayuda: {
            titulo: 'Ti II 4468 Å — criterio B9.5',
            contenido: '<p>El titanio ionizado (Ti II) tiene una línea a 4468 Å, inmediatamente a la izquierda de He I 4471. En B9 es muy débil o imperceptible. En B9.5 crece hasta igualar a He I, marcando la frontera B–A.</p><ul><li><b>Ti II ausente o mucho más débil:</b> B9 (~11 000 K)</li><li><b>Ti II ≈ He I 4471:</b> B9.5 (~10 500 K, límite B–A)</li></ul>'
        },
        lineas: [
            { lambda: 4468, label: 'Ti II 4468', color: '#3498db' },
            { lambda: 4471, label: 'He I 4471',  color: '#9b59b6' },
            { lambda: 4481, label: 'Mg II 4481', color: '#27ae60' },
            { lambda: 4861, label: 'Hβ',          color: '#fbbf24' },
        ],
        opciones: [
            { texto: 'Ti II 4468 ausente o mucho más débil que He I', icono: '⬅️', clase: 'opcion-menor',
              detalle: 'B9',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B9', criterio: 'Mg II >> He I, Ti II 4468 no equipara a He I. Frontera B–A, ~11 000 K.' }},
            { texto: 'Ti II 4468 ≈ He I 4471 (intensidades similares)', icono: '⚖️', clase: 'opcion-comp',
              detalle: 'B9.5',
              siguiente: 'resultado',
              resultado: { tipo: 'B', subtipo: 'B9.5', criterio: 'Ti II 4468 iguala a He I 4471. Criterio definitivo de B9.5. ~10 500 K. Transición exacta B–A.' }},
        ]
    },
};

// ---------- ESTADO DEL ÁRBOL ----------
let arbolEstado = {
    nodoActual: null,
    historial: [],       // [{nodoId, opcionTexto, opcionDetalle}]
    profundidad: 0,
    resultado: null,
};

// Espectro cargado por el usuario para visualización
let arbolEspectroData = null;  // { wavelength:[], flux:[], filename:'', wmin, wmax }

// Estado de zoom sobre una línea diagnóstico
let arbolZoom = null;  // null = vista completa | { lmin, lmax, lambda, label }

function arbolZomarLinea(lambda, label) {
    const ventana = 150; // ±150 Å alrededor de la línea
    arbolZoom = { lmin: lambda - ventana, lmax: lambda + ventana, lambda, label };
    const nodo = ARBOL_NODOS[arbolEstado.nodoActual];
    if (nodo) arbolDibujarEspectro(nodo.lineas);
}

function arbolResetZoom() {
    arbolZoom = null;
    const nodo = ARBOL_NODOS[arbolEstado.nodoActual];
    if (nodo) arbolDibujarEspectro(nodo.lineas);
}

// ---------- CARGA DE ESPECTRO DEL USUARIO ----------

function arbolCargarArchivo(input) {
    const file = input.files[0];
    if (!file) return;

    // Mostrar spinner
    document.getElementById('arbolCargaRowVacio').style.display   = 'none';
    document.getElementById('arbolCargaRowActivo').style.display  = 'none';
    document.getElementById('arbolCargaRowCargando').style.display = 'flex';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/spectrum_raw', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            document.getElementById('arbolCargaRowCargando').style.display = 'none';
            if (!data.success) {
                alert('Error al cargar espectro: ' + data.error);
                document.getElementById('arbolCargaRowVacio').style.display = 'flex';
                return;
            }
            arbolEspectroData = {
                wavelength: data.wavelength,
                flux:       data.flux,
                filename:   data.filename,
                wmin:       data.wmin,
                wmax:       data.wmax,
            };
            document.getElementById('arbolCargaNombre').textContent = data.filename;
            document.getElementById('arbolCargaPuntos').textContent = `(${data.n_points} pts, ${Math.round(data.wmin)}–${Math.round(data.wmax)} Å)`;
            document.getElementById('arbolCargaRowActivo').style.display = 'flex';

            // Re-dibujar el SVG del nodo actual con el espectro real
            if (arbolEstado.nodoActual && ARBOL_NODOS[arbolEstado.nodoActual]) {
                arbolDibujarEspectro(ARBOL_NODOS[arbolEstado.nodoActual].lineas);
            }
        })
        .catch(err => {
            document.getElementById('arbolCargaRowCargando').style.display = 'none';
            document.getElementById('arbolCargaRowVacio').style.display = 'flex';
            alert('Error de conexión al cargar el espectro: ' + err.message);
        });

    // Limpiar input para poder cargar el mismo archivo otra vez
    input.value = '';
}

function arbolQuitarEspectro() {
    arbolEspectroData = null;
    document.getElementById('arbolCargaRowActivo').style.display  = 'none';
    document.getElementById('arbolCargaRowVacio').style.display   = 'flex';
    // Re-dibujar sin espectro real
    if (arbolEstado.nodoActual && ARBOL_NODOS[arbolEstado.nodoActual]) {
        arbolDibujarEspectro(ARBOL_NODOS[arbolEstado.nodoActual].lineas);
    }
}

// ---------- INICIALIZACIÓN ----------
document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btnArbolStart');
    if (btnStart) btnStart.addEventListener('click', arbolComenzar);
});

// ---------- FUNCIONES PRINCIPALES ----------

function arbolComenzar() {
    arbolEstado = { nodoActual: 'inicio', historial: [], profundidad: 0, resultado: null };
    document.getElementById('arbolEstadoInicio').style.display = 'none';
    document.getElementById('arbolEstadoResultado').style.display = 'none';
    document.getElementById('arbolEstadoPregunta').style.display = 'flex';
    arbolRenderizarNodo('inicio');
}

function arbolReiniciar() {
    document.getElementById('arbolEstadoPregunta').style.display = 'none';
    document.getElementById('arbolEstadoResultado').style.display = 'none';
    document.getElementById('arbolEstadoInicio').style.display = 'flex';
    arbolEstado = { nodoActual: null, historial: [], profundidad: 0, resultado: null };
    arbolActualizarProgreso(0);
    document.getElementById('arbolBreadcrumb').innerHTML = '';
}

function arbolVolver() {
    if (arbolEstado.historial.length === 0) { arbolReiniciar(); return; }
    // Si estábamos en resultado, volvemos al último nodo
    if (arbolEstado.resultado) {
        arbolEstado.resultado = null;
        document.getElementById('arbolEstadoResultado').style.display = 'none';
        document.getElementById('arbolEstadoPregunta').style.display = 'flex';
    }
    const anterior = arbolEstado.historial.pop();
    arbolEstado.nodoActual = anterior.nodoId;
    arbolEstado.profundidad = arbolEstado.historial.length;
    arbolRenderizarNodo(anterior.nodoId);
}

function arbolElegirOpcion(opcion) {
    // Guardar en historial
    arbolEstado.historial.push({
        nodoId: arbolEstado.nodoActual,
        opcionTexto: opcion.texto,
        opcionDetalle: opcion.detalle,
    });
    arbolEstado.profundidad++;

    if (opcion.siguiente === 'resultado') {
        arbolEstado.resultado = opcion.resultado;
        arbolMostrarResultado();
    } else {
        arbolEstado.nodoActual = opcion.siguiente;
        arbolRenderizarNodo(opcion.siguiente);
    }
}

function arbolRenderizarNodo(nodoId) {
    const nodo = ARBOL_NODOS[nodoId];
    if (!nodo) return;
    arbolZoom = null; // resetear zoom al cambiar de paso

    // Textos
    document.getElementById('arbolPasoBadge').textContent   = nodo.paso;
    document.getElementById('arbolPreguntaTexto').textContent = nodo.pregunta;
    document.getElementById('arbolPreguntaHint').textContent  = nodo.hint;

    // Tooltip científico de la pregunta
    const helpEl    = document.getElementById('arbolPreguntaHelp');
    const tooltipEl = document.getElementById('arbolPreguntaTooltip');
    if (helpEl && tooltipEl) {
        if (nodo.ayuda) {
            tooltipEl.innerHTML = `<div class="tooltip-title">${nodo.ayuda.titulo}</div>${nodo.ayuda.contenido}`;
            helpEl.style.display = 'inline-flex';
        } else {
            helpEl.style.display = 'none';
        }
    }

    // Título espectro
    document.getElementById('arbolEspectroTitle').textContent =
        `Región diagnóstico — ${nodo.bloque}`;

    // SVG espectro esquemático
    arbolDibujarEspectro(nodo.lineas);

    // Leyenda
    const ley = document.getElementById('arbolEspectroLeyenda');
    ley.innerHTML = nodo.lineas.map(l =>
        `<div class="arbol-leyenda-item">
            <div class="arbol-leyenda-dot" style="background:${l.color}"></div>
            <span>${l.label} (${l.lambda} Å)</span>
         </div>`
    ).join('') +
    (arbolEspectroData
        ? `<div class="arbol-leyenda-ew-hint">
               <span class="arbol-leyenda-ew-swatch"></span>
               <span>Área coloreada = zona de integración del Ancho Equivalente
               (EW&nbsp;=&nbsp;∫(1&nbsp;−&nbsp;F/Fc)dλ).
               El rectángulo muestra la anchura equivalente.</span>
           </div>`
        : `<div class="arbol-leyenda-ew-hint arbol-leyenda-ew-hint--off">
               <span>💡 Carga tu espectro para ver el área EW de cada línea</span>
           </div>`);

    // Opciones
    const cont = document.getElementById('arbolOpciones');
    cont.innerHTML = nodo.opciones.map((op, i) => {
        // Construir texto de ayuda expandible
        let ayudaTexto = '';
        if (op.resultado) {
            ayudaTexto = op.resultado.criterio;
        } else if (op.siguiente && ARBOL_NODOS[op.siguiente]) {
            const sig = ARBOL_NODOS[op.siguiente];
            ayudaTexto = `<strong>Siguiente paso:</strong> ${sig.bloque}.<br>${sig.hint}`;
        }
        const ayudaId = `arbolAyuda_${nodoId}_${i}`;
        return `
        <div class="arbol-opcion-wrapper">
            <div class="arbol-opcion-row">
                <button class="btn-arbol-opcion ${op.clase}" onclick="arbolElegirOpcion(ARBOL_NODOS['${nodoId}'].opciones[${i}])">
                    <span class="arbol-opcion-icono">${op.icono}</span>
                    <span class="arbol-opcion-cuerpo">
                        <span class="arbol-opcion-label">${op.texto}</span>
                        <span class="arbol-opcion-detalle">${op.detalle}</span>
                    </span>
                </button>
                ${ayudaTexto ? `<button class="arbol-info-btn" onclick="arbolToggleAyuda('${ayudaId}')" title="Más información">?</button>` : ''}
            </div>
            ${ayudaTexto ? `<div class="arbol-opcion-ayuda" id="${ayudaId}">${ayudaTexto}</div>` : ''}
        </div>`;
    }).join('');

    // ── Aviso de sin cobertura azul en nodos que requieren Ca II K (3934 Å) ──
    // Si el espectro cargado empieza por encima de 3950 Å, Ca II K y Hε son
    // inobservables. Mostramos banner + botón que redirige al nodo alternativo.
    const NODOS_SIN_COBERTURA = {
        'intermedias': { nodoAlt: 'intermedias_alt', descripcion: 'Balmer (Hδ, Hγ, Hβ) + Mg II 4481 + Fe I 4383' },
        'tipoA':       { nodoAlt: 'tipoA_alt',       descripcion: 'anchura de Hδ/Hγ + aparición de Mg II 4481' },
    };
    if (NODOS_SIN_COBERTURA[nodoId] && arbolEspectroData && arbolEspectroData.wmin > 3950) {
        const cfg = NODOS_SIN_COBERTURA[nodoId];
        // Banner de aviso
        const banner = document.createElement('div');
        banner.className = 'criterio-banner criterio-alternativo';
        banner.style.marginBottom = '0.75rem';
        banner.innerHTML =
            `⚠️ <b>Tu espectro empieza en ${Math.round(arbolEspectroData.wmin)} Å</b> — ` +
            `Ca II K (3934 Å) y Hε (3968 Å) están <b>fuera de rango</b>. ` +
            `Las opciones anteriores no son observables en tu espectro.`;
        cont.insertBefore(banner, cont.firstChild);

        // Botón de desvío al nodo alternativo
        const wrapper = document.createElement('div');
        wrapper.className = 'arbol-opcion-wrapper';
        wrapper.innerHTML =
            `<div class="arbol-opcion-row">
                <button class="btn-arbol-opcion opcion-gen"
                    onclick="arbolElegirOpcion({texto:'Sin cobertura Ca II K / Hε', siguiente:'${cfg.nodoAlt}'})">
                    <span class="arbol-opcion-icono">🔭</span>
                    <span class="arbol-opcion-cuerpo">
                        <span class="arbol-opcion-label">Mi espectro no cubre esa región (empieza en ~4000 Å)</span>
                        <span class="arbol-opcion-detalle">→ Usar criterio alternativo: ${cfg.descripcion}</span>
                    </span>
                </button>
            </div>`;
        cont.appendChild(wrapper);
    }

    // Botón volver
    const btnBack = document.getElementById('btnArbolBack');
    btnBack.disabled = arbolEstado.historial.length === 0;

    // Progreso y breadcrumb
    const maxPasos = 6;
    const pct = Math.min(100, Math.round((arbolEstado.profundidad / maxPasos) * 100));
    arbolActualizarProgreso(pct);
    arbolActualizarBreadcrumb();
}

function arbolToggleAyuda(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('visible');
}

// Mide el ancho equivalente (EW) aproximado en una ventana alrededor de lambda.
// Retorna EW en Å (positivo = absorción) o null si no hay espectro/cobertura.
function arbolMedirEW(lambda, ventana = 18) {
    if (!arbolEspectroData) return null;
    const wArr = arbolEspectroData.wavelength;
    const fArr = arbolEspectroData.flux;

    // Verificar que lambda esté dentro del rango del espectro
    if (lambda < wArr[0] - ventana || lambda > wArr[wArr.length - 1] + ventana) return null;

    let ew = 0;
    let count = 0;
    for (let i = 1; i < wArr.length; i++) {
        const w = wArr[i];
        if (Math.abs(w - lambda) <= ventana) {
            const dw = wArr[i] - wArr[i - 1];
            if (dw > 0 && dw < 15) {          // descartar gaps grandes
                ew += (1 - fArr[i]) * dw;     // EW = ∫(1 - F/Fc)dλ
                count++;
            }
        }
    }
    if (count < 2) return null;
    return ew; // Å
}

function arbolDibujarEspectro(lineas) {
    const svg = document.getElementById('arbolSVG');
    const W = 1400;
    const X_LEFT = 65;  // margen izquierdo para eje Y
    const Y_TOP  = 90;  // espacio extra para etiquetas escalonadas
    const Y_BOT  = 572;
    const Y_TICK = 582;
    const Y_TICK2= 600;
    const Y_LABEL= 624;

    // Rango λ a mostrar
    let LMIN, LMAX;
    if (arbolZoom) {
        LMIN = arbolZoom.lmin;
        LMAX = arbolZoom.lmax;
    } else if (arbolEspectroData) {
        const margin = (arbolEspectroData.wmax - arbolEspectroData.wmin) * 0.01;
        LMIN = arbolEspectroData.wmin - margin;
        LMAX = arbolEspectroData.wmax + margin;
    } else if (lineas.length > 0) {
        const lambdas = lineas.map(l => l.lambda);
        const lCenter = (Math.min(...lambdas) + Math.max(...lambdas)) / 2;
        const spread  = Math.max(500, Math.max(...lambdas) - Math.min(...lambdas) + 500);
        LMIN = Math.max(3700, lCenter - spread / 2 - 200);
        LMAX = Math.min(7200, lCenter + spread / 2 + 200);
    } else {
        LMIN = 3700; LMAX = 7200;
    }

    // Expandir la vista hacia la IZQUIERDA para incluir líneas diagnóstico
    // de longitud de onda corta (ej. Ca II K 3934 Å) aunque estén fuera
    // de la cobertura espectral. NO se expande hacia la derecha (>6000 Å).
    if (!arbolZoom && lineas.length > 0) {
        lineas.forEach(lin => {
            if (lin.lambda < 6000 && lin.lambda - 80 < LMIN) LMIN = lin.lambda - 80;
        });
    }

    const lambdaToX = l => X_LEFT + ((l - LMIN) / (LMAX - LMIN)) * (W - X_LEFT);

    // ── Pre-calcular rango de flujo (para eje Y y escala EW) ──────────
    let fMin = 0.0, fMax = 1.5;
    let ptsVisible = [];
    let scaleF = null;
    if (arbolEspectroData) {
        const wArr = arbolEspectroData.wavelength;
        const fArr = arbolEspectroData.flux;
        let fMinLocal = Infinity, fMaxLocal = -Infinity;
        for (let i = 0; i < wArr.length; i++) {
            if (wArr[i] >= LMIN && wArr[i] <= LMAX) {
                if (fArr[i] < fMinLocal) fMinLocal = fArr[i];
                if (fArr[i] > fMaxLocal) fMaxLocal = fArr[i];
                ptsVisible.push([wArr[i], fArr[i]]);
            }
        }
        if (ptsVisible.length >= 2 && fMaxLocal > fMinLocal) {
            fMin = fMinLocal; fMax = fMaxLocal;
            scaleF = f => Y_TOP + (1 - (f - fMin) / (fMax - fMin)) * (Y_BOT - Y_TOP);
        }
    }

    let s = '';

    // Definiciones: gradiente de color + clipPath + filtro glow EW
    s += `<defs>
      <linearGradient id="specGrad" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%"   stop-color="#7b00ff"/>
        <stop offset="15%"  stop-color="#4466ff"/>
        <stop offset="30%"  stop-color="#44aaff"/>
        <stop offset="45%"  stop-color="#44ffaa"/>
        <stop offset="60%"  stop-color="#aaff44"/>
        <stop offset="75%"  stop-color="#ffdd00"/>
        <stop offset="88%"  stop-color="#ff6600"/>
        <stop offset="100%" stop-color="#cc0000"/>
      </linearGradient>
      <clipPath id="svgClip">
        <rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"/>
      </clipPath>
      <filter id="ewGlow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="ewGlowSoft" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="9" result="blur"/>
        <feMerge><feMergeNode in="blur"/></feMerge>
      </filter>
    </defs>`;

    // Franja cromática de fondo
    s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${W - X_LEFT}" height="${Y_BOT - Y_TOP}"
            fill="url(#specGrad)" opacity="0.10" rx="4"/>`;

    // Línea base del eje X
    s += `<line x1="${X_LEFT}" y1="${Y_BOT}" x2="${W}" y2="${Y_BOT}"
            stroke="#333" stroke-width="1"/>`;

    // ── Eje Y (flujo) ──────────────────────────────────────────────────
    s += `<line x1="${X_LEFT}" y1="${Y_TOP}" x2="${X_LEFT}" y2="${Y_BOT}"
            stroke="#555" stroke-width="1.5"/>`;

    const yRange = fMax - fMin;
    let yTickStep = 0.5;
    if (yRange <= 0.3)      yTickStep = 0.05;
    else if (yRange <= 0.6) yTickStep = 0.1;
    else if (yRange <= 1.2) yTickStep = 0.2;
    else if (yRange <= 3.0) yTickStep = 0.5;
    else                    yTickStep = 1.0;

    const scaleY = scaleF || (f => Y_TOP + (1 - (f - fMin) / (fMax - fMin)) * (Y_BOT - Y_TOP));
    const yTickStart = Math.ceil(fMin / yTickStep) * yTickStep;
    const yTickCount = Math.ceil((fMax - yTickStart) / yTickStep) + 2;
    for (let ti = 0; ti < yTickCount; ti++) {
        const fVal = Math.round((yTickStart + ti * yTickStep) * 1e6) / 1e6;
        if (fVal > fMax + 1e-9) break;
        const y = scaleY(fVal);
        if (y < Y_TOP - 5 || y > Y_BOT + 5) continue;
        // Grid horizontal sutil
        s += `<line x1="${X_LEFT}" y1="${y.toFixed(1)}" x2="${W}" y2="${y.toFixed(1)}"
                stroke="#2a3040" stroke-width="1" opacity="0.6"/>`;
        // Tick externo
        s += `<line x1="${(X_LEFT - 8).toFixed(1)}" y1="${y.toFixed(1)}"
                x2="${X_LEFT}" y2="${y.toFixed(1)}"
                stroke="#667" stroke-width="1.5"/>`;
        // Etiqueta de flujo
        s += `<text x="${(X_LEFT - 10).toFixed(1)}" y="${(y + 5).toFixed(1)}"
                text-anchor="end" font-size="14" fill="#778"
                font-family="monospace">${fVal.toFixed(2)}</text>`;
    }
    // Título del eje Y (rotado)
    const yMid = ((Y_TOP + Y_BOT) / 2).toFixed(1);
    s += `<text x="14" y="${yMid}" text-anchor="middle" font-size="14" fill="#556"
            font-family="monospace"
            transform="rotate(-90, 14, ${yMid})">Flujo</text>`;

    // ── Espectro real del usuario ──────────────────────────────────────
    if (scaleF && ptsVisible.length >= 2) {
        const step = Math.max(1, Math.floor(ptsVisible.length / 1200));
        const pts  = ptsVisible
            .filter((_, i) => i % step === 0)
            .map(([w, f]) => `${lambdaToX(w).toFixed(1)},${scaleF(f).toFixed(1)}`)
            .join(' ');

        const ptsFill = `${X_LEFT},${Y_BOT} ` + pts + ` ${W},${Y_BOT}`;
        s += `<polygon points="${ptsFill}" fill="#7ecfff" opacity="0.07"
                clip-path="url(#svgClip)"/>`;
        s += `<polyline points="${pts}" fill="none" stroke="#7ecfff"
                stroke-width="2.5" opacity="0.92" clip-path="url(#svgClip)"/>`;
        s += `<text x="${X_LEFT + 10}" y="${Y_TOP + 26}" font-size="18" fill="#7ecfff"
                opacity="0.80" font-family="monospace">▲ tu espectro</text>`;

        const yContRef = scaleF(1.0);
        if (yContRef >= Y_TOP && yContRef <= Y_BOT) {
            s += `<line x1="${X_LEFT}" y1="${yContRef.toFixed(1)}" x2="${W}" y2="${yContRef.toFixed(1)}"
                    stroke="#ffffff" stroke-width="1" stroke-dasharray="8,12" opacity="0.28"/>`;
            s += `<text x="${X_LEFT + 8}" y="${(yContRef - 5).toFixed(1)}" font-size="15"
                    fill="#ffffff" opacity="0.40" font-family="monospace">cont=1</text>`;
        }
    } else if (!arbolEspectroData) {
        // Sin espectro: continuo simulado como guía
        const continuo = arbolContinuoSimulado(W, X_LEFT, Y_TOP, Y_BOT);
        s += `<polyline points="${continuo}" fill="none"
                stroke="rgba(255,255,255,0.25)" stroke-width="2" stroke-dasharray="10,8"/>`;
        s += `<text x="${X_LEFT + 10}" y="${Y_TOP + 28}" font-size="18" fill="#445" opacity="0.7"
                font-family="monospace">▲ carga tu espectro para verlo aquí</text>`;
    }

    // ── Indicador de zoom (arriba a la derecha) ────────────────────────
    if (arbolZoom) {
        s += `<rect x="1200" y="4" width="196" height="34" rx="6"
                fill="#dc2626" opacity="0.88" cursor="pointer"
                onclick="arbolResetZoom()"/>`;
        s += `<text x="1298" y="26" text-anchor="middle" font-size="16"
                fill="white" font-family="monospace" font-weight="bold"
                cursor="pointer" onclick="arbolResetZoom()">↩ Vista completa</text>`;
        s += `<text x="${W - 8}" y="14" text-anchor="end" font-size="11" fill="#94a3b8"
                font-family="monospace" opacity="0.80">🔍 zoom: ${arbolZoom.label} ${arbolZoom.lambda} Å</text>`;
    } else if (lineas.length > 0) {
        s += `<text x="${W - 8}" y="14" text-anchor="end" font-size="11" fill="#334155"
                font-family="monospace" opacity="0.65">💡 toca una línea para hacer zoom</text>`;
    }

    // ── Eje de longitud de onda ────────────────────────────────────────
    const rango = LMAX - LMIN;
    const tickStep = rango > 1500 ? 500 : rango > 500 ? 200 : rango > 200 ? 50 : 20;
    const tickStart = Math.ceil(LMIN / tickStep) * tickStep;
    for (let l = tickStart; l <= LMAX; l += tickStep) {
        const x = lambdaToX(l);
        if (x < X_LEFT || x > W) continue;
        s += `<line x1="${x.toFixed(1)}" y1="${Y_TICK}" x2="${x.toFixed(1)}" y2="${Y_TICK2}"
                stroke="#444" stroke-width="2"/>`;
        s += `<text x="${x.toFixed(1)}" y="${Y_LABEL}" text-anchor="middle"
                font-size="18" fill="#778">${l}</text>`;
    }
    s += `<text x="${((X_LEFT + W) / 2).toFixed(1)}" y="${Y_LABEL + 22}" text-anchor="middle"
            font-size="16" fill="#556">Longitud de onda (Å)</text>`;

    // ── Región "sin cobertura" si la vista se expandió más allá del espectro ──
    const specWmin = arbolEspectroData ? arbolEspectroData.wmin : null;
    const specWmax = arbolEspectroData ? arbolEspectroData.wmax : null;
    if (!arbolZoom && specWmin !== null && specWmin > LMIN) {
        const xCovStart = lambdaToX(specWmin);
        const xCovWidth = (xCovStart - X_LEFT).toFixed(1);
        const yMidPlot  = ((Y_TOP + Y_BOT) / 2).toFixed(1);
        s += `<rect x="${X_LEFT}" y="${Y_TOP}" width="${xCovWidth}"
                height="${Y_BOT - Y_TOP}" fill="#0d1117" opacity="0.55"/>`;
        s += `<text x="${((X_LEFT + xCovStart) / 2).toFixed(1)}" y="${(+yMidPlot - 12).toFixed(1)}"
                text-anchor="middle" font-size="14" fill="#475569"
                font-family="monospace" opacity="0.80">sin cobertura</text>`;
        s += `<text x="${((X_LEFT + xCovStart) / 2).toFixed(1)}" y="${(+yMidPlot + 10).toFixed(1)}"
                text-anchor="middle" font-size="12" fill="#334155"
                font-family="monospace" opacity="0.70">λ &lt; ${specWmin.toFixed(0)} Å</text>`;
        s += `<line x1="${xCovStart.toFixed(1)}" y1="${Y_TOP}"
                x2="${xCovStart.toFixed(1)}" y2="${Y_BOT}"
                stroke="#475569" stroke-width="2" stroke-dasharray="6,4" opacity="0.75"/>`;
    }

    // ── Líneas diagnóstico — con escalonamiento vertical anti-solapamiento ──
    // Pre-calcular ranuras verticales: líneas a < 75 px se escalonan hacia arriba
    const _xPos   = lineas.map(lin => lambdaToX(lin.lambda));
    const _slotOf = new Array(lineas.length).fill(0);
    for (let _i = 0; _i < lineas.length; _i++) {
        const _used = new Set();
        for (let _j = 0; _j < _i; _j++) {
            if (Math.abs(_xPos[_j] - _xPos[_i]) < 75) _used.add(_slotOf[_j]);
        }
        let _s = 0;
        while (_used.has(_s)) _s++;
        _slotOf[_i] = Math.min(_s, 2);  // máximo 3 niveles (slots 0, 1, 2)
    }

    for (let _idx = 0; _idx < lineas.length; _idx++) {
        const lin  = lineas[_idx];
        const x    = _xPos[_idx];
        if (x < X_LEFT - 10 || x > W + 10) continue;
        const xc   = x.toFixed(1);
        const slot = _slotOf[_idx];
        const slotOff = slot * 26;   // 26 px por nivel

        // Posiciones Y de las etiquetas (escalonadas)
        const yName = Y_TOP - 42 - slotOff;
        const yEW   = Y_TOP - 24 - slotOff;
        const yConn = Y_TOP - 16 - slotOff;

        // ¿La línea está dentro de la cobertura real del espectro?
        const inRange = specWmin === null ||
                        (lin.lambda >= specWmin && lin.lambda <= specWmax);
        const ew = inRange ? arbolMedirEW(lin.lambda) : null;

        const drawColor   = inRange ? lin.color : '#64748b';
        const drawOpacity = inRange ? '0.92'    : '0.45';
        const fillOpacity = inRange ? '0.20'    : '0.08';
        const dashArray   = inRange ? '10,6'    : '4,4';

        s += `<rect x="${(x - 3).toFixed(1)}" y="${Y_TOP}" width="6"
                height="${Y_BOT - Y_TOP}" fill="${drawColor}" opacity="${fillOpacity}" rx="1"/>`;
        s += `<line x1="${xc}" y1="${Y_TOP}" x2="${xc}" y2="${Y_BOT}"
                stroke="${drawColor}" stroke-width="1.5" opacity="${drawOpacity}"
                stroke-dasharray="${dashArray}"/>`;

        // ── Visualización del Ancho Equivalente (EW) ────────────────────
        if (inRange && ew !== null && scaleF && arbolEspectroData) {
            const ventana = 18;
            const wArr    = arbolEspectroData.wavelength;
            const fArr    = arbolEspectroData.flux;
            const yContLn = Math.max(Y_TOP, Math.min(Y_BOT, scaleF(1.0)));

            // 1. Extraer puntos del perfil real en la ventana de integración
            const ptsEW = [];
            for (let pi = 0; pi < wArr.length; pi++) {
                if (Math.abs(wArr[pi] - lin.lambda) <= ventana) {
                    const xp = lambdaToX(wArr[pi]);
                    const yp = Math.max(Y_TOP, Math.min(Y_BOT, scaleF(fArr[pi])));
                    if (xp >= X_LEFT && xp <= W) ptsEW.push([xp, yp]);
                }
            }

            if (ptsEW.length >= 3) {
                const xL = ptsEW[0][0].toFixed(1);
                const xR = ptsEW[ptsEW.length - 1][0].toFixed(1);
                const yC = yContLn.toFixed(1);

                // Puntos del espectro (izq → der)
                const specStr = ptsEW.map(([px, py]) =>
                    `${px.toFixed(1)},${py.toFixed(1)}`).join(' ');

                // Polígono de absorción: espectro + cierre por el continuo
                const polyPts = specStr +
                    ` ${xR},${yC} ${xL},${yC}`;

                // Halo difuminado (capa exterior, muy suave)
                s += `<polygon points="${polyPts}"
                        fill="${lin.color}" opacity="0.10"
                        filter="url(#ewGlowSoft)"
                        clip-path="url(#svgClip)"/>`;

                // Área de integración (polígono principal con glow)
                s += `<polygon points="${polyPts}"
                        fill="${lin.color}" opacity="0.18"
                        filter="url(#ewGlow)"
                        clip-path="url(#svgClip)"/>`;

                // Borde superior del perfil (contorno sutil)
                s += `<polyline points="${specStr}"
                        fill="none" stroke="${lin.color}"
                        stroke-width="1.8" opacity="0.45"
                        clip-path="url(#svgClip)"/>`;
            }

            // 2. Rectángulo equivalente: ancho = EW, altura = 1 unidad de flujo
            //    → muestra que este rectángulo tiene el mismo área que la absorción
            if (ew > 0.05) {
                const pxPerAng = (W - X_LEFT) / (LMAX - LMIN);
                const ewPx     = ew * pxPerAng;
                const xEwL     = (x - ewPx / 2).toFixed(1);
                const yC       = yContLn.toFixed(1);
                const rectH    = Math.min(30, (Y_BOT - Y_TOP) * 0.12);

                // Sombra difuminada del rectángulo
                s += `<rect x="${xEwL}" y="${yC}"
                        width="${ewPx.toFixed(1)}" height="${rectH.toFixed(1)}"
                        fill="${lin.color}" opacity="0.12"
                        filter="url(#ewGlowSoft)"
                        clip-path="url(#svgClip)" rx="2"/>`;

                // Rectángulo equivalente principal
                s += `<rect x="${xEwL}" y="${yC}"
                        width="${ewPx.toFixed(1)}" height="${rectH.toFixed(1)}"
                        fill="${lin.color}" opacity="0.22"
                        clip-path="url(#svgClip)" rx="2"/>`;

                // Corchetes ⟵ EW → (líneas laterales del rectángulo)
                s += `<line x1="${xEwL}" y1="${yC}"
                        x2="${xEwL}" y2="${(+yC + rectH).toFixed(1)}"
                        stroke="${lin.color}" stroke-width="1.5" opacity="0.70"/>`;
                s += `<line x1="${(x + ewPx/2).toFixed(1)}" y1="${yC}"
                        x2="${(x + ewPx/2).toFixed(1)}" y2="${(+yC + rectH).toFixed(1)}"
                        stroke="${lin.color}" stroke-width="1.5" opacity="0.70"/>`;

                // Etiqueta EW con flecha bidireccional
                const yEWLabel = (+yC + rectH + 14).toFixed(1);
                s += `<text x="${x.toFixed(1)}" y="${yEWLabel}"
                        text-anchor="middle" font-size="11"
                        fill="${lin.color}" opacity="0.80"
                        font-family="monospace">←EW=${ew.toFixed(1)} Å→</text>`;
            }
        } else if (inRange && scaleF && !arbolEspectroData) {
            // Sin espectro cargado: mostrar sólo el bracket de referencia
            const fluxMin = 1.0 - 0.05;
            const yFlux   = scaleF(fluxMin);
            if (yFlux >= Y_TOP && yFlux <= Y_BOT) {
                s += `<line x1="${(x-22).toFixed(1)}" y1="${yFlux.toFixed(1)}"
                        x2="${(x+22).toFixed(1)}" y2="${yFlux.toFixed(1)}"
                        stroke="${lin.color}" stroke-width="2.5" opacity="0.75"/>`;
                s += `<line x1="${(x-22).toFixed(1)}" y1="${(yFlux-7).toFixed(1)}"
                        x2="${(x-22).toFixed(1)}" y2="${(yFlux+7).toFixed(1)}"
                        stroke="${lin.color}" stroke-width="1.5" opacity="0.60"/>`;
                s += `<line x1="${(x+22).toFixed(1)}" y1="${(yFlux-7).toFixed(1)}"
                        x2="${(x+22).toFixed(1)}" y2="${(yFlux+7).toFixed(1)}"
                        stroke="${lin.color}" stroke-width="1.5" opacity="0.60"/>`;
            }
        }

        // Etiquetas en la zona superior (escalonadas según slot)
        s += `<text x="${xc}" y="${yName}" text-anchor="middle"
                font-size="13" fill="${drawColor}" font-family="monospace"
                font-weight="bold" opacity="${inRange ? '1.0' : '0.60'}">${lin.label}</text>`;
        if (inRange) {
            const ewLabel = ew !== null ? `EW≈${ew.toFixed(1)} Å` : `${lin.lambda} Å`;
            const ewColor = ew !== null ? '#e2e8f0' : '#64748b';
            s += `<text x="${xc}" y="${yEW}" text-anchor="middle"
                    font-size="11" fill="${ewColor}" font-family="monospace"
                    opacity="0.85">${ewLabel}</text>`;
        } else {
            s += `<text x="${xc}" y="${yEW}" text-anchor="middle"
                    font-size="10" fill="#64748b" font-family="monospace"
                    opacity="0.75">sin cobertura</text>`;
        }
        // Línea conectora desde la etiqueta hasta la parte superior del plot
        s += `<line x1="${xc}" y1="${yConn}" x2="${xc}" y2="${Y_TOP}"
                stroke="${drawColor}" stroke-width="1" stroke-dasharray="3,3" opacity="0.35"/>`;

        // Área cliqueable para zoom (solo si en rango)
        if (inRange) {
            const safeLabel = lin.label.replace(/'/g, "\\'");
            s += `<rect x="${(x - 22).toFixed(1)}" y="0" width="44"
                    height="${Y_BOT}" fill="transparent" style="cursor:pointer"
                    onclick="arbolZomarLinea(${lin.lambda}, '${safeLabel}')"
                    title="Zoom en ${safeLabel}"/>`;
        }
    }

    svg.innerHTML = s;
}

function arbolContinuoSimulado(W, xLeft, yTop, yBot) {
    // Continuo tipo cuerpo negro (pico en ~5500 Å → franja central)
    const pts = [];
    const rango = yBot - yTop;
    for (let xi = xLeft; xi <= W; xi += 15) {
        const frac = (xi - xLeft) / (W - xLeft);
        const norm = Math.exp(-3 * (frac - 0.50) * (frac - 0.50));
        const y = yBot - rango * 0.55 * norm - rango * 0.1;
        pts.push(`${xi},${Math.min(yBot, Math.max(yTop, Math.round(y)))}`);
    }
    return pts.join(' ');
}

function arbolActualizarProgreso(pct) {
    const fill = document.getElementById('arbolProgressFill');
    const label = document.getElementById('arbolProgressLabel');
    if (fill) fill.style.width = pct + '%';
    if (label) {
        const paso = arbolEstado.historial.length;
        label.textContent = `Paso ${paso} / ≈6 preguntas · 26 nodos`;
    }
}

function arbolActualizarBreadcrumb() {
    const bc = document.getElementById('arbolBreadcrumb');
    if (!bc) return;
    let html = '';
    arbolEstado.historial.forEach((item, i) => {
        const nodo = ARBOL_NODOS[item.nodoId];
        if (i > 0) html += '<span class="arbol-crumb-arrow">›</span>';
        html += `<span class="arbol-crumb" title="${item.opcionTexto}">${nodo ? nodo.bloque : item.nodoId}</span>`;
    });
    bc.innerHTML = html;
}

// ---------- RESULTADO FINAL ----------

function arbolMostrarResultado() {
    document.getElementById('arbolEstadoPregunta').style.display = 'none';
    const r = arbolEstado.resultado;

    // Badge con letra del tipo
    const badge = document.getElementById('arbolResultadoBadge');
    badge.textContent = r.tipo;

    document.getElementById('arbolResultadoTipo').textContent =
        `Tipo Espectral: ${r.tipo}`;
    document.getElementById('arbolResultadoSubtipo').textContent =
        `Subtipo estimado: ${r.subtipo}`;
    document.getElementById('arbolResultadoCriterio').textContent =
        `📌 Criterio: ${r.criterio}`;

    // Archivo analizado
    const archivoEl = document.getElementById('arbolResultadoArchivo');
    if (archivoEl) {
        if (arbolEspectroData) {
            archivoEl.textContent = `📂 Espectro analizado: ${arbolEspectroData.filename}`;
            archivoEl.style.display = 'block';
        } else {
            archivoEl.style.display = 'none';
        }
    }

    // ── Clase de luminosidad MK ──────────────────────────────────────────
    const lumSection  = document.getElementById('arbolLumSection');
    const lumLoading  = document.getElementById('arbolLumLoading');
    const lumResult   = document.getElementById('arbolLumResult');
    const lumError    = document.getElementById('arbolLumError');
    const lumNoSpec   = document.getElementById('arbolLumNoSpectrum');
    lumSection.style.display = 'block';
    lumResult.style.display  = 'none';
    lumError.style.display   = 'none';
    lumNoSpec.style.display  = 'none';

    if (arbolEspectroData) {
        lumLoading.style.display = 'block';
        // Llamar al endpoint backend con los datos del espectro cargado
        fetch('/api/luminosity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                wavelength:    arbolEspectroData.wavelength,
                flux:          arbolEspectroData.flux,
                spectral_type: r.subtipo || r.tipo   // usar subtipo del árbol como tipo espectral
            })
        })
        .then(resp => resp.json())
        .then(data => {
            lumLoading.style.display = 'none';
            if (data.success) {
                document.getElementById('arbolMKFull').textContent  = data.mk_full;
                document.getElementById('arbolLumName').textContent = data.lum_name;
                // Mostrar indicadores clave usados
                const indDiv = document.getElementById('arbolLumIndicators');
                const lumNames = {
                    'Ia': 'Supergigante muy luminosa', 'Ib': 'Supergigante',
                    'II': 'Gigante brillante', 'III': 'Gigante',
                    'IV': 'Subgigante', 'V': 'Secuencia principal'
                };
                const indNames = {
                    'HeI_HeII':  'He I / He II',   'NIII_HeII': 'N III / He II',
                    'SrII_FeI':  'Sr II / Fe I',   'CaI_FeI':   'Ca I / Fe I',
                    'BaII_FeI':  'Ba II / Fe I',   'NaI_CaI':   'Na I D / Ca I',
                    'MgIb_FeI':  'Mg b / Fe I',    'TiO_CaH':   'TiO / CaH'
                };
                const inds = data.indicators || {};
                indDiv.innerHTML = Object.entries(inds)
                    .filter(([, v]) => v > 0)
                    .map(([k, v]) => `<span class="lum-ind-pill">${indNames[k] || k}: <b>${v}</b></span>`)
                    .join('');
                lumResult.style.display = 'flex';
            } else {
                lumError.textContent   = `⚠️ No se pudo calcular: ${data.error}`;
                lumError.style.display = 'block';
            }
        })
        .catch(err => {
            lumLoading.style.display = 'none';
            lumError.textContent   = `⚠️ Error de conexión: ${err.message}`;
            lumError.style.display = 'block';
        });
    } else {
        lumLoading.style.display = 'none';
        lumNoSpec.style.display  = 'block';
    }

    // Camino recorrido
    const lista = document.getElementById('arbolCaminoLista');
    lista.innerHTML = arbolEstado.historial.map((item, i) => {
        const nodo = ARBOL_NODOS[item.nodoId];
        return `<li><strong>${nodo ? nodo.bloque : 'Paso ' + (i+1)}:</strong> ${item.opcionTexto}</li>`;
    }).join('');

    // Progreso al 100%
    arbolActualizarProgreso(100);
    document.getElementById('arbolProgressLabel').textContent = 'Completado';
    arbolActualizarBreadcrumb();

    document.getElementById('arbolEstadoResultado').style.display = 'flex';
}

// ---------- EXPORTAR ----------

function arbolExportarTXT() {
    if (!arbolEstado.resultado) return;
    const r = arbolEstado.resultado;
    const lineas = [
        '=== CLASIFICACIÓN ESPECTRAL INTERACTIVA ===',
        `Fecha: ${new Date().toLocaleString()}`,
        `Espectro       : ${arbolEspectroData ? arbolEspectroData.filename : 'No especificado'}`,
        '',
        `Tipo Espectral : ${r.tipo}`,
        `Subtipo        : ${r.subtipo}`,
        `Tipo MK completo: ${document.getElementById('arbolMKFull').textContent || 'No calculado'}`,
        `Clase luminosidad: ${document.getElementById('arbolLumName').textContent || 'No calculada'}`,
        '',
        `Criterio aplicado:`,
        r.criterio,
        '',
        '--- Camino de decisión ---',
        ...arbolEstado.historial.map((item, i) => {
            const nodo = ARBOL_NODOS[item.nodoId];
            return `${i+1}. [${nodo ? nodo.bloque : 'Paso'}] ${item.opcionTexto} → ${item.opcionDetalle}`;
        }),
        '',
        '=== FIN DEL INFORME ==='
    ];
    const blob = new Blob([lineas.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clasificacion_${r.tipo}_${r.subtipo.replace(/[^a-zA-Z0-9]/g, '_')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

function arbolExportarPDF() {
    if (!arbolEstado.resultado) return;
    const r = arbolEstado.resultado;

    // Construir HTML del informe para imprimir
    const camino = arbolEstado.historial.map((item, i) => {
        const nodo = ARBOL_NODOS[item.nodoId];
        return `<li><strong>${nodo ? nodo.bloque : 'Paso ' + (i+1)}:</strong> ${item.opcionTexto} <em>(${item.opcionDetalle})</em></li>`;
    }).join('');

    const html = `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Clasificación Espectral — ${r.tipo} ${r.subtipo}</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; color: #222; }
  h1 { color: #764ba2; border-bottom: 2px solid #764ba2; padding-bottom: 8px; }
  .tipo { font-size: 3rem; font-weight: 900; color: #3498db; }
  .subtipo { font-size: 1.2rem; color: #555; }
  .criterio { background: #f4f6f9; border-left: 4px solid #764ba2; padding: 12px 16px; margin: 20px 0; border-radius: 0 8px 8px 0; }
  ol li { margin: 6px 0; font-size: 0.93rem; }
  .footer { margin-top: 40px; font-size: 0.8rem; color: #aaa; border-top: 1px solid #eee; padding-top: 10px; }
</style>
</head>
<body>
<h1>🌟 Clasificación Espectral Interactiva</h1>
<p><strong>Fecha:</strong> ${new Date().toLocaleString()}</p>
${arbolEspectroData ? `<p><strong>Espectro:</strong> ${arbolEspectroData.filename}</p>` : ''}
<div class="tipo">${r.tipo}</div>
<div class="subtipo">Subtipo estimado: <strong>${r.subtipo}</strong></div>
${document.getElementById('arbolMKFull').textContent
  ? `<div class="criterio" style="background:#e8f5e9;border-color:#27ae60">⭐ <strong>Tipo MK completo:</strong> ${document.getElementById('arbolMKFull').textContent} — ${document.getElementById('arbolLumName').textContent}</div>`
  : ''}
<div class="criterio"><strong>Criterio:</strong> ${r.criterio}</div>
<h2>Camino de decisión</h2>
<ol>${camino}</ol>
<div class="footer">Sistema de Clasificación Espectral v2.0 — Árbol Interactivo</div>
</body>
</html>`;

    const w = window.open('', '_blank');
    if (!w) { alert('Activa las ventanas emergentes para exportar a PDF.'); return; }
    w.document.write(html);
    w.document.close();
    w.focus();
    setTimeout(() => { w.print(); }, 600);
}

// ============================================================
// PANEL DE PESOS DE VOTACIÓN
// ============================================================

function toggleWeightsPanel() {
    const panel = document.getElementById('weightsPanelFull');
    const isHidden = panel.style.display === 'none' || panel.style.display === '';
    panel.style.display = isHidden ? 'block' : 'none';
    if (isHidden) {
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Sincronizar sliders con pesos globales actuales
        document.getElementById('sliderPhysical').value = Math.round(globalWeights.physical       * 100);
        document.getElementById('sliderDT').value       = Math.round(globalWeights.decision_tree  * 100);
        document.getElementById('sliderTemplate').value = Math.round(globalWeights.template        * 100);
        document.getElementById('sliderKNN').value      = Math.round(globalWeights.knn             * 100);
        document.getElementById('sliderCNN1D').value    = Math.round(globalWeights.cnn_1d          * 100);
        document.getElementById('sliderCNN2D').value    = Math.round(globalWeights.cnn_2d          * 100);
        refreshWeightsUI();
    }
}

function onWeightSlider() {
    refreshWeightsUI();
}

function refreshWeightsUI() {
    const p    = parseInt(document.getElementById('sliderPhysical').value) / 100;
    const dt   = parseInt(document.getElementById('sliderDT').value)       / 100;
    const t    = parseInt(document.getElementById('sliderTemplate').value) / 100;
    const knn  = parseInt(document.getElementById('sliderKNN').value)      / 100;
    const c1d  = parseInt(document.getElementById('sliderCNN1D').value)    / 100;
    const c2d  = parseInt(document.getElementById('sliderCNN2D').value)    / 100;
    const total = p + dt + t + knn + c1d + c2d;

    // Badges con valor actual
    document.getElementById('badgePhysical').textContent  = p.toFixed(2);
    document.getElementById('badgeDT').textContent        = dt.toFixed(2);
    document.getElementById('badgeTemplate').textContent  = t.toFixed(2);
    document.getElementById('badgeKNN').textContent       = knn.toFixed(2);
    document.getElementById('badgeCNN1D').textContent     = c1d.toFixed(2);
    document.getElementById('badgeCNN2D').textContent     = c2d.toFixed(2);

    // Barras de progreso individuales (relativas a 1.0)
    document.getElementById('barPhysical').style.width  = (p   * 100).toFixed(1) + '%';
    document.getElementById('barDT').style.width        = (dt  * 100).toFixed(1) + '%';
    document.getElementById('barTemplate').style.width  = (t   * 100).toFixed(1) + '%';
    document.getElementById('barKNN').style.width       = (knn * 100).toFixed(1) + '%';
    document.getElementById('barCNN1D').style.width     = (c1d * 100).toFixed(1) + '%';
    document.getElementById('barCNN2D').style.width     = (c2d * 100).toFixed(1) + '%';

    // Suma total
    document.getElementById('wm-total-value').textContent = total.toFixed(2);
    const valid = Math.abs(total - 1.0) < 0.011;
    document.getElementById('wm-total-ok').style.display   = valid ? 'inline' : 'none';
    document.getElementById('wm-total-warn').style.display = valid ? 'none'   : 'inline';
    document.getElementById('btnApplyWeights').disabled = !valid;

    // Preview proporcional (% del total)
    const safeTotal = total > 0 ? total : 1;
    document.getElementById('previewPhysical').style.width = ((p   / safeTotal) * 100).toFixed(1) + '%';
    document.getElementById('previewDT').style.width       = ((dt  / safeTotal) * 100).toFixed(1) + '%';
    document.getElementById('previewTemplate').style.width = ((t   / safeTotal) * 100).toFixed(1) + '%';
    document.getElementById('previewKNN').style.width      = ((knn / safeTotal) * 100).toFixed(1) + '%';
    document.getElementById('previewCNN1D').style.width    = ((c1d / safeTotal) * 100).toFixed(1) + '%';
    document.getElementById('previewCNN2D').style.width    = ((c2d / safeTotal) * 100).toFixed(1) + '%';
}

function applyWeights() {
    const p   = parseInt(document.getElementById('sliderPhysical').value) / 100;
    const dt  = parseInt(document.getElementById('sliderDT').value)       / 100;
    const t   = parseInt(document.getElementById('sliderTemplate').value) / 100;
    const knn = parseInt(document.getElementById('sliderKNN').value)      / 100;
    const c1d = parseInt(document.getElementById('sliderCNN1D').value)    / 100;
    const c2d = parseInt(document.getElementById('sliderCNN2D').value)    / 100;

    globalWeights.physical      = p;
    globalWeights.decision_tree = dt;
    globalWeights.template      = t;
    globalWeights.knn           = knn;
    globalWeights.cnn_1d        = c1d;
    globalWeights.cnn_2d        = c2d;

    const btn = document.getElementById('btnApplyWeights');
    btn.textContent = '✓ Aplicado';
    btn.style.background = '#27ae60';
    setTimeout(() => {
        btn.textContent = '✓ Aplicar Pesos';
        btn.style.background = '';
    }, 1800);
}

function resetWeights() {
    document.getElementById('sliderPhysical').value = 10;
    document.getElementById('sliderDT').value       = 40;
    document.getElementById('sliderTemplate').value = 10;
    document.getElementById('sliderKNN').value      = 20;
    document.getElementById('sliderCNN1D').value    = 20;
    document.getElementById('sliderCNN2D').value    = 0;
    refreshWeightsUI();
}

/* =========================================================
   ML CODE COPY — botón "Copiar" en los bloques de código
   ========================================================= */
function mlCopyCode(btn) {
    const pre  = btn.closest('.ml-code-block').querySelector('.ml-code-content');
    const text = pre ? pre.innerText : '';
    navigator.clipboard.writeText(text).then(function () {
        const orig = btn.textContent;
        btn.textContent = '✅ Copiado';
        btn.style.background = '#16a34a';
        setTimeout(function () {
            btn.textContent = orig;
            btn.style.background = '';
        }, 2000);
    }).catch(function () {
        btn.textContent = '⚠️ Error';
        setTimeout(function () { btn.textContent = '📋 Copiar'; }, 2000);
    });
}

/* =========================================================
   TOOLTIP FLOTANTE GLOBAL
   Un único div fixed anclado al body, siempre por encima de
   cualquier contenedor. Evita recortes por overflow/z-index.
   ========================================================= */
(function initFloatingTooltip() {
    const MARGIN = 12;

    // Crear el div flotante una sola vez
    const ft = document.createElement('div');
    ft.id = 'floatingTooltip';
    document.body.appendChild(ft);

    document.addEventListener('mouseenter', function(e) {
        const icon = e.target.closest('.help-icon');
        if (!icon) return;
        const src = icon.querySelector('.tooltip');
        if (!src) return;

        // Copiar contenido
        ft.innerHTML = src.innerHTML;
        ft.classList.remove('ft-below');
        ft.classList.remove('ft-above');

        // Posición provisional para medir alto del tooltip
        ft.style.left       = '-9999px';
        ft.style.top        = '0px';
        ft.style.opacity    = '0';
        ft.style.visibility = 'visible';

        requestAnimationFrame(function() {
            const iconRect = icon.getBoundingClientRect();
            const ttH      = ft.offsetHeight;
            const ttW      = ft.offsetWidth;
            const vw       = window.innerWidth;
            const vh       = window.innerHeight;

            // Vertical: arriba del icono si cabe, si no abajo
            let top;
            if (iconRect.top - ttH - 10 >= MARGIN) {
                top = iconRect.top - ttH - 10;
                ft.classList.add('ft-above');
                ft.classList.remove('ft-below');
            } else {
                top = iconRect.bottom + 10;
                ft.classList.add('ft-below');
                ft.classList.remove('ft-above');
            }

            // Horizontal: centrado en el icono, con clamp
            let left = iconRect.left + iconRect.width / 2 - ttW / 2;
            if (left < MARGIN) left = MARGIN;
            if (left + ttW > vw - MARGIN) left = vw - MARGIN - ttW;

            ft.style.left       = left + 'px';
            ft.style.top        = top  + 'px';
            ft.style.opacity    = '1';
            ft.style.visibility = 'visible';
        });
    }, true);

    document.addEventListener('mouseleave', function(e) {
        const icon = e.target.closest('.help-icon');
        if (!icon) return;
        ft.style.opacity    = '0';
        ft.style.visibility = 'hidden';
    }, true);
})();

// ═══════════════════════════════════════════════════════════
// EXTRACTOR DE METADATOS FITS → TXT (LOTE)
// ═══════════════════════════════════════════════════════════

function setupFitsExtractor() {
    const dropZone   = document.getElementById('dropZoneFits');
    const fileInput  = document.getElementById('fileInputFits');
    const btnSelect  = document.getElementById('btnSelectFits');
    const filesList  = document.getElementById('fitsFilesList');
    const actions    = document.getElementById('fitsActions');
    const processBtn = document.getElementById('processFitsBtn');
    const clearBtn   = document.getElementById('clearFitsBtn');
    if (!dropZone) return;

    let selectedFiles = [];

    btnSelect.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => addFiles(Array.from(e.target.files)));

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        addFiles(Array.from(e.dataTransfer.files).filter(f => /\.(fits?|fit)$/i.test(f.name)));
    });

    function addFiles(files) {
        files.forEach(f => { if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f); });
        renderFilesList();
    }

    function renderFilesList() {
        filesList.innerHTML = selectedFiles.map((f, i) =>
            `<div class="batch-file-item">
                <span class="file-icon">🔭</span>
                <span class="file-name">${f.name}</span>
                <span class="file-size">${(f.size / 1024 / 1024).toFixed(1)} MB</span>
                <button class="file-remove" onclick="fitsRemoveFile(${i})">✕</button>
            </div>`
        ).join('');
        actions.style.display = selectedFiles.length ? 'flex' : 'none';
        document.getElementById('fitsResults').style.display = 'none';
    }

    window.fitsRemoveFile = idx => { selectedFiles.splice(idx, 1); renderFilesList(); };

    clearBtn.addEventListener('click', () => {
        selectedFiles = [];
        fileInput.value = '';
        renderFilesList();
        document.getElementById('fitsProgress').style.display = 'none';
        document.getElementById('fitsResults').style.display  = 'none';
    });

    // ── Procesar de a UNO para evitar límites de tamaño ──────────────────────
    processBtn.addEventListener('click', async () => {
        if (!selectedFiles.length) return;

        const total    = selectedFiles.length;
        const prog     = document.getElementById('fitsProgress');
        const progFill = document.getElementById('fitsProgressFill');
        const progText = document.getElementById('fitsProgressText');
        const titleEl  = document.getElementById('fitsProgressTitle');
        const logEl    = document.getElementById('fitsLog');

        // ── Pedir carpeta de destino UNA sola vez antes de empezar ────────────
        let dirHandle = null;
        const usarCarpeta = 'showDirectoryPicker' in window;
        if (usarCarpeta) {
            try {
                titleEl.textContent = 'Seleccioná la carpeta donde guardar los TXT…';
                prog.style.display  = 'block';
                dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
            } catch (err) {
                if (err.name === 'AbortError') { prog.style.display = 'none'; return; }
                // Si falla el picker continuamos sin él (fallback al final)
                dirHandle = null;
            }
        }

        prog.style.display = 'block';
        document.getElementById('fitsResults').style.display = 'none';
        logEl.innerHTML = '';
        progFill.style.width = '0%';

        const resultados  = [];
        const txtFallback = [];   // solo se usa si no hay dirHandle
        let okCount = 0, errCount = 0;

        for (let i = 0; i < total; i++) {
            const f = selectedFiles[i];
            titleEl.textContent  = `Procesando ${i + 1} / ${total}: ${f.name}`;
            progText.textContent = `${i} / ${total} procesados`;
            progFill.style.width = `${Math.round((i / total) * 100)}%`;

            const fd = new FormData();
            fd.append('file', f);

            try {
                const resp = await fetch('/fits_extract_one', { method: 'POST', body: fd });
                const data = await resp.json();
                resultados.push(data);

                if (data.ok) {
                    okCount++;

                    if (dirHandle) {
                        // Guardar directamente en la carpeta elegida
                        try {
                            const fh = await dirHandle.getFileHandle(data.nombre_out, { create: true });
                            const wr = await fh.createWritable();
                            await wr.write(data.txt_content);
                            await wr.close();
                            logEl.innerHTML += `<div style="color:#4ade80;font-size:0.78rem;padding:1px 0;">
                                ✅ ${f.name} → <b>${data.nombre_out}</b>
                                (${data.rango_lambda}, ${data.n_puntos} pts) — guardado</div>`;
                        } catch (we) {
                            logEl.innerHTML += `<div style="color:#fbbf24;font-size:0.78rem;padding:1px 0;">
                                ⚠️ ${data.nombre_out}: no se pudo escribir — ${we.message}</div>`;
                            txtFallback.push({ nombre: data.nombre_out, contenido: data.txt_content });
                        }
                    } else {
                        // Sin picker: acumular para descarga al final
                        txtFallback.push({ nombre: data.nombre_out, contenido: data.txt_content });
                        logEl.innerHTML += `<div style="color:#4ade80;font-size:0.78rem;padding:1px 0;">
                            ✅ ${f.name} → <b>${data.nombre_out}</b>
                            (${data.rango_lambda}, ${data.n_puntos} pts)</div>`;
                    }
                } else {
                    errCount++;
                    logEl.innerHTML += `<div style="color:#f87171;font-size:0.78rem;padding:1px 0;">
                        ❌ ${f.name}: ${data.error}</div>`;
                }
            } catch (err) {
                errCount++;
                resultados.push({ ok: false, original: f.name, error: err.message });
                logEl.innerHTML += `<div style="color:#f87171;font-size:0.78rem;padding:1px 0;">
                    ❌ ${f.name}: Error de red — ${err.message}</div>`;
            }

            logEl.scrollTop = logEl.scrollHeight;
        }

        progFill.style.width = '100%';
        titleEl.textContent  = dirHandle
            ? `✅ Completado — archivos guardados en carpeta seleccionada`
            : `✅ Procesamiento completado`;
        progText.textContent = `${okCount} convertidos, ${errCount} omitidos`;

        // Contadores
        document.getElementById('fitsOkCount').textContent  = okCount;
        document.getElementById('fitsErrCount').textContent = errCount;

        // Tabla resumen
        document.getElementById('fitsResultsTable').innerHTML = `
            <table class="all-lines-table" style="margin-top:0.8rem;">
                <thead><tr>
                    <th>Original</th><th>Objeto</th><th>SPTYPE</th>
                    <th>Salida</th><th>Rango λ</th><th>Puntos</th><th>Estado</th>
                </tr></thead>
                <tbody>
                ${resultados.map(r => `<tr>
                    <td class="fits-key">${r.original || ''}</td>
                    <td>${r.objeto || '—'}</td>
                    <td><b>${r.sptype || '—'}</b></td>
                    <td class="fits-val">${r.nombre_out || '—'}</td>
                    <td>${r.rango_lambda || '—'}</td>
                    <td>${r.n_puntos || '—'}</td>
                    <td style="color:${r.ok ? '#4ade80' : '#f87171'}">${r.ok ? '✅ OK' : '❌ ' + (r.error||'')}</td>
                </tr>`).join('')}
                </tbody>
            </table>`;

        document.getElementById('fitsResults').style.display = 'block';

        // Botón TXT: solo relevante si NO se usó picker (o si hubo errores de escritura)
        const btnTxt = document.getElementById('downloadFitsTxtZip');
        if (dirHandle && txtFallback.length === 0) {
            // Todo guardado en carpeta → ocultar botón de descarga
            btnTxt.style.display = 'none';
        } else {
            btnTxt.style.display = '';
            btnTxt.textContent   = dirHandle
                ? `📥 Reintentar ${txtFallback.length} TXT no guardados`
                : `📂 Guardar TXT en carpeta...`;
            btnTxt.onclick = async () => {
                if (!txtFallback.length) { alert('No hay archivos pendientes.'); return; }
                if (usarCarpeta) {
                    try {
                        const dh = await window.showDirectoryPicker({ mode: 'readwrite' });
                        for (const arch of txtFallback) {
                            const fh = await dh.getFileHandle(arch.nombre, { create: true });
                            const wr = await fh.createWritable();
                            await wr.write(arch.contenido);
                            await wr.close();
                        }
                        alert(`✅ ${txtFallback.length} archivos guardados.`);
                    } catch (err) {
                        if (err.name !== 'AbortError') _fitsFallbackDownload(txtFallback);
                    }
                } else {
                    _fitsFallbackDownload(txtFallback);
                }
            };
        }

        // CSV metadatos
        const metaCSV = ['original,objeto,sptype,nombre_salida,rango_lambda,n_puntos,ok,error']
            .concat(resultados.map(r =>
                `"${r.original||''}","${r.objeto||''}","${r.sptype||''}","${r.nombre_out||''}",` +
                `"${r.rango_lambda||''}",${r.n_puntos||0},${r.ok},"${r.error||''}"`
            )).join('\n');

        document.getElementById('downloadFitsMeta').onclick = () => {
            const blob = new Blob([metaCSV], { type: 'text/csv;charset=utf-8;' });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href = url; a.download = 'metadatos_fits.csv'; a.click();
            URL.revokeObjectURL(url);
        };
    });

    function _fitsFallbackDownload(archivos) {
        archivos.forEach((f, idx) => {
            setTimeout(() => {
                const blob = new Blob([f.contenido], { type: 'text/plain;charset=utf-8;' });
                const url  = URL.createObjectURL(blob);
                const a    = document.createElement('a');
                a.href = url; a.download = f.nombre; a.click();
                URL.revokeObjectURL(url);
            }, idx * 300);
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NORMALIZACIÓN DE ESPECTROS CRUDOS
// ═══════════════════════════════════════════════════════════════════════════

let _normTextoDescarga = null;  // Guarda el texto para el botón de descarga
let _normSpectrumId    = null;  // ID del espectro en caché del servidor
let _normSmoothTimer   = null;  // Timer para debounce del slider

/** Verifica el estado del módulo y actualiza la barra de estado */
function normVerificarEstado() {
    fetch('/normalizacion_estado')
        .then(r => r.json())
        .then(d => {
            const bar = document.getElementById('norm-status-bar');
            if (!d.disponible) {
                bar.textContent = '❌ Módulo de normalización no encontrado. Verifica la ruta de instalación.';
                bar.style.borderLeftColor = '#ff4444';
                document.getElementById('norm-btn-cargar').disabled = true;
            } else if (d.modelo_cargado) {
                bar.textContent = '✅ Modelo de normalización cargado y listo.';
                bar.style.borderLeftColor = '#44bb44';
            } else {
                bar.textContent = '⚠️ Módulo disponible. Haz clic en "Cargar modelo" para inicializarlo.';
                bar.style.borderLeftColor = '#ffaa00';
            }
        })
        .catch(() => {
            document.getElementById('norm-status-bar').textContent =
                '⚠️ No se pudo contactar con el servidor.';
        });
}

/** Carga (o recarga) el modelo de normalización en el servidor */
function normCargarModelo() {
    const modelo = document.querySelector('input[name="norm-modelo"]:checked').value;
    const bar    = document.getElementById('norm-status-bar');
    const btn    = document.getElementById('norm-btn-cargar');

    bar.textContent = '⏳ Cargando modelo… esto puede tardar ~10 segundos.';
    bar.style.borderLeftColor = '#4a9eff';
    btn.disabled = true;

    fetch('/normalizacion_cargar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelo })
    })
    .then(r => r.json())
    .then(d => {
        if (d.success) {
            bar.textContent = '✅ Modelo cargado correctamente. Ya puedes normalizar espectros.';
            bar.style.borderLeftColor = '#44bb44';
        } else {
            bar.textContent = '❌ Error al cargar: ' + (d.error || 'desconocido');
            bar.style.borderLeftColor = '#ff4444';
        }
    })
    .catch(e => {
        bar.textContent = '❌ Error de red: ' + e.message;
        bar.style.borderLeftColor = '#ff4444';
    })
    .finally(() => { btn.disabled = false; });
}

/** Maneja drag-and-drop en el área de carga */
function normDropFile(event) {
    event.preventDefault();
    document.getElementById('norm-drop-area').style.borderColor = '#4a9eff44';
    const file = event.dataTransfer.files[0];
    if (file) _normSetArchivo(file);
}

/** Maneja la selección de archivo desde el input */
function normArchivoSeleccionado(input) {
    if (input.files[0]) _normSetArchivo(input.files[0]);
}

function _normSetArchivo(file) {
    document.getElementById('norm-filename').textContent = file.name;
    document.getElementById('norm-btn-procesar').disabled = false;
    document.getElementById('norm-resultados').style.display = 'none';
    _normTextoDescarga = null;
    _normSpectrumId    = null;
    // Resetear slider al default
    const slider = document.getElementById('norm-smooth-slider');
    if (slider) { slider.value = 20; }
    const val = document.getElementById('norm-smooth-val');
    if (val) val.textContent = '1.00';
    // Guardar referencia al archivo
    window._normArchivoActual = file;
}

/** Envía el archivo al servidor y muestra los resultados */
function normProcesar() {
    const file = window._normArchivoActual;
    if (!file) return;

    const bar = document.getElementById('norm-status-bar');
    const btn = document.getElementById('norm-btn-procesar');

    bar.textContent = '⏳ Normalizando espectro…';
    bar.style.borderLeftColor = '#4a9eff';
    btn.disabled = true;
    document.getElementById('norm-resultados').style.display = 'none';

    const fd = new FormData();
    fd.append('archivo', file);

    fetch('/normalizacion_procesar', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                bar.textContent = '❌ ' + d.error;
                bar.style.borderLeftColor = '#ff4444';
                return;
            }
            // Mostrar imágenes
            document.getElementById('norm-img-original').src =
                'data:image/png;base64,' + d.imagen_original;
            document.getElementById('norm-img-normalizado').src =
                'data:image/png;base64,' + d.imagen_normalizado;

            // Info del espectro
            document.getElementById('norm-info').innerHTML =
                `<b>${d.n_puntos}</b> puntos &nbsp;|&nbsp; rango: <b>${d.rango_lambda}</b>`;

            // Guardar texto para descarga y el ID de caché
            _normTextoDescarga = d.texto_descarga;
            _normSpectrumId    = d.spectrum_id;

            // Resetear slider a 1.0
            const slider = document.getElementById('norm-smooth-slider');
            if (slider) slider.value = 20;
            const val = document.getElementById('norm-smooth-val');
            if (val) val.textContent = '1.00';
            const status = document.getElementById('norm-smooth-status');
            if (status) status.textContent = '';

            document.getElementById('norm-resultados').style.display = 'block';
            bar.textContent = '✅ Normalización completada. Ajusta el factor de suavizado si es necesario.';
            bar.style.borderLeftColor = '#44bb44';
        })
        .catch(e => {
            bar.textContent = '❌ Error de red: ' + e.message;
            bar.style.borderLeftColor = '#ff4444';
        })
        .finally(() => { btn.disabled = false; });
}

/** Descarga el espectro normalizado como archivo .txt */
function normDescargar() {
    if (!_normTextoDescarga) return;
    const nombre  = (window._normArchivoActual?.name || 'espectro').replace(/\.[^.]+$/, '');
    const blob    = new Blob([_normTextoDescarga], { type: 'text/plain;charset=utf-8;' });
    const url     = URL.createObjectURL(blob);
    const a       = document.createElement('a');
    a.href        = url;
    a.download    = nombre + '_normalizado.txt';
    a.click();
    URL.revokeObjectURL(url);
}

/** Convierte posición del slider (1-200) a smooth_factor (0.05-10.0) y actualiza el label */
function normSmoothInput(sliderPos) {
    // Escala logarítmica: pos=20 → factor=1.0, rango 0.05–10
    const factor = Math.exp((sliderPos - 20) / 20 * Math.log(10));
    const clamped = Math.max(0.05, Math.min(factor, 10.0));
    const val = document.getElementById('norm-smooth-val');
    if (val) val.textContent = clamped.toFixed(2);
}

/** Se dispara al soltar el slider (debounce de 300 ms) */
function normSmoothAplicar() {
    if (!_normSpectrumId) return;
    clearTimeout(_normSmoothTimer);
    _normSmoothTimer = setTimeout(() => {
        const sliderPos = parseInt(document.getElementById('norm-smooth-slider').value, 10);
        const factor    = Math.exp((sliderPos - 20) / 20 * Math.log(10));
        const clamped   = Math.max(0.05, Math.min(factor, 10.0));

        const status = document.getElementById('norm-smooth-status');
        if (status) status.textContent = '⏳ Recalculando continuo…';

        fetch('/normalizacion_ajustar_smooth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ spectrum_id: _normSpectrumId, smooth_factor: clamped })
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                if (status) status.textContent = '❌ ' + d.error;
                return;
            }
            document.getElementById('norm-img-original').src =
                'data:image/png;base64,' + d.imagen_original;
            document.getElementById('norm-img-normalizado').src =
                'data:image/png;base64,' + d.imagen_normalizado;
            _normTextoDescarga = d.texto_descarga;
            if (status) status.textContent = '✅ Continuo actualizado.';
        })
        .catch(e => {
            if (status) status.textContent = '❌ Error de red: ' + e.message;
        });
    }, 300);
}

// Inicializar verificación de estado cuando se active la pestaña de normalización
document.addEventListener('DOMContentLoaded', () => {
    // Conectar al switch de pestañas existente
    document.querySelectorAll('.tab-btn[data-tab="normalizacion"]').forEach(btn => {
        btn.addEventListener('click', () => {
            normVerificarEstado();
        });
    });
});

// ============================================================================
// MÉTRICAS GRÁFICAS — Redes Neuronales (Chart.js)
// ============================================================================

let _chartAcc  = null;
let _chartLoss = null;
let _currentMetricsModel = 'cnn_1d';
let _metricsData = null;  // Datos cargados desde /neural_history

function switchMetricsModel(model, btn) {
    _currentMetricsModel = model;
    document.querySelectorAll('.nm-tab').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    if (_metricsData) _renderMetrics(_metricsData);
}

async function loadTrainingHistory(trainedModelType) {
    const panel = document.getElementById('nnMetricsPanel');
    panel.style.display = 'block';

    // Si se pasó el tipo recién entrenado, seleccionar esa pestaña automáticamente
    // (evita el bug donde el usuario tenía KNN seleccionado y el CNN aparece en blanco)
    if (trainedModelType && trainedModelType !== _currentMetricsModel) {
        const btn = document.querySelector(`.nm-tab[data-model="${trainedModelType}"]`);
        switchMetricsModel(trainedModelType, btn);
    }

    try {
        const res  = await fetch('/neural_history');
        const data = await res.json();
        if (!data.success) { _showNoData('Error al cargar métricas.'); return; }
        _metricsData = data;
        _renderMetrics(data);
        // Scroll suave DESPUÉS de renderizar (evita medir canvas durante scroll)
        setTimeout(() => panel.scrollIntoView({ behavior: 'smooth' }), 120);
    } catch (e) {
        _showNoData('No se pudo contactar con el servidor.');
    }
}

function _showNoData(msg) {
    document.getElementById('nmNoData').style.display = 'block';
    document.getElementById('nmNoData').textContent = msg;
    document.getElementById('nmChartsSection').style.display = 'none';
    document.getElementById('nmConfusionSection').style.display = 'none';
    document.getElementById('nmPerClassSection').style.display = 'none';
}

function _renderMetrics(data) {
    const m   = data[_currentMetricsModel] || {};
    const noD = document.getElementById('nmNoData');
    const hasAny = m.history || m.confusion_matrix || m.per_class;

    noD.style.display = hasAny ? 'none' : 'block';
    if (!hasAny) { noD.textContent = 'Aún no hay métricas guardadas para este modelo. Entrena el modelo primero.'; return; }

    // Badge de accuracy
    const badge = document.getElementById('nmAccuracyBadge');
    badge.textContent = m.accuracy != null ? `Accuracy: ${m.accuracy}%` : '—';

    // Gráficas de épocas (solo CNN con historial)
    const chartsSection = document.getElementById('nmChartsSection');
    if (m.history) {
        chartsSection.style.display = 'block';
        _renderAccChart(m.history);
        _renderLossChart(m.history);
    } else {
        chartsSection.style.display = 'none';
    }

    // Matriz de confusión
    const confSection = document.getElementById('nmConfusionSection');
    if (m.confusion_matrix) {
        confSection.style.display = 'block';
        _renderConfusionMatrix(m.confusion_matrix, document.getElementById('nmConfusionMatrix'));
    } else {
        confSection.style.display = 'none';
    }

    // Métricas por clase
    const pcSection = document.getElementById('nmPerClassSection');
    if (m.per_class) {
        pcSection.style.display = 'block';
        _renderPerClass(m.per_class, document.getElementById('nmPerClassTable'));
    } else {
        pcSection.style.display = 'none';
    }
}

function _chartDefaults() {
    return {
        plugins: { legend: { labels: { color: '#a0aec0', font: { size: 11 } } } },
        scales: {
            x: { ticks: { color: '#718096' }, grid: { color: '#2d3748' } },
            y: { ticks: { color: '#718096' }, grid: { color: '#2d3748' } },
        },
        animation: { duration: 500 },
        responsive: true,
        maintainAspectRatio: false,
    };
}

function _renderAccChart(history) {
    const canvas = document.getElementById('chartAccuracy');
    if (!canvas) return;
    const epochs = history.accuracy.map((_, i) => i + 1);

    if (_chartAcc) {
        // Reusar el chart existente (evita destroy → canvas colapsa → blanco)
        _chartAcc.data.labels = epochs;
        _chartAcc.data.datasets[0].data = [...history.accuracy];
        _chartAcc.data.datasets[1].data = [...history.val_accuracy];
        // Restaurar responsive y animación para la vista estática final
        _chartAcc.options.responsive          = true;
        _chartAcc.options.animation           = { duration: 400 };
        _chartAcc.options.scales.y.min        = 0;
        _chartAcc.options.scales.y.max        = 1;
        _chartAcc.options.scales.y.ticks      = { color: '#718096', callback: v => (v * 100).toFixed(0) + '%' };
        _chartAcc.update();
    } else {
        // Primera vez: no hay live chart, crear desde cero
        _chartAcc = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    { label: 'Train Acc', data: history.accuracy,     borderColor: '#68d391', backgroundColor: 'rgba(104,211,145,0.15)', tension: 0.3, fill: true,  pointRadius: 3 },
                    { label: 'Val Acc',   data: history.val_accuracy, borderColor: '#63b3ed', backgroundColor: 'rgba(99,179,237,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 3 },
                ],
            },
            options: {
                ..._chartDefaults(),
                scales: { ..._chartDefaults().scales,
                    y: { ..._chartDefaults().scales.y, min: 0, max: 1,
                         ticks: { color: '#718096', callback: v => (v * 100).toFixed(0) + '%' } },
                },
            },
        });
    }
}

function _renderLossChart(history) {
    const canvas = document.getElementById('chartLoss');
    if (!canvas) return;
    const epochs = history.loss.map((_, i) => i + 1);

    if (_chartLoss) {
        _chartLoss.data.labels = epochs;
        _chartLoss.data.datasets[0].data = [...history.loss];
        _chartLoss.data.datasets[1].data = [...history.val_loss];
        _chartLoss.options.responsive = true;
        _chartLoss.options.animation  = { duration: 400 };
        _chartLoss.update();
    } else {
        _chartLoss = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    { label: 'Train Loss', data: history.loss,     borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,0.15)', tension: 0.3, fill: true,  pointRadius: 3 },
                    { label: 'Val Loss',   data: history.val_loss, borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 3 },
                ],
            },
            options: _chartDefaults(),
        });
    }
}

function _renderConfusionMatrix(cmData, container) {
    const { matrix, labels } = cmData;
    const maxVal = Math.max(...matrix.flat().filter(v => v > 0));

    let html = '<table class="nm-confusion-table"><thead><tr><th>Real \\ Pred</th>';
    labels.forEach(l => { html += `<th>${l}</th>`; });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, ri) => {
        html += `<tr><th>${labels[ri]}</th>`;
        row.forEach((val, ci) => {
            let cls = 'nm-cm-off';
            if (ri === ci) { cls = 'nm-cm-diag'; }
            else if (val > 0) {
                const ratio = val / maxVal;
                cls = ratio < 0.2 ? 'nm-cm-low' : ratio < 0.5 ? 'nm-cm-mid' : 'nm-cm-high';
            }
            html += `<td class="${cls}">${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function _renderPerClass(perClass, container) {
    const classes = Object.keys(perClass);
    let html = `<table class="nm-perclass-table">
        <thead><tr>
            <th>Clase</th><th>Precisión</th><th>Recall</th><th>F1-Score</th><th>Soporte</th>
        </tr></thead><tbody>`;

    classes.forEach(cls => {
        const m = perClass[cls];
        const f1Pct = Math.round(m.f1 * 100);
        html += `<tr>
            <td><b>${cls}</b></td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.recall    * 100).toFixed(1)}%</td>
            <td>
                ${(m.f1 * 100).toFixed(1)}%
                <span class="nm-f1-bar" style="width:${f1Pct}px"></span>
            </td>
            <td>${m.support}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ── Actualización en tiempo real durante entrenamiento CNN ──────────────────
// Se llama desde trainNeuralModel() cuando llega un evento type='epoch'
let _liveAccData  = [];
let _liveLossData = [];

function _initLiveCharts() {
    _liveAccData  = [];
    _liveLossData = [];

    // Mostrar el panel primero (necesita estar visible para que Chart.js mida bien)
    document.getElementById('nnMetricsPanel').style.display = 'block';
    document.getElementById('nmChartsSection').style.display = 'block';
    document.getElementById('nmConfusionSection').style.display = 'none';
    document.getElementById('nmPerClassSection').style.display = 'none';
    document.getElementById('nmNoData').style.display = 'none';

    const accCanvas  = document.getElementById('chartAccuracy');
    const lossCanvas = document.getElementById('chartLoss');
    if (!accCanvas || !lossCanvas) return;

    // Destruir charts previos (solo aquí, no en _renderAccChart)
    if (_chartAcc)  { _chartAcc.destroy();  _chartAcc  = null; }
    if (_chartLoss) { _chartLoss.destroy(); _chartLoss = null; }

    // El canvas está dentro de .nm-canvas-wrap { position:relative; height:200px }
    // → Chart.js lee 200px fijos del wrapper, NO el scroll-container que crece
    //   con cada appendLog(). Así el ResizeObserver nunca dispara por el log.
    // animation:false evita que la animación de creación choque con update('none').
    const liveOpts = {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#a0aec0', font: { size: 11 } } } },
        scales: {
            x: { ticks: { color: '#718096', font: { size: 10 } }, grid: { color: '#2d3748' } },
            y: { ticks: { color: '#718096', font: { size: 10 } }, grid: { color: '#2d3748' } },
        },
    };

    _chartAcc = new Chart(accCanvas.getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Train Acc', data: [], borderColor: '#68d391', backgroundColor: 'rgba(104,211,145,0.15)', tension: 0.3, fill: true,  pointRadius: 2 },
            { label: 'Val Acc',   data: [], borderColor: '#63b3ed', backgroundColor: 'rgba(99,179,237,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 2 },
        ]},
        options: { ...liveOpts,
            scales: { ...liveOpts.scales,
                y: { ...liveOpts.scales.y, min: 0, max: 1,
                     ticks: { color: '#718096', font: { size: 10 }, callback: v => (v*100).toFixed(0)+'%' } } } },
    });

    _chartLoss = new Chart(lossCanvas.getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Train Loss', data: [], borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,0.15)', tension: 0.3, fill: true,  pointRadius: 2 },
            { label: 'Val Loss',   data: [], borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 2 },
        ]},
        options: liveOpts,
    });
}

function _updateLiveCharts(epochData) {
    if (!_chartAcc || !_chartLoss) return;
    const ep = epochData.epoch;
    _chartAcc.data.labels.push(ep);
    _chartAcc.data.datasets[0].data.push(epochData.acc);
    _chartAcc.data.datasets[1].data.push(epochData.val_acc);
    _chartAcc.update('none');   // 'none' = sin animación por epoch

    _chartLoss.data.labels.push(ep);
    _chartLoss.data.datasets[0].data.push(epochData.loss);
    _chartLoss.data.datasets[1].data.push(epochData.val_loss);
    _chartLoss.update('none');
}

// ============================================================================
// MÉTRICAS — Panel completo en Herramientas (IDs prefijo hm-)
// ============================================================================

let _hmChartAcc   = null;
let _hmChartLoss  = null;
let _hmCurrentModel = 'decision_tree';
let _hmAllData = null;   // respuesta completa de /metrics_all

const _hmModelLabels = {
    decision_tree: '🌿 Árbol / RF / GB',
    knn:           '📍 KNN',
    cnn_1d:        '〰️ CNN 1D',
    cnn_2d:        '🖼️ CNN 2D',
};

function switchHmModel(model, btn) {
    _hmCurrentModel = model;
    document.querySelectorAll('[id^="hmTab_"]').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    if (_hmAllData) _renderHmMetrics(_hmAllData);
}

async function loadMetricsHerramientas() {
    try {
        const [mRes, hRes] = await Promise.all([
            fetch('/metrics_all'),
            fetch('/training_log'),
        ]);
        const mData = await mRes.json();
        const hData = await hRes.json();

        if (!mData.success) { _hmShowNoData('Error al cargar métricas.'); return; }
        _hmAllData = mData;
        _renderHmMetrics(mData);
        _renderHmHistory(hData.entries || []);
    } catch (e) {
        _hmShowNoData('No se pudo contactar con el servidor: ' + e.message);
    }
}

function _hmShowNoData(msg) {
    const el = document.getElementById('hmNoData');
    if (el) { el.style.display = 'block'; el.textContent = msg; }
    ['hmChartsSection','hmConfusionSection','hmPerClassSection'].forEach(id => {
        const s = document.getElementById(id); if (s) s.style.display = 'none';
    });
    const badge = document.getElementById('hmAccuracyBadge');
    if (badge) badge.textContent = '—';
}

function _renderHmMetrics(data) {
    const m   = (data.models || {})[_hmCurrentModel];
    const noD = document.getElementById('hmNoData');

    if (!m) {
        _hmShowNoData('Sin datos para este modelo. Entrénalo primero.');
        return;
    }
    noD.style.display = 'none';

    // Badge de accuracy
    const badge = document.getElementById('hmAccuracyBadge');
    if (badge) {
        badge.textContent = m.accuracy != null ? `Accuracy: ${m.accuracy}%` : '—';
        badge.style.color = m.accuracy >= 80 ? '#68d391' : m.accuracy >= 60 ? '#fbbf24' : '#fc8181';
    }

    // Info rápida del modelo
    const infoDiv = document.getElementById('hmModelInfo');
    if (infoDiv) {
        const extra = m.extra || {};
        const rows = [];
        if (m.accuracy_cv)     rows.push(`CV: ${m.accuracy_cv}% ±${m.accuracy_cv_std||0}%`);
        if (m.n_samples)       rows.push(`Muestras: ${m.n_samples}`);
        if (m.classes?.length) rows.push(`Clases: ${m.classes.join(', ')}`);
        if (extra.n_neighbors) rows.push(`K vecinos: ${extra.n_neighbors}`);
        if (extra.metric)      rows.push(`Métrica: ${extra.metric}`);
        if (extra.epochs_trained) rows.push(`Épocas: ${extra.epochs_trained}`);
        if (extra.learning_rate)  rows.push(`LR: ${extra.learning_rate}`);
        if (m.params?.max_depth)  rows.push(`Profundidad: ${m.params.max_depth}`);
        if (m.timestamp)       rows.push(`Entrenado: ${m.timestamp.slice(0,16).replace('T',' ')}`);
        infoDiv.innerHTML = rows.map(r => `<span class="hm-info-chip">${r}</span>`).join('');
    }

    // Gráficas por época (solo CNN)
    const chartsSection = document.getElementById('hmChartsSection');
    if (m.history) {
        chartsSection.style.display = 'block';
        _hmRenderAccChart(m.history);
        _hmRenderLossChart(m.history);
    } else {
        chartsSection.style.display = 'none';
    }

    // Matriz de confusión
    const confSection = document.getElementById('hmConfusionSection');
    if (m.confusion_matrix) {
        confSection.style.display = 'block';
        _renderConfusionMatrix(m.confusion_matrix, document.getElementById('hmConfusionMatrix'));
    } else {
        confSection.style.display = 'none';
    }

    // Métricas por clase
    const pcSection = document.getElementById('hmPerClassSection');
    if (m.per_class) {
        pcSection.style.display = 'block';
        _renderPerClass(m.per_class, document.getElementById('hmPerClassTable'));
    } else {
        pcSection.style.display = 'none';
    }

    // Curva de diagnóstico (KNN / DT / RF)
    const diagSection = document.getElementById('hmDiagSection');
    if (m.diag_curve) {
        diagSection.style.display = 'block';
        _renderHmDiagCurve(m.diag_curve, _hmCurrentModel);
    } else {
        diagSection.style.display = 'none';
    }
}

// ── Curva de diagnóstico ─────────────────────────────────────────────────────
let _hmChartDiag = null;

function _renderHmDiagCurve(curve, modelType) {
    const canvas = document.getElementById('hmChartDiag');
    if (!canvas) return;
    if (_hmChartDiag) { _hmChartDiag.destroy(); _hmChartDiag = null; }

    let xLabels, datasets, titleText, descText, xAxisLabel, yAxisLabel;

    if (modelType === 'knn' && curve.k_values) {
        // KNN: accuracy vs K
        titleText   = 'Accuracy vs Número de Vecinos (K)';
        descText    = 'A medida que aumenta K el modelo se generaliza más. El punto óptimo equilibra sesgo y varianza.';
        xAxisLabel  = 'Número de vecinos (K)';
        yAxisLabel  = 'Accuracy';
        xLabels     = curve.k_values;
        const optK  = curve.optimal_k;
        datasets = [
            {
                label: 'Train Accuracy',
                data: curve.train_acc.map(v => +(v * 100).toFixed(1)),
                borderColor: '#48bb78', backgroundColor: 'rgba(72,187,120,0.10)',
                tension: 0.3, fill: false, pointRadius: 3,
            },
            {
                label: 'Test Accuracy',
                data: curve.test_acc.map(v => +(v * 100).toFixed(1)),
                borderColor: '#63b3ed', backgroundColor: 'rgba(99,179,237,0.15)',
                tension: 0.3, fill: true, pointRadius: 3,
                pointBackgroundColor: curve.k_values.map(k => k === optK ? '#fbbf24' : 'transparent'),
                pointBorderColor:     curve.k_values.map(k => k === optK ? '#fbbf24' : '#63b3ed'),
                pointRadius:          curve.k_values.map(k => k === optK ? 7 : 3),
            },
        ];

    } else if ((modelType === 'decision_tree') && curve.depths) {
        // DT: accuracy vs profundidad
        titleText  = 'Accuracy vs Profundidad del Árbol';
        descText   = 'Cuando train accuracy >> test accuracy hay sobreajuste (overfitting). La profundidad óptima maximiza test accuracy.';
        xAxisLabel = 'Profundidad máxima';
        yAxisLabel = 'Accuracy (%)';
        xLabels    = curve.depths;
        const optD = curve.optimal_depth;
        datasets = [
            {
                label: 'Train Accuracy',
                data: curve.train_acc.map(v => +(v * 100).toFixed(1)),
                borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,0.10)',
                tension: 0.2, fill: false, pointRadius: 3,
            },
            {
                label: 'Test Accuracy',
                data: curve.test_acc.map(v => +(v * 100).toFixed(1)),
                borderColor: '#68d391', backgroundColor: 'rgba(104,211,145,0.15)',
                tension: 0.2, fill: true, pointRadius: 3,
                pointBackgroundColor: curve.depths.map(d => d === optD ? '#fbbf24' : 'transparent'),
                pointBorderColor:     curve.depths.map(d => d === optD ? '#fbbf24' : '#68d391'),
                pointRadius:          curve.depths.map(d => d === optD ? 7 : 3),
            },
        ];

    } else if (curve.n_estimators) {
        // RF: error vs n_estimators
        titleText  = 'Error vs Número de Árboles (Random Forest)';
        descText   = 'El error se estabiliza a medida que se agregan más árboles. Después del "codo" agregar más árboles aporta poco.';
        xAxisLabel = 'Número de árboles (n_estimators)';
        yAxisLabel = 'Error (1 − Accuracy)';
        xLabels    = curve.n_estimators;
        datasets = [
            {
                label: 'Train Error',
                data: curve.train_error.map(v => +(v * 100).toFixed(1)),
                borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,0.10)',
                tension: 0.3, fill: false, pointRadius: 3,
            },
            {
                label: 'Test Error',
                data: curve.test_error.map(v => +(v * 100).toFixed(1)),
                borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,0.15)',
                tension: 0.3, fill: true, pointRadius: 3,
            },
        ];
    } else {
        document.getElementById('hmDiagSection').style.display = 'none';
        return;
    }

    document.getElementById('hmDiagTitle').textContent = titleText;
    document.getElementById('hmDiagDesc').textContent  = descText;

    _hmChartDiag = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: xLabels, datasets },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: {
                legend: { labels: { color: '#a0aec0', font: { size: 11 } } },
                tooltip: { mode: 'index', intersect: false,
                    callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` } },
            },
            scales: {
                x: { title: { display: true, text: xAxisLabel, color: '#718096', font: { size: 11 } },
                     ticks: { color: '#718096', font: { size: 10 } },
                     grid:  { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: yAxisLabel, color: '#718096', font: { size: 11 } },
                     ticks: { color: '#718096', font: { size: 10 },
                              callback: v => v + '%' },
                     grid:  { color: 'rgba(255,255,255,0.06)' } },
            },
        },
    });
}

// ── Historial de entrenamientos ─────────────────────────────────────────────
function _renderHmHistory(entries) {
    const list  = document.getElementById('hmHistoryList');
    const count = document.getElementById('hmHistoryCount');
    if (!list) return;

    if (count) count.textContent = `${entries.length} entrenamiento${entries.length !== 1 ? 's' : ''} registrado${entries.length !== 1 ? 's' : ''}`;

    if (entries.length === 0) {
        list.innerHTML = '<p class="nm-nodata">Sin historial guardado aún.</p>';
        return;
    }

    const typeColor = {
        decision_tree: '#00b894', random_forest: '#00b894', gradient_boosting: '#00b894',
        knn: '#0984e3', cnn_1d: '#74b9ff', cnn_2d: '#a29bfe',
    };

    list.innerHTML = entries.map(e => {
        const acc   = e.accuracy_test != null ? (e.accuracy_test * 100).toFixed(1) : '—';
        const cv    = e.accuracy_cv_mean != null ? ` | CV: ${(e.accuracy_cv_mean * 100).toFixed(1)}%` : '';
        const ts    = (e.timestamp || '').slice(0, 16).replace('T', ' ');
        const color = typeColor[e.model_type] || '#94a3b8';
        const label = _hmModelLabels[e.model_type] || e.model_type;
        const cls   = (e.classes || []).join(', ');
        const paramStr = Object.entries(e.params || {})
            .filter(([,v]) => v != null)
            .map(([k,v]) => `${k}: ${v}`)
            .join(' · ') || '';
        return `
        <div class="hm-history-item">
            <div class="hm-hist-left">
                <span class="hm-hist-badge" style="background:${color}22; color:${color}; border-color:${color}44;">${label}</span>
                <span class="hm-hist-ts">🕐 ${ts}</span>
            </div>
            <div class="hm-hist-center">
                <span class="hm-hist-acc" style="color:${parseFloat(acc)>=80?'#68d391':parseFloat(acc)>=60?'#fbbf24':'#fc8181'}">
                    ${acc}%${cv}
                </span>
                <span class="hm-hist-classes">${cls}</span>
            </div>
            <div class="hm-hist-right">
                <span class="hm-hist-params">${paramStr}</span>
            </div>
        </div>`;
    }).join('');
}

function _hmRenderAccChart(history) {
    const canvas = document.getElementById('hmChartAccuracy');
    if (!canvas) return;
    if (_hmChartAcc) _hmChartAcc.destroy();
    const epochs = history.accuracy.map((_, i) => i + 1);
    _hmChartAcc = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: epochs, datasets: [
            { label: 'Train Acc', data: history.accuracy,     borderColor: '#68d391', backgroundColor: 'rgba(104,211,145,0.15)', tension: 0.3, fill: true,  pointRadius: 3 },
            { label: 'Val Acc',   data: history.val_accuracy, borderColor: '#63b3ed', backgroundColor: 'rgba(99,179,237,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 3 },
        ]},
        options: { ..._chartDefaults(),
            scales: { ..._chartDefaults().scales,
                y: { ..._chartDefaults().scales.y, min: 0, max: 1,
                     ticks: { color: '#718096', callback: v => (v*100).toFixed(0)+'%' } } } },
    });
}

function _hmRenderLossChart(history) {
    const canvas = document.getElementById('hmChartLoss');
    if (!canvas) return;
    if (_hmChartLoss) _hmChartLoss.destroy();
    const epochs = history.loss.map((_, i) => i + 1);
    _hmChartLoss = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: epochs, datasets: [
            { label: 'Train Loss', data: history.loss,     borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,0.15)', tension: 0.3, fill: true,  pointRadius: 3 },
            { label: 'Val Loss',   data: history.val_loss, borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,0.10)',  tension: 0.3, fill: false, borderDash: [5,3], pointRadius: 3 },
        ]},
        options: _chartDefaults(),
    });
}


// ════════════════════════════════════════════════════════════════════
// PANEL DE GESTIÓN DE DATASET
// ════════════════════════════════════════════════════════════════════

let _dmPage         = 1;
let _dmPerPage      = 25;
let _dmTotalPages   = 1;
let _dmAllFiles     = [];   // last inspector fetch
let _dmCleanupFiles = [];   // for cleanup tab
let _dmSvChart      = null; // spectrum viewer chart
let _dmDistChart    = null; // distribution bar chart
let _dmSelectedCleanup  = new Set();
let _dmSelectedRestore  = new Set();

// ── Sub-tab navigation ──────────────────────────────────────────────
function dmSetupSubnav() {
    document.querySelectorAll('.dm-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.dm-tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.dm-panel').forEach(p => p.style.display = 'none');
            btn.classList.add('active');
            const panel = document.getElementById(`dm-${btn.dataset.dmtab}`);
            if (panel) panel.style.display = '';
        });
    });
}

function dmCatalog() {
    return (document.getElementById('dmCatalogPath')?.value || 'data/elodie/').trim();
}

// ── Main load ───────────────────────────────────────────────────────
async function dmLoadPanel() {
    await dmLoadOverview();
    await dmLoadInspector(1);
    await dmLoadCleanup();
    dmPopulateClassFilter();
}

// ── 1. VISTA GENERAL ───────────────────────────────────────────────
async function dmLoadOverview() {
    const content = document.getElementById('dmOverviewContent');
    content.innerHTML = '<p class="dm-hint">Cargando…</p>';

    const res  = await fetch(`/dataset/overview?catalog=${encodeURIComponent(dmCatalog())}`);
    const data = await res.json();
    if (data.error) { content.innerHTML = `<p class="dm-msg err">${data.error}</p>`; return; }

    // Stats cards
    const most = data.most_represented ? `${data.most_represented.class} (${data.most_represented.count})` : '—';
    const least= data.least_represented? `${data.least_represented.class} (${data.least_represented.count})`: '—';

    content.innerHTML = `
        <div class="dm-overview-grid">
            <div class="dm-stat-card"><div class="dm-stat-val">${data.total}</div><div class="dm-stat-lbl">Espectros activos</div></div>
            <div class="dm-stat-card"><div class="dm-stat-val">${data.n_classes}</div><div class="dm-stat-lbl">Clases presentes</div></div>
            <div class="dm-stat-card"><div class="dm-stat-val">${data.total_discarded}</div><div class="dm-stat-lbl">Descartados</div></div>
            <div class="dm-stat-card"><div class="dm-stat-val">${data.total_augmented}</div><div class="dm-stat-lbl">Sintéticos</div></div>
            <div class="dm-stat-card"><div class="dm-stat-val" style="font-size:1rem;">${most}</div><div class="dm-stat-lbl">Clase más grande</div></div>
            <div class="dm-stat-card"><div class="dm-stat-val" style="font-size:1rem;color:#fc8181;">${least}</div><div class="dm-stat-lbl">Clase más pequeña</div></div>
        </div>
        <table class="dm-dist-table">
            <thead><tr><th>Clase</th><th>Espectros</th><th>%</th><th class="dm-dist-bar-cell">Distribución</th></tr></thead>
            <tbody>
                ${data.distribution.map(d => `
                <tr>
                    <td><span class="dm-badge">${d.class}</span></td>
                    <td>${d.count}</td>
                    <td>${d.pct}%</td>
                    <td class="dm-dist-bar-cell">
                        <div class="dm-dist-bar-wrap">
                            <div class="dm-dist-bar-fill" style="width:${d.pct}%"></div>
                        </div>
                    </td>
                </tr>`).join('')}
            </tbody>
        </table>`;

    // Bar chart
    const wrap = document.getElementById('dmDistChartWrap');
    wrap.style.display = '';
    const canvas = document.getElementById('dmDistChart');
    if (_dmDistChart) _dmDistChart.destroy();
    _dmDistChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels:   data.distribution.map(d => d.class),
            datasets: [{
                label: 'Espectros por clase',
                data:  data.distribution.map(d => d.count),
                backgroundColor: data.distribution.map((_, i) => `hsl(${200 + i*28},70%,55%)`),
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false },
                       tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} espectros` } } },
            scales: {
                x: { ticks: { color: '#a0aec0' }, grid: { color: '#1e293b' } },
                y: { ticks: { color: '#a0aec0', stepSize: 1 }, grid: { color: '#2d3748' }, beginAtZero: true },
            }
        }
    });
}

// ── 2. INSPECTOR ───────────────────────────────────────────────────
async function dmLoadInspector(page) {
    _dmPage = page;
    const catalog    = dmCatalog();
    const filterName = document.getElementById('dmFilterName')?.value || '';
    const filterCls  = document.getElementById('dmFilterClass')?.value || '';
    const inclDisc   = document.getElementById('dmIncDiscarded')?.checked ? 'true' : 'false';

    const url = `/dataset/spectra?catalog=${encodeURIComponent(catalog)}`
              + `&page=${page}&per_page=${_dmPerPage}`
              + `&filter_name=${encodeURIComponent(filterName)}`
              + `&filter_class=${encodeURIComponent(filterCls)}`
              + `&include_discarded=${inclDisc}`;

    const res  = await fetch(url);
    const data = await res.json();
    const wrap = document.getElementById('dmInspectorTable');
    if (data.error) { wrap.innerHTML = `<p class="dm-msg err">${data.error}</p>`; return; }

    _dmAllFiles   = data.files;
    _dmTotalPages = data.pages;

    wrap.innerHTML = `
        <table class="dm-table">
            <thead><tr>
                <th>Archivo</th><th>Tipo</th><th>Clase</th><th>Tamaño</th><th>Fecha</th><th>Estado</th>
            </tr></thead>
            <tbody>
                ${data.files.map(f => `
                <tr onclick="dmViewSpectrum('${escHtml(f.filename)}','${f.source}')"
                    title="Click para ver espectro">
                    <td>${escHtml(f.filename)}</td>
                    <td>${f.sptype}</td>
                    <td><span class="dm-badge">${f.class}</span></td>
                    <td>${f.size_kb} KB</td>
                    <td>${f.mtime}</td>
                    <td>${f.source === 'catalog' ? '' :
                         f.source === '_descartados' ? '<span class="dm-badge dm-badge-disc">descartado</span>' :
                         '<span class="dm-badge dm-badge-aug">sintético</span>'}</td>
                </tr>`).join('')}
            </tbody>
        </table>
        <p style="color:#718096;font-size:0.78rem;margin-top:0.4rem;">${data.total} resultado(s)</p>`;

    dmRenderPager(data.total, page, data.pages, 'dmLoadInspector');
}

function dmInspectorSearch() { dmLoadInspector(1); }

function dmRenderPager(total, page, pages, callbackFn) {
    const pager = document.getElementById('dmInspectorPager');
    if (!pager) return;
    if (pages <= 1) { pager.innerHTML = ''; return; }
    let html = `<button onclick="${callbackFn}(${page-1})" ${page<=1?'disabled':''}>‹ Ant.</button>`;
    const start = Math.max(1, page - 2), end = Math.min(pages, page + 2);
    for (let p = start; p <= end; p++) {
        html += `<button class="${p===page?'active':''}" onclick="${callbackFn}(${p})">${p}</button>`;
    }
    html += `<button onclick="${callbackFn}(${page+1})" ${page>=pages?'disabled':''}>Sig. ›</button>`;
    html += `<span class="dm-pager-info">Pág. ${page} / ${pages} · ${total} total</span>`;
    pager.innerHTML = html;
}

async function dmViewSpectrum(filename, source) {
    const viewer = document.getElementById('dmSpectrumViewer');
    const title  = document.getElementById('dmSvTitle');
    const meta   = document.getElementById('dmSvMeta');
    viewer.style.display = '';
    title.textContent = '⌛ Cargando…';
    meta.textContent  = '';

    const url = `/dataset/spectrum_data?catalog=${encodeURIComponent(dmCatalog())}`
              + `&file=${encodeURIComponent(filename)}&source=${encodeURIComponent(source)}`;
    const res  = await fetch(url);
    const data = await res.json();
    if (data.error) {
        title.textContent = 'Error: ' + data.error;
        return;
    }

    title.textContent = `${data.filename}  —  Tipo: ${data.sptype}`;
    meta.innerHTML = `λ ${data.wave_min.toFixed(1)} – ${data.wave_max.toFixed(1)} Å &nbsp;|&nbsp; ${data.n_points} puntos`;

    const canvas = document.getElementById('dmSvCanvas');
    if (_dmSvChart) _dmSvChart.destroy();
    _dmSvChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: data.wave,
                datasets: [{ label: data.sptype, data: data.flux,
                             borderColor: '#63b3ed', borderWidth: 1.2,
                             pointRadius: 0, tension: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false },
                       tooltip: { callbacks: { label: ctx => ` flux: ${ctx.parsed.y.toFixed(4)}`,
                                               title: ctx => `λ ${ctx[0].label} Å` } } },
            scales: {
                x: { ticks: { color: '#718096', maxTicksLimit: 8,
                              callback: (v,i,a) => data.wave[i] ? data.wave[i].toFixed(0)+'Å' : '' },
                     grid: { color: '#1e293b' } },
                y: { ticks: { color: '#718096' }, grid: { color: '#2d3748' } },
            }
        }
    });

    // Also pre-fill reclassify tab
    const reFile = document.getElementById('dmReFile');
    const reSrc  = document.getElementById('dmReSource');
    if (reFile) reFile.value = filename;
    if (reSrc)  reSrc.value  = source === '_descartados' ? 'discarded' : 'catalog';
}

function dmCloseSpectrumViewer() {
    document.getElementById('dmSpectrumViewer').style.display = 'none';
    if (_dmSvChart) { _dmSvChart.destroy(); _dmSvChart = null; }
}

function dmPopulateClassFilter() {
    fetch(`/dataset/overview?catalog=${encodeURIComponent(dmCatalog())}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            const sel = document.getElementById('dmFilterClass');
            if (!sel) return;
            const current = sel.value;
            sel.innerHTML = '<option value="">Todas las clases</option>'
                + data.classes_present.map(c => `<option value="${c}">${c}</option>`).join('');
            sel.value = current;
        });
}

// ── 3. LIMPIEZA ─────────────────────────────────────────────────────
async function dmLoadCleanup() {
    _dmSelectedCleanup.clear();
    _dmSelectedRestore.clear();
    const res  = await fetch(`/dataset/spectra?catalog=${encodeURIComponent(dmCatalog())}&per_page=1000`);
    const data = await res.json();
    if (data.error) return;
    _dmCleanupFiles = data.files;
    dmRenderCleanupTable(_dmCleanupFiles);
    await dmLoadDiscarded();
}

function dmCleanupFilter() {
    const maxN = parseInt(document.getElementById('dmCleanupMaxN')?.value || '5');
    // Group by class, keep only files whose class has <= maxN total spectra
    const classCounts = {};
    _dmCleanupFiles.forEach(f => { classCounts[f.class] = (classCounts[f.class]||0)+1; });
    const filtered = _dmCleanupFiles.filter(f => classCounts[f.class] <= maxN);
    dmRenderCleanupTable(filtered);
}

function dmCleanupShowAll() {
    dmRenderCleanupTable(_dmCleanupFiles);
}

function dmRenderCleanupTable(files) {
    _dmSelectedCleanup.clear();
    const wrap = document.getElementById('dmCleanupTable');
    const actions = document.getElementById('dmCleanupActions');
    if (!files.length) {
        wrap.innerHTML = '<p class="dm-hint">No hay espectros que mostrar.</p>';
        actions.style.display = 'none'; return;
    }
    wrap.innerHTML = `
        <table class="dm-table">
            <thead><tr>
                <th><input type="checkbox" id="dmCleanupCheckAll" onchange="dmCleanupToggleAll(this)"></th>
                <th>Archivo</th><th>Tipo</th><th>Clase</th><th>Tamaño</th>
            </tr></thead>
            <tbody>
                ${files.map(f => `
                <tr>
                    <td><input type="checkbox" class="dm-cleanup-chk" value="${escHtml(f.filename)}"
                               onchange="dmCleanupCheckChange(this)"></td>
                    <td onclick="dmViewSpectrum('${escHtml(f.filename)}','catalog')" style="cursor:pointer;color:#90cdf4;">
                        ${escHtml(f.filename)}</td>
                    <td>${f.sptype}</td>
                    <td><span class="dm-badge">${f.class}</span></td>
                    <td>${f.size_kb} KB</td>
                </tr>`).join('')}
            </tbody>
        </table>`;
    actions.style.display = 'none';
}

function dmCleanupToggleAll(chk) {
    document.querySelectorAll('.dm-cleanup-chk').forEach(c => {
        c.checked = chk.checked;
        if (chk.checked) _dmSelectedCleanup.add(c.value);
        else             _dmSelectedCleanup.delete(c.value);
    });
    dmUpdateCleanupActions();
}

function dmCleanupCheckChange(chk) {
    if (chk.checked) _dmSelectedCleanup.add(chk.value);
    else             _dmSelectedCleanup.delete(chk.value);
    dmUpdateCleanupActions();
}

function dmUpdateCleanupActions() {
    const n = _dmSelectedCleanup.size;
    document.getElementById('dmCleanupSelCount').textContent = `${n} seleccionado${n!==1?'s':''}`;
    document.getElementById('dmCleanupActions').style.display = n > 0 ? '' : 'none';
}

async function dmDiscardSelected() {
    if (!_dmSelectedCleanup.size) return;
    const files = [..._dmSelectedCleanup];
    const res   = await fetch('/dataset/discard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ catalog: dmCatalog(), files }),
    });
    const data = await res.json();
    const msg = data.error ? `❌ ${data.error}` :
        `✅ ${data.count} archivo(s) movido(s) a _descartados/${data.errors.length ? '. Errores: '+data.errors.join(', ') : ''}`;
    showToast(msg);
    await dmLoadCleanup();
    await dmLoadOverview();
}

async function dmLoadDiscarded() {
    _dmSelectedRestore.clear();
    const res  = await fetch(`/dataset/spectra?catalog=${encodeURIComponent(dmCatalog())}&include_discarded=true&per_page=1000`);
    const data = await res.json();
    if (data.error) return;
    const discarded = data.files.filter(f => f.source === '_descartados');
    const section   = document.getElementById('dmDiscardedSection');
    section.style.display = discarded.length ? '' : 'none';
    if (!discarded.length) return;

    document.getElementById('dmDiscardedTable').innerHTML = `
        <table class="dm-table">
            <thead><tr>
                <th><input type="checkbox" id="dmRestoreCheckAll" onchange="dmRestoreToggleAll(this)"></th>
                <th>Archivo</th><th>Tipo</th><th>Clase</th>
            </tr></thead>
            <tbody>
                ${discarded.map(f => `
                <tr>
                    <td><input type="checkbox" class="dm-restore-chk" value="${escHtml(f.filename)}"
                               onchange="dmRestoreCheckChange(this)"></td>
                    <td onclick="dmViewSpectrum('${escHtml(f.filename)}','_descartados')" style="cursor:pointer;color:#fc8181;">
                        ${escHtml(f.filename)}</td>
                    <td>${f.sptype}</td>
                    <td><span class="dm-badge">${f.class}</span></td>
                </tr>`).join('')}
            </tbody>
        </table>`;
}

function dmRestoreToggleAll(chk) {
    document.querySelectorAll('.dm-restore-chk').forEach(c => {
        c.checked = chk.checked;
        if (chk.checked) _dmSelectedRestore.add(c.value);
        else             _dmSelectedRestore.delete(c.value);
    });
    dmUpdateRestoreActions();
}

function dmRestoreCheckChange(chk) {
    if (chk.checked) _dmSelectedRestore.add(chk.value);
    else             _dmSelectedRestore.delete(chk.value);
    dmUpdateRestoreActions();
}

function dmUpdateRestoreActions() {
    const n = _dmSelectedRestore.size;
    document.getElementById('dmRestoreSelCount').textContent = `${n} seleccionado${n!==1?'s':''}`;
    document.getElementById('dmRestoreActions').style.display = n > 0 ? '' : 'none';
}

async function dmRestoreSelected() {
    if (!_dmSelectedRestore.size) return;
    const res  = await fetch('/dataset/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ catalog: dmCatalog(), files: [..._dmSelectedRestore] }),
    });
    const data = await res.json();
    const msg = data.error ? `❌ ${data.error}` :
        `✅ ${data.count} archivo(s) restaurado(s)${data.errors.length ? '. Errores: '+data.errors.join(', ') : ''}`;
    showToast(msg);
    await dmLoadCleanup();
    await dmLoadOverview();
}

// ── 4. RECLASIFICAR ─────────────────────────────────────────────────
async function dmDoReclassify() {
    const file      = document.getElementById('dmReFile')?.value.trim();
    const newSptype = document.getElementById('dmReNewSptype')?.value.trim();
    const source    = document.getElementById('dmReSource')?.value || 'catalog';
    const resultEl  = document.getElementById('dmReResult');

    if (!file || !newSptype) {
        resultEl.className = 'dm-msg err';
        resultEl.textContent = 'Completá el nombre de archivo y el nuevo tipo espectral.';
        return;
    }

    const res  = await fetch('/dataset/reclassify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ catalog: dmCatalog(), file, new_sptype: newSptype, source }),
    });
    const data = await res.json();

    if (data.error) {
        resultEl.className = 'dm-msg err';
        resultEl.textContent = `❌ ${data.error}`;
    } else {
        resultEl.className = 'dm-msg ok';
        resultEl.innerHTML = `✅ Renombrado:<br>
            <code>${escHtml(data.old_filename)}</code><br>→ <code>${escHtml(data.new_filename)}</code>`;
        document.getElementById('dmReFile').value = '';
        await dmLoadInspector(1);
        await dmLoadCleanup();
        await dmLoadOverview();
        dmPopulateClassFilter();
    }
}

// ── 5. COMPENSACIÓN ─────────────────────────────────────────────────
async function dmAugPreview() {
    const target   = document.getElementById('dmAugTarget')?.value;
    const maxRatio = document.getElementById('dmAugMaxRatio')?.value || 5;
    const noise    = document.getElementById('dmAugNoise')?.value || 0.015;

    const res  = await fetch('/dataset/augment_preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            catalog: dmCatalog(),
            target_per_class: target ? parseInt(target) : null,
            max_augment_ratio: parseInt(maxRatio),
        }),
    });
    const data = await res.json();
    const wrap = document.getElementById('dmAugPreviewTable');
    const applyBtn = document.getElementById('dmAugApplyBtn');

    if (data.error) {
        wrap.innerHTML = `<p class="dm-msg err">${data.error}</p>`;
        applyBtn.style.display = 'none'; return;
    }

    const rows = data.preview.map(p => `
        <tr>
            <td><span class="dm-badge">${p.class}</span></td>
            <td>${p.original}</td>
            <td class="${p.synthetic > 0 ? 'aug-warn' : 'aug-ok'}">${p.synthetic > 0 ? '+'+p.synthetic : '—'}</td>
            <td class="${p.capped ? 'aug-warn' : ''}">${p.total}${p.capped ? ' ⚠️' : ''}</td>
        </tr>`).join('');

    wrap.innerHTML = `
        <table class="dm-aug-table">
            <thead><tr><th>Clase</th><th>Original</th><th>Sintéticos</th><th>Total</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <div class="dm-aug-summary">
            Target por clase: <strong>${data.target_per_class}</strong>
            (mediana del dataset: ${data.median_count})<br>
            Sintéticos a generar: <strong>${data.total_synthetic}</strong> &nbsp;|&nbsp;
            Total tras augmentación: <strong>${data.total_after}</strong>
            ${data.preview.some(p=>p.capped) ? '<br>⚠️ Algunas clases limitadas por el multiplicador máximo.' : ''}
        </div>`;

    applyBtn.style.display = data.total_synthetic > 0 ? '' : 'none';
    const augRes = document.getElementById('dmAugResult');
    augRes.textContent = '';
    augRes.style.display = 'none';
}

async function dmAugApply() {
    const target    = document.getElementById('dmAugTarget')?.value;
    const maxRatio  = document.getElementById('dmAugMaxRatio')?.value || 5;
    const noise     = parseFloat(document.getElementById('dmAugNoise')?.value || 0.015);
    const resultEl  = document.getElementById('dmAugResult');
    const applyBtn  = document.getElementById('dmAugApplyBtn');
    const previewBtn = document.querySelector('[onclick="dmAugPreview()"]');

    // Deshabilitar botones y mostrar spinner
    applyBtn.disabled  = true;
    if (previewBtn) previewBtn.disabled = true;
    applyBtn.innerHTML = '<span class="dm-spinner"></span> Generando…';
    resultEl.className = 'dm-msg';
    resultEl.style.display = '';
    resultEl.textContent = '⌛ Leyendo espectros y generando variantes sintéticas…';

    try {
        const res  = await fetch('/dataset/augment_apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                catalog:           dmCatalog(),
                target_per_class:  target ? parseInt(target) : null,
                max_augment_ratio: parseInt(maxRatio),
                noise_factor:      noise,
            }),
        });
        const data = await res.json();
        if (data.error) {
            resultEl.className = 'dm-msg err';
            resultEl.textContent = `❌ ${data.error}`;
        } else {
            resultEl.className = 'dm-msg ok';
            resultEl.innerHTML = `✅ ${data.generated} espectros sintéticos generados en <code>${data.aug_dir}</code>`;
            await dmLoadOverview();
        }
    } catch (e) {
        resultEl.className = 'dm-msg err';
        resultEl.textContent = `❌ Error de red: ${e.message}`;
    } finally {
        applyBtn.disabled  = false;
        if (previewBtn) previewBtn.disabled = false;
        applyBtn.innerHTML = '⚡ Generar sintéticos';
    }
}

// ── 6. CALIDAD ──────────────────────────────────────────────────────
async function dmRunQuality() {
    const btn    = document.getElementById('dmQualityBtn');
    const res_el = document.getElementById('dmQualityResult');
    btn.disabled  = true;
    btn.innerHTML = '<span class="dm-spinner"></span> Analizando…';
    res_el.innerHTML = '';

    try {
        const res  = await fetch(`/dataset/quality?catalog=${encodeURIComponent(dmCatalog())}`);
        const data = await res.json();
        if (data.error) { res_el.innerHTML = `<p class="dm-msg err">${data.error}</p>`; return; }

        const rows = data.results.map(r => {
            const issues = r.issues.length ? r.issues.join(', ') : '—';
            const snr    = r.snr_est != null ? r.snr_est : '—';
            const cls    = r.ok ? 'dm-quality-ok' : (r.issues.some(i=>i.startsWith('error')) ? 'dm-quality-err' : 'dm-quality-warn');
            return `<tr>
                <td>${r.ok ? '✅' : '⚠️'}</td>
                <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                    title="${escHtml(r.filename)}">${escHtml(r.filename)}</td>
                <td><span class="dm-badge">${r.class}</span></td>
                <td>${r.n_points || '—'}</td>
                <td>${snr}</td>
                <td class="${cls}">${issues}</td>
            </tr>`;
        }).join('');

        res_el.innerHTML = `
            <div class="dm-aug-summary" style="margin-bottom:0.8rem;">
                Total: ${data.total} &nbsp;|&nbsp;
                <span style="color:#48bb78;">✅ OK: ${data.ok}</span> &nbsp;|&nbsp;
                <span style="color:#f6ad55;">⚠️ Con problemas: ${data.with_issues}</span>
            </div>
            <table class="dm-quality-table">
                <thead><tr><th></th><th>Archivo</th><th>Clase</th><th>Puntos</th><th>SNR est.</th><th>Problemas</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>`;
    } catch(e) {
        res_el.innerHTML = `<p class="dm-msg err">❌ Error: ${e.message}</p>`;
    } finally {
        btn.disabled  = false;
        btn.innerHTML = '🔬 Analizar calidad';
    }
}

// ── 7. EXPORTAR ─────────────────────────────────────────────────────
function dmExport(fmt) {
    const catalog = dmCatalog();
    const info = document.getElementById('dmExportInfo');
    info.className = 'dm-msg ok';
    info.style.display = '';
    info.textContent = `⌛ Generando ${fmt.toUpperCase()}…`;
    const url = `/dataset/export?catalog=${encodeURIComponent(catalog)}&format=${fmt}`;
    // Trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `dataset_export.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => { info.textContent = `✅ Descarga iniciada (${fmt.toUpperCase()})`; }, 400);
}

// ── Utility ──────────────────────────────────────────────────────────
function escHtml(str) {
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showToast(msg) {
    // If there's already a toast function in the codebase reuse it, else simple alert
    if (typeof window._showToast === 'function') { window._showToast(msg); return; }
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;bottom:1.5rem;right:1.5rem;background:#2d3748;color:#e2e8f0;'
        + 'padding:0.7rem 1.2rem;border-radius:8px;font-size:0.85rem;z-index:9999;'
        + 'border-left:4px solid #48bb78;max-width:360px;line-height:1.5;';
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3500);
}
