#!/usr/bin/env bash
# ============================================================
#  INSTALACION - Sistema de Clasificacion Espectral v3.0
#  Compatible con macOS (Intel y Apple Silicon M1/M2/M3)
# ============================================================

set -e

# Colores para la terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # Sin color

echo ""
echo "================================================================"
echo "  INSTALACION - Sistema de Clasificacion Espectral v3.0"
echo "  macOS / Apple Silicon / Intel"
echo "================================================================"
echo ""

# ── [1/4] Verificar Python 3 ─────────────────────────────────────────────────
echo -e "${CYAN}[1/4] Verificando Python 3...${NC}"

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1)
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
        if [ "$MAJOR" = "3" ]; then
            PYTHON="$cmd"
            echo -e "${GREEN}  OK  $VER detectado${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}  ERROR: Python 3 no esta instalado.${NC}"
    echo ""
    echo "  Opciones de instalacion:"
    echo "  a) Homebrew (recomendado):  brew install python"
    echo "  b) Descarga oficial:        https://www.python.org/downloads/"
    echo ""
    exit 1
fi

echo ""

# ── [2/4] Actualizar pip ──────────────────────────────────────────────────────
echo -e "${CYAN}[2/4] Actualizando pip...${NC}"
"$PYTHON" -m pip install --upgrade pip --quiet
echo -e "${GREEN}  OK  pip actualizado${NC}"
echo ""

# ── [3/4] Instalar dependencias ───────────────────────────────────────────────
echo -e "${CYAN}[3/4] Instalando dependencias...${NC}"
echo ""

"$PYTHON" -m pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    pandas>=1.3.0 \
    astropy>=5.0.0 \
    scikit-learn>=1.0.0 \
    seaborn \
    flask>=2.0.0 \
    werkzeug>=2.0.0 \
    Pillow>=8.0.0 \
    joblib>=1.0.0

echo ""
echo -e "${GREEN}  OK  Dependencias instaladas${NC}"
echo ""

# ── [4/4] TensorFlow (opcional) ───────────────────────────────────────────────
echo "================================================================"
echo -e "${CYAN}[4/4] TensorFlow (opcional - para CNN)${NC}"
echo "================================================================"
echo ""
echo "  TensorFlow es necesario solo para clasificacion con CNN."
echo "  En Apple Silicon (M1/M2/M3) se usa 'tensorflow-macos' + 'tensorflow-metal'."
echo "  Es una descarga grande (~500 MB) y puede tardar varios minutos."
echo ""

read -rp "  Deseas instalar TensorFlow? (s/n): " INSTALAR_TF
echo ""

if [[ "$INSTALAR_TF" =~ ^[Ss]$ ]]; then
    # Detectar arquitectura
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        echo -e "${CYAN}  Apple Silicon detectado (arm64).${NC}"
        echo "  Instalando tensorflow-macos y tensorflow-metal..."
        "$PYTHON" -m pip install tensorflow-macos tensorflow-metal && \
            echo -e "${GREEN}  OK  TensorFlow para Apple Silicon instalado${NC}" || \
            echo -e "${YELLOW}  ADVERTENCIA: No se pudo instalar TensorFlow.${NC}"
    else
        echo -e "${CYAN}  Mac Intel detectado (x86_64).${NC}"
        echo "  Instalando tensorflow..."
        "$PYTHON" -m pip install tensorflow && \
            echo -e "${GREEN}  OK  TensorFlow instalado${NC}" || \
            echo -e "${YELLOW}  ADVERTENCIA: No se pudo instalar TensorFlow.${NC}"
    fi
else
    echo "  TensorFlow no instalado."
    echo "  Puedes instalarlo despues con:"
    if [ "$(uname -m)" = "arm64" ]; then
        echo "    pip install tensorflow-macos tensorflow-metal"
    else
        echo "    pip install tensorflow"
    fi
fi

echo ""
echo "================================================================"
echo -e "${GREEN}  INSTALACION COMPLETADA${NC}"
echo "================================================================"
echo ""
echo "  Paquetes instalados:"
echo "    - numpy, scipy, matplotlib  (procesamiento cientifico)"
echo "    - pandas                    (manejo de datos)"
echo "    - astropy                   (archivos FITS)"
echo "    - scikit-learn, joblib      (machine learning)"
echo "    - seaborn                   (visualizacion)"
echo "    - flask, werkzeug           (servidor web)"
echo ""
echo "  Siguiente paso: ejecuta  ./INICIAR.sh"
echo ""
