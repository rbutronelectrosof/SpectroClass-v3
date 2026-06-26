#!/usr/bin/env bash
# ============================================================
#  INICIO - Sistema de Clasificacion Espectral v3.0
#  Compatible con macOS (Intel y Apple Silicon M1/M2/M3)
# ============================================================

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "================================================================"
echo "  WebApp - Sistema de Clasificacion Espectral v3.0"
echo "  macOS / Apple Silicon / Intel"
echo "================================================================"
echo ""

# ── Directorio del proyecto ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detectar Python 3 ────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        if [ "$MAJOR" = "3" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}  ERROR: Python 3 no esta instalado.${NC}"
    echo "  Ejecuta primero: ./INSTALAR.sh"
    exit 1
fi

# ── Verificar Flask ───────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import flask" &>/dev/null; then
    echo -e "${RED}  ERROR: Las dependencias no estan instaladas.${NC}"
    echo "  Ejecuta primero: ./INSTALAR.sh"
    exit 1
fi

# ── Verificar app.py ─────────────────────────────────────────────────────────
if [ ! -f "webapp/app.py" ]; then
    echo -e "${RED}  ERROR: No se encuentra webapp/app.py${NC}"
    echo "  Asegurate de ejecutar este script desde el directorio del proyecto."
    exit 1
fi

# ── Advertencia si no hay modelos ML entrenados ──────────────────────────────
if [ ! -f "models/decision_tree.pkl" ]; then
    echo "================================================================"
    echo -e "${YELLOW}  ADVERTENCIA: Modelo ML no encontrado${NC}"
    echo "================================================================"
    echo ""
    echo "  El sistema funcionara con clasificador fisico + template matching."
    echo "  Para mayor precision, entrena los modelos desde la interfaz web."
    echo ""
    sleep 4
fi

# ── Abrir navegador automaticamente ──────────────────────────────────────────
(sleep 5 && open "http://localhost:5000") &

echo "================================================================"
echo ""
echo -e "  Servidor iniciado en: ${GREEN}http://localhost:5000${NC}"
echo ""
echo "  Tu navegador se abrira automaticamente en 5 segundos..."
echo ""
echo "  Para DETENER el servidor: presiona Ctrl+C"
echo ""
echo "================================================================"
echo ""

"$PYTHON" webapp/app.py

echo ""
echo "  Servidor detenido."
