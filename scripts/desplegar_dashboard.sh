#!/bin/bash
# Despliegue del dashboard con un túnel de Cloudflare (entregable: accesible desde un computador y un celular).
#
#   bash scripts/desplegar_dashboard.sh
#
# Requiere `cloudflared` (macOS: `brew install cloudflared`; Windows: `winget install Cloudflare.cloudflared`).
# Usa un túnel rápido (sin cuenta): Cloudflare imprime una dirección https://<algo>.trycloudflare.com que sirve
# mientras este script esté corriendo. Ctrl+C cierra el túnel y el dashboard.
# Para subir un CT por el túnel: el cargador normal sirve hasta ~150 MB (Cloudflare corta las peticiones de más de
# unos 2 minutos, HTTP 524); para archivos más pesados, la «subida por partes» del dashboard (probada con 434 MB).
set -e
cd "$(dirname "$0")/.."
PUERTO="${PUERTO:-8501}"
mkdir -p results

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "Falta cloudflared. macOS: brew install cloudflared | Windows: winget install Cloudflare.cloudflared"
    exit 1
fi

# Python del entorno del proyecto (`uv sync`): .venv/bin en macOS y Linux, .venv/Scripts en Windows (Git Bash)
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
if [ ! -x "$PY" ]; then
    echo "No encuentro el entorno .venv. Créalo primero con: uv sync"
    exit 1
fi

# Dashboard en segundo plano, solo en esta máquina (el túnel es la única puerta de entrada)
"$PY" -m streamlit run dashboard/app.py --server.port "$PUERTO" --server.address 127.0.0.1 \
    --server.headless true --browser.gatherUsageStats false > results/dashboard_streamlit.log 2>&1 &
DASH=$!
trap 'kill $DASH 2>/dev/null' EXIT

echo "Esperando al dashboard en el puerto $PUERTO..."
for _ in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:$PUERTO" && break
    sleep 1
done

echo "Abriendo el túnel. La dirección pública aparece abajo (https://....trycloudflare.com):"
cloudflared tunnel --url "http://127.0.0.1:$PUERTO"
