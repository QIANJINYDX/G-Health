#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${LOG_DIR}"

# ---- helpers ----
log() { echo "[$(date '+%F %T')] $*"; }

start_bg() {
  # start_bg <name> <cmd...>
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"
  log "Starting ${name} ..."
  nohup "$@" >"${logfile}" 2>&1 &
  echo $! > "${LOG_DIR}/${name}.pid"
  log "  PID=$(cat "${LOG_DIR}/${name}.pid")  LOG=${logfile}"
}

wait_http() {
  # wait_http <url> <timeout_sec>
  local url="$1"
  local timeout="${2:-60}"
  local t=0
  until curl -fsS "${url}" >/dev/null 2>&1; do
    sleep 1
    t=$((t+1))
    if [[ "${t}" -ge "${timeout}" ]]; then
      log "ERROR: Timeout waiting for ${url}"
      return 1
    fi
  done
}

ollama_keepalive() {
  # 通过 API 预热并设置 keep_alive（不会进入交互）
  # ollama_keepalive <model> <keep_alive>
  local model="$1"
  local keep="$2"
  log "Preloading model ${model} (keep_alive=${keep}) ..."
  curl -fsS "http://127.0.0.1:11434/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${model}\",\"prompt\":\" \",\"stream\":false,\"keep_alive\":\"${keep}\"}" \
    >/dev/null
}

cleanup() {
  log "Stopping all started processes..."
  for pidfile in "${LOG_DIR}"/*.pid; do
    [[ -e "${pidfile}" ]] || continue
    pid="$(cat "${pidfile}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "  kill ${pid} ($(basename "${pidfile}" .pid))"
      kill "${pid}" 2>/dev/null || true
    fi
  done
  log "Done."
}
trap cleanup INT TERM

cd "${BASE_DIR}"

# ---- 1) start ollama serve in background ----
start_bg "ollama_serve" ollama serve

# ---- 2) wait ollama ready ----
log "Waiting for Ollama API..."
wait_http "http://127.0.0.1:11434/api/tags" 120
log "Ollama is ready."

# ---- 3) preload models and keep them resident ----
# 注意：这里用 -1s 表示一直不卸载（负 duration 且带单位）
ollama_keepalive "qwen3:14b" "-1s"
ollama_keepalive "qwen3:0.6b" "-1s"

# ---- 4) activate conda env (non-interactive safe) ----
# 任选其一：如果你知道 conda 安装路径，请改成对应的
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Miniconda
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # Miniforge
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  # fallback: try conda hook
  eval "$(conda shell.bash hook)" || true
fi

conda activate jianxiaozhi_web

# ---- 5) start your python services ----
start_bg "run_py" python run.py
start_bg "rag_service" python app/util/rag_service.py --mode serve
start_bg "risk_assessment" python risk_assessment/app.py

# MCP
start_bg "mcp_all" bash -lc "cd '${BASE_DIR}/MCP' && python start_all_mcp.py"

log "All services started."
log "Logs are in: ${LOG_DIR}"
log "Press Ctrl+C to stop everything."

# keep script running
wait
