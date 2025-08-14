#!/usr/bin/env bash
set -euo pipefail

# --- settings you can tweak ---
MYSQL_DB=mlflowdb
MYSQL_USER=mlflow
MYSQL_PASS=mlflowpw
MYSQL_ROOT_PASS=rootpw
MYSQL_PORT=3306
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/mlflow/mysql_data"
ART_DIR="${PROJECT_DIR}/mlflow/artifacts"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"

msg() { echo -e "\033[1;32m[+] $*\033[0m"; }
warn() { echo -e "\033[1;33m[!] $*\033[0m"; }
err() { echo -e "\033[1;31m[x] $*\033[0m"; }

# --- 0) sanity: need curl, gpg, lsb-release on minimal images ---
msg "Installing basic tools..."
apt-get update -y
apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# --- 1) Install Docker Engine + Compose plugin (official repo) ---
if ! command -v docker >/dev/null 2>&1; then
  msg "Installing Docker Engine..."
  . /etc/os-release
  DIST_ID="${ID:-debian}"
  ARCH="$(dpkg --print-architecture)"

  curl -fsSL "https://download.docker.com/linux/${DIST_ID}/gpg" | gpg --dearmor -o /usr/share/keyrings/docker.gpg
  echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/${DIST_ID} $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
else
  msg "Docker CLI already present."
fi

# --- 2) Start Docker daemon (systemd or manual) ---
start_docker_daemon() {
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^docker\.service'; then
    msg "Starting docker with systemd..."
    systemctl enable --now docker
  else
    warn "systemd docker service not available. Starting dockerd in background..."
    # create log dir if needed
    mkdir -p /var/log
    # start only if not already running
    if ! pgrep -x dockerd >/dev/null 2>&1; then
      nohup dockerd >/var/log/dockerd.log 2>&1 &
      disown || true
      sleep 2
    fi
  fi

  # wait for docker to be responsive
  msg "Waiting for Docker daemon..."
  for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
      msg "Docker is up."
      return
    fi
    sleep 1
  done
  err "Docker daemon did not become ready. Check /var/log/dockerd.log"
  exit 1
}

start_docker_daemon

# --- 3) Prepare local folders ---
msg "Preparing local data folders..."
mkdir -p "${DATA_DIR}" "${ART_DIR}"

# --- 4) Write docker-compose.yml (MySQL only) ---
if [ ! -f "${COMPOSE_FILE}" ]; then
  msg "Creating ${COMPOSE_FILE}"
  cat > "${COMPOSE_FILE}" <<YAML
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: ${MYSQL_DB}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASS}
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASS}
    command: >
      --default-authentication-plugin=mysql_native_password
      --character-set-server=utf8mb4 --collation-server=utf8mb4_0900_ai_ci
      --bind-address=0.0.0.0
    volumes:
      - ./mlflow/mysql_data:/var/lib/mysql
    ports:
      - "127.0.0.1:${MYSQL_PORT}:3306"
    restart: unless-stopped
YAML
else
  warn "${COMPOSE_FILE} already exists; leaving it untouched."
fi

# --- 5) Start MySQL container ---
msg "Starting MySQL container (data stored in ./mlflow/mysql_data)..."
docker compose -f "${COMPOSE_FILE}" up -d

# --- 6) Wait for MySQL TCP port to accept connections ---
msg "Waiting for MySQL to accept connections on 127.0.0.1:${MYSQL_PORT}..."
for i in {1..60}; do
  (exec 3<>/dev/tcp/127.0.0.1/${MYSQL_PORT}) >/dev/null 2>&1 && { msg "MySQL is reachable."; break; }
  sleep 1
  if [ "$i" -eq 60 ]; then
    warn "Could not confirm MySQL TCP readiness; proceeding anyway."
  fi
done

# --- 7) Print MLflow command & quick test info ---
cat <<EOF

============================================================
MySQL is running locally and persists data in:
  ${DATA_DIR}

Artifacts directory for MLflow (create it if you haven't):
  ${ART_DIR}

Use this MLflow Tracking Server command (run in your Python env):
------------------------------------------------------------
pip install "mlflow>=2.10" pymysql

ART_ROOT="${ART_DIR}"
mlflow server \\
  --backend-store-uri "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASS}@127.0.0.1:${MYSQL_PORT}/${MYSQL_DB}" \\
  --default-artifact-root "file://\${ART_ROOT}" \\
  --host 0.0.0.0 --port 5000
------------------------------------------------------------

Then, in each training process:
------------------------------------------------------------
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my_local_exp")
with mlflow.start_run():
    mlflow.log_param("lr", 1e-3)
    mlflow.log_metric("val_loss", 0.123, step=1)
------------------------------------------------------------

Quick Docker checks:
  docker ps                      # see the mysql container
  docker logs \$(docker ps -q -f name=_mysql_) --tail 50
  docker compose down            # stop MySQL
  docker compose up -d          # start again

Security note:
  These defaults are for local development only.
  Change MYSQL_* passwords for anything beyond localhost.
============================================================

EOF