#!/usr/bin/env bash
#
# Deploy the Sprint 2 serving stack (FastAPI + Streamlit + Gmail poller) to EC2.
#
# Provisioning is idempotent: re-running just re-syncs code + artifacts and
# bounces the systemd services. Safe to call after every `git pull`.
#
# Requirements on the dev machine:
#   - ssh + rsync
#   - SSH key with access to EC2 (default: ~/Desktop/email-spam-key.pem)
#   - You have run `python -m src.pipelines.train --snapshot <YYYY-MM>` locally
#     so models/best_spam_classifier.pkl + models/train.json exist
#   - data/gold/snapshot=<S>/artifacts/{tfidf_vectorizer,numeric_scaler}.pkl
#     exist (built by `src.etl.gold_build`)
#
# Usage:
#   EC2_HOST=ec2-XX-XX-XX-XX.compute-1.amazonaws.com \
#   KEY=~/Desktop/email-spam-key.pem \
#       scripts/deploy_to_ec2.sh
#
#   # or one-shot:
#   EC2_HOST=ec2-... KEY=~/Desktop/key.pem bash scripts/deploy_to_ec2.sh
#
#   # to install the systemd units for the first time:
#   INSTALL_SYSTEMD=1 EC2_HOST=... bash scripts/deploy_to_ec2.sh

set -euo pipefail

EC2_HOST="${EC2_HOST:?Set EC2_HOST=ec2-XXX.compute-1.amazonaws.com}"
KEY="${KEY:-$HOME/Desktop/email-spam-key.pem}"
EC2_USER="${EC2_USER:-ec2-user}"
REMOTE_DIR="${REMOTE_DIR:-/home/${EC2_USER}/email-spam-classification}"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-0}"

EC2="${EC2_USER}@${EC2_HOST}"
SSH=(ssh -i "${KEY}" "${EC2}")
RSYNC_BASE=(rsync -avz -e "ssh -i ${KEY}" --exclude='__pycache__/' --exclude='*.pyc')

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "── Target ────────────────────────────────────────"
echo "  host : ${EC2}"
echo "  key  : ${KEY}"
echo "  dest : ${REMOTE_DIR}"
echo "──────────────────────────────────────────────────"

# ── Pre-flight: artifacts must exist locally ─────────────────────────────────
[[ -f models/best_spam_classifier.pkl ]] \
    || { echo "ERR: models/best_spam_classifier.pkl missing — run src.pipelines.train first"; exit 1; }
[[ -f models/train.json ]] \
    || { echo "ERR: models/train.json missing"; exit 1; }

SNAPSHOT="$(python3 -c 'import json; print(json.load(open("models/train.json"))["snapshot"])')"
ARTIFACT_DIR="data/gold/snapshot=${SNAPSHOT}/artifacts"
[[ -d "${ARTIFACT_DIR}" ]] \
    || { echo "ERR: ${ARTIFACT_DIR} missing — run src.etl.gold_build --month ${SNAPSHOT}"; exit 1; }

echo "  champion snapshot = ${SNAPSHOT}"
echo

# ── 1. Provision Python + venv on EC2 (idempotent) ───────────────────────────
echo "[1/5] Provisioning Python 3.11 + venv (idempotent)…"
"${SSH[@]}" bash <<EOF
set -e
sudo dnf install -y -q python3.11 python3.11-pip gcc rsync tar gzip 2>&1 | tail -3
mkdir -p ${REMOTE_DIR}/{logs,app/secrets}
cd ${REMOTE_DIR}
[ ! -d .venv ] && python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade -q pip wheel setuptools
EOF
echo

# ── 2. Sync code + artifacts ─────────────────────────────────────────────────
echo "[2/5] Rsync code + artifacts…"

# Source code (minimal subset needed by serving)
"${RSYNC_BASE[@]}" --include='__init__.py' --include='utils/' --include='utils/**' \
    --exclude='*' src/ "${EC2}:${REMOTE_DIR}/src/"

# Serving app
"${RSYNC_BASE[@]}" app/ "${EC2}:${REMOTE_DIR}/app/"

# Model + metadata
"${RSYNC_BASE[@]}" models/ "${EC2}:${REMOTE_DIR}/models/"

# Gold snapshot artifacts ONLY (not the parquet/npz training data — too big and unused at serving)
"${RSYNC_BASE[@]}" --include='snapshot=*/' \
    --include='snapshot=*/artifacts/' --include='snapshot=*/artifacts/*' \
    --exclude='*' data/gold/ "${EC2}:${REMOTE_DIR}/data/gold/"

# Requirements
"${RSYNC_BASE[@]}" requirements-serving.txt "${EC2}:${REMOTE_DIR}/"

# Gmail token (only if exists; needed for the poller)
if [[ -f app/secrets/gmail_token.json ]]; then
    rsync -avz -e "ssh -i ${KEY}" \
        app/secrets/gmail_token.json "${EC2}:${REMOTE_DIR}/app/secrets/"
    echo "  ✓ gmail_token.json synced (poller can authenticate)"
else
    echo "  ⚠ app/secrets/gmail_token.json not found — skipping (poller will fail until present)"
fi
echo

# ── 3. Install Python deps on EC2 ────────────────────────────────────────────
echo "[3/5] Installing Python deps on EC2…"
"${SSH[@]}" bash <<EOF
set -e
cd ${REMOTE_DIR}
.venv/bin/pip install -q -r requirements-serving.txt
EOF
echo

# ── 4. (Optional) Install systemd units ──────────────────────────────────────
if [[ "${INSTALL_SYSTEMD}" == "1" ]]; then
    echo "[4/5] Installing systemd units (INSTALL_SYSTEMD=1)…"
    "${RSYNC_BASE[@]}" scripts/systemd/ "${EC2}:/tmp/email-spam-systemd/"
    "${SSH[@]}" bash <<EOF
set -e
sudo cp /tmp/email-spam-systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable email-spam-api.service email-spam-streamlit.service email-spam-gmail-poller.service
rm -rf /tmp/email-spam-systemd
EOF
    echo "  ✓ units installed + enabled"
else
    echo "[4/5] Skipping systemd install (set INSTALL_SYSTEMD=1 to install on first deploy)"
fi
echo

# ── 5. Restart services + smoke ──────────────────────────────────────────────
echo "[5/5] Restarting services + smoke test…"
"${SSH[@]}" bash <<'EOF'
set -e
for u in email-spam-api email-spam-streamlit email-spam-gmail-poller; do
    if systemctl list-unit-files | grep -q "^${u}.service"; then
        sudo systemctl restart "${u}"
        echo "  restarted ${u}"
    else
        echo "  (skip ${u} — unit not installed yet; run with INSTALL_SYSTEMD=1)"
    fi
done

# Give uvicorn ~3s to come up before hitting /health
sleep 3
echo
echo "── /health ──"
curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool || echo "API not responding"
EOF

echo
echo "── Done ──────────────────────────────────────────"
echo "  curl -fsS http://${EC2_HOST}:8000/health"
echo "  open http://${EC2_HOST}:8501"
echo
echo "  Tail logs:"
echo "    ssh -i ${KEY} ${EC2} 'sudo journalctl -u email-spam-api -f'"
echo "    ssh -i ${KEY} ${EC2} 'sudo journalctl -u email-spam-gmail-poller -f'"
