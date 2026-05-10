# Deploy Baseline Model lên EC2 — Hướng dẫn từng bước

> **Mục đích**: deploy nhanh baseline model train từ
> [notebooks/baseline/Email_Classification_V2.ipynb](../notebooks/baseline/Email_Classification_V2.ipynb)
> lên 1 EC2 instance để test khả thi (FastAPI + Streamlit). KHÔNG phải bản
> production full Medallion + Airflow + MLflow — đó là track riêng do team
> khác phụ trách.
>
> **Liên quan tới Sprint 2 tickets**:
> - `ACW3-65` — [TECH LEAD] Provision EC2 + systemd services (toàn bộ doc này)
> - `ACW3-91` — [MODEL] Docker compose `app/` (alternative cho dev local)
> - Cross-ref: [Sprint_2_Tickets.md §5.1, §5.5](Sprint_2_Tickets.md), [PROJECT_CONTEXT.md §11](PROJECT_CONTEXT.md), [SETUP_CLOUDFRONT.md](SETUP_CLOUDFRONT.md) (HTTPS frontend), [SETUP_GMAIL_API.md](SETUP_GMAIL_API.md) (Gmail integration).

---

## 1. Context

| | |
|---|---|
| **Model** | `Pipeline(TfidfVectorizer + LogisticRegression)`, F1 ≈ 0.958, Precision ≥ 0.99, threshold 0.770 |
| **Artifact** | [models/best_spam_classifier.pkl](../models/best_spam_classifier.pkl) (~2.2 MB) + [models/model_metadata.json](../models/model_metadata.json) |
| **Preprocessing pipeline** | Notebook 2-stage (`TextCleaner.aggressive_clean` → `preprocess_str`). Khác với canonical [src/text_preprocessing.py](../src/text_preprocessing.py) → cần dùng [src/notebook_preprocessing.py](../src/notebook_preprocessing.py) ở serving để tránh train/serve skew. |
| **Serving** | FastAPI (port 8000) + Streamlit UI (port 8501), cả 2 chạy bằng systemd |
| **EC2** | Amazon Linux 2023, t3.small, 20GB gp3, region us-east-1 |

---

## 2. Prerequisites (local máy dev)

- macOS / Linux có `ssh`, `rsync`, `curl`
- Python 3.11 venv trong project (`python3.11 -m venv .venv`) đã cài đủ deps để chạy notebook
- Đã chạy notebook → có sẵn `models/best_spam_classifier.pkl` + `models/model_metadata.json`
- AWS account + AWS Console access

---

## 3. Tạo EC2 instance

### 3.1 Launch instance (AWS Console → EC2 → Launch instance)

| Field | Value |
|---|---|
| Name | `email-spam-classifier` |
| AMI | **Amazon Linux 2023** (free tier eligible) — KHÔNG dùng AL2 (cũ, Python 3.7) |
| Instance type | **t3.small** (2 vCPU, 2 GB RAM) — t3.micro 1GB không đủ chạy đồng thời FastAPI + Streamlit |
| Key pair | Create new → tên `email-spam-key` → format `.pem` → **download ngay** (chỉ tải được 1 lần). Lưu vào `~/Desktop/email-spam-key.pem` |
| Network → SSH from | `My IP` (an toàn). Nếu IP hay đổi: `Anywhere` cho POC. |
| Storage | **20 GiB**, **gp3** |

### 3.2 Mở port trong Security Group

EC2 console → Instances → click instance → tab **Security** → click vào SG → **Edit inbound rules**:

| Type | Port | Source | Description |
|---|---|---|---|
| Custom TCP | `8000` | `0.0.0.0/0` | FastAPI |
| Custom TCP | `8501` | `0.0.0.0/0` | Streamlit |

> ⚠️ `0.0.0.0/0` = public Internet. OK cho POC. Production: dùng nginx + auth + HTTPS.

### 3.3 chmod + smoke test SSH

```bash
chmod 400 ~/Desktop/email-spam-key.pem
ssh -i ~/Desktop/email-spam-key.pem ec2-user@<public-dns> 'echo OK; uname -a'
```

> SSH user là `ec2-user` (Amazon Linux), KHÔNG phải `ubuntu`.

---

## 4. Provision EC2 (Python 3.11 + venv)

```bash
EC2=ec2-user@<public-dns>
KEY=~/Desktop/email-spam-key.pem

ssh -i $KEY $EC2 'set -e
sudo dnf install -y -q python3.11 python3.11-pip gcc rsync tar gzip
mkdir -p ~/email-spam-classification
cd ~/email-spam-classification
[ ! -d .venv ] && python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade -q pip wheel setuptools
'
```

---

## 5. Sync code + model lên EC2

Từ project root local:

```bash
EC2=ec2-user@<public-dns>
KEY=~/Desktop/email-spam-key.pem
DEST=/home/ec2-user/email-spam-classification

# src/ (gồm cả src/notebook_preprocessing.py — bắt buộc cho baseline serving)
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' --exclude='*.pyc' \
  src/ $EC2:$DEST/src/

# app/ (api.py + streamlit_app.py)
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' --exclude='*.pyc' \
  app/ $EC2:$DEST/app/

# models/ (pkl + metadata)
rsync -avz -e "ssh -i $KEY" \
  models/ $EC2:$DEST/models/

# requirements
rsync -avz -e "ssh -i $KEY" \
  requirements-serving.txt $EC2:$DEST/
```

> Dùng `requirements-serving.txt` (lean: chỉ sklearn / fastapi / streamlit / nltk) thay
> vì `requirements.txt` (có thêm mlflow / dev tooling không cần ở serving).

---

## 6. Install Python deps + NLTK data

```bash
ssh -i $KEY $EC2 'set -e
cd ~/email-spam-classification
source .venv/bin/activate
pip install -q -r requirements-serving.txt
python -m nltk.downloader -q stopwords wordnet punkt_tab averaged_perceptron_tagger_eng omw-1.4
'
```

Smoke test predict trên EC2:

```bash
ssh -i $KEY $EC2 '
cd ~/email-spam-classification
PREPROCESS_PIPELINE=notebook PYTHONPATH=src .venv/bin/python -c "
from predict import load_model, predict_spam
load_model()
print(predict_spam(\"Free iPhone\", \"Click here now to claim\"))
"
'
```

Phải in ra `{'label': 'spam', 'spam_probability': ~0.93, 'threshold': 0.7702...}`.

---

## 7. Cài systemd services (FastAPI + Streamlit)

```bash
ssh -i $KEY $EC2 'set -e
PROJECT=/home/ec2-user/email-spam-classification
mkdir -p $PROJECT/logs

# --- FastAPI ---
sudo tee /etc/systemd/system/email-spam-api.service > /dev/null <<EOF
[Unit]
Description=Email Spam FastAPI service
After=network.target

[Service]
User=ec2-user
WorkingDirectory=$PROJECT
Environment="PYTHONPATH=$PROJECT/src:$PROJECT"
Environment="PREPROCESS_PIPELINE=notebook"
Environment="MODEL_PATH=$PROJECT/models/best_spam_classifier.pkl"
Environment="MODEL_METADATA_PATH=$PROJECT/models/model_metadata.json"
Environment="PREDICTION_LOG_PATH=$PROJECT/logs/predictions.csv"
ExecStart=$PROJECT/.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# --- Streamlit ---
sudo tee /etc/systemd/system/email-spam-streamlit.service > /dev/null <<EOF
[Unit]
Description=Email Spam Streamlit UI
After=network.target email-spam-api.service
Wants=email-spam-api.service

[Service]
User=ec2-user
WorkingDirectory=$PROJECT
Environment="API_URL=http://127.0.0.1:8000/predict"
ExecStart=$PROJECT/.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now email-spam-api.service email-spam-streamlit.service
'
```

Quan trọng:
- `PREPROCESS_PIPELINE=notebook` → `src/predict.py` import `notebook_preprocessing` thay vì `text_preprocessing`. Bắt buộc cho baseline model. KHI deploy bản full pipeline (Airflow training), bỏ env var này (default = `canonical`).
- `--server.headless true` để Streamlit không tự mở browser, không hỏi email config.

---

## 8. Smoke test từ máy local

```bash
EC2_DNS=<public-dns>

# Health
curl -fsS http://$EC2_DNS:8000/health

# Predict
curl -fsS -X POST http://$EC2_DNS:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject":"Free iPhone","body":"Click here now to claim"}'

# Streamlit (chỉ check 200)
curl -o /dev/null -w "HTTP %{http_code}\n" http://$EC2_DNS:8501/
```

Mở browser: **http://\<public-dns\>:8501** → thấy UI Email Spam Classifier → nhập subject + body → Predict.

---

## 9. Operate / Debug

### Check service

```bash
sudo systemctl status email-spam-api email-spam-streamlit
sudo systemctl is-active email-spam-api
```

### Tail logs

```bash
sudo journalctl -u email-spam-api -f
sudo journalctl -u email-spam-streamlit -f
tail -f ~/email-spam-classification/logs/predictions.csv
```

### Update code rồi restart

```bash
# Local: rsync code mới lên EC2
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' \
  src/ $EC2:$DEST/src/
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' \
  app/ $EC2:$DEST/app/

# EC2: restart service
ssh -i $KEY $EC2 'sudo systemctl restart email-spam-api email-spam-streamlit'
```

### Update model

```bash
# Sau khi có pkl mới ở local
rsync -avz -e "ssh -i $KEY" models/ $EC2:$DEST/models/
ssh -i $KEY $EC2 'sudo systemctl restart email-spam-api'
```

### Stop / Start instance (tiết kiệm cost)

- AWS Console → EC2 → Instances → select → **Stop instance** (vẫn tốn EBS ~ $1.6/tháng/20GB, không tốn compute)
- **Start instance** lại sẽ có public DNS mới — phải update lại tất cả URL test.
- Để giữ public IP cố định: **allocate Elastic IP** + associate (Elastic IP idle có phí).

---

## 10. Cost estimate

| Resource | Giá | Tháng (24/7) |
|---|---|---|
| t3.small running | $0.0208/h | ~$15 |
| EBS gp3 20GB | $0.08/GB-month | ~$1.6 |
| Data transfer out (light demo) | $0.09/GB sau 100GB free | gần như $0 |
| **Total** | | **~$16-17/tháng** |

> Stop instance khi không dùng: chỉ trả EBS ($1.6/tháng).
> Terminate instance: trả $0 (mất hết data).

---

## 11. Known limitations / TODO

1. **HTTP, không HTTPS** — token/payload đi plaintext. Cần nginx + Let's Encrypt cho production.
2. **Không có auth** — `/predict` mở cho public Internet. Thêm API key header hoặc đặt sau Cloudflare Access.
3. **Single instance, không HA** — instance crash → service down. Cần ASG + ALB cho production.
4. **Không có monitoring** — chỉ có `journalctl`. Cần CloudWatch / Prometheus.
5. **Train/serve skew với canonical pipeline** — model trained bằng notebook preprocessing. Khi team chuyển sang Airflow training (canonical pipeline ở [src/text_preprocessing.py](../src/text_preprocessing.py)) phải retrain hoặc bỏ `PREPROCESS_PIPELINE=notebook` env var.
6. **Gmail integration chưa setup** — xem section 12.

---

## 12. (Optional) Nối Gmail near-real-time poller

> **Hướng dẫn đầy đủ tách riêng**: [SETUP_GMAIL_API.md](SETUP_GMAIL_API.md)
> (cover Google Cloud Console setup, OAuth, verify end-to-end). Section
> dưới đây chỉ là quick-reference command.

```bash
# Local: chạy OAuth bootstrap 1 lần
python scripts/gmail_oauth_bootstrap.py
# → tạo app/secrets/gmail_token.json

# Local: copy token + cài thêm Gmail deps trên EC2
scp -i $KEY app/secrets/gmail_token.json $EC2:$DEST/app/secrets/

ssh -i $KEY $EC2 '
cd ~/email-spam-classification
source .venv/bin/activate
pip install -q google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
'

# EC2: thêm systemd unit cho gmail_poller (chưa setup ở deploy đầu này)
# Reference: app/gmail_poller.py
```

Service unit gợi ý:

```ini
[Unit]
Description=Gmail polling worker
After=network.target email-spam-api.service
Wants=email-spam-api.service

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/email-spam-classification
Environment="PYTHONPATH=/home/ec2-user/email-spam-classification"
Environment="API_URL=http://127.0.0.1:8000/predict"
Environment="GMAIL_TOKEN_JSON=/home/ec2-user/email-spam-classification/app/secrets/gmail_token.json"
Environment="GMAIL_STATE_PATH=/home/ec2-user/email-spam-classification/logs/gmail_state.json"
Environment="GMAIL_PREDICTION_LOG=/home/ec2-user/email-spam-classification/logs/gmail_predictions.csv"
Environment="GMAIL_POLL_INTERVAL_SECONDS=30"
ExecStart=/home/ec2-user/email-spam-classification/.venv/bin/python -m app.gmail_poller
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 13. Quick reference — toàn bộ deploy trong 1 block

```bash
# Biến
EC2=ec2-user@ec2-54-164-98-169.compute-1.amazonaws.com
KEY=~/Desktop/email-spam-key.pem
DEST=/home/ec2-user/email-spam-classification

# 1. Provision
ssh -i $KEY $EC2 'sudo dnf install -y -q python3.11 python3.11-pip gcc rsync &&
  mkdir -p ~/email-spam-classification && cd ~/email-spam-classification &&
  [ ! -d .venv ] && python3.11 -m venv .venv;
  .venv/bin/pip install -q --upgrade pip wheel'

# 2. Sync
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' src/ $EC2:$DEST/src/
rsync -avz -e "ssh -i $KEY" --exclude='__pycache__/' app/ $EC2:$DEST/app/
rsync -avz -e "ssh -i $KEY" models/ $EC2:$DEST/models/
rsync -avz -e "ssh -i $KEY" requirements-serving.txt $EC2:$DEST/

# 3. Install deps + NLTK
ssh -i $KEY $EC2 'cd ~/email-spam-classification && source .venv/bin/activate &&
  pip install -q -r requirements-serving.txt &&
  python -m nltk.downloader -q stopwords wordnet punkt_tab averaged_perceptron_tagger_eng omw-1.4'

# 4. systemd (xem section 7 cho file đầy đủ)

# 5. Test từ Mac
curl http://${EC2#*@}:8000/health
open http://${EC2#*@}:8501
```
