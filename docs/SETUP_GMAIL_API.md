# Setup Gmail API cho Email Spam Classifier — Hướng dẫn

> **Mục đích**: setup Gmail API để poller chạy 24/7 trên EC2 đọc email mới
> trong inbox → POST sang FastAPI `/predict` → áp label `AI_SPAM` / `AI_HAM`
> trở lại Gmail.
>
> **Pricing**: Gmail API miễn phí với personal Gmail. Quota 1 tỷ units/ngày,
> use case này dùng khoảng 17k units/ngày → 0.0017% quota.
>
> Doc này chỉ cover phần setup OAuth (Phase 1). Phase 2-4 (install deps,
> deploy poller lên EC2, verify) mình sẽ làm sau khi bạn xong Phase 1 — xem
> [DEPLOY_BASELINE_EC2.md](DEPLOY_BASELINE_EC2.md) section 12 cho overview.
>
> **Liên quan tới Sprint 2 tickets**:
> - `ACW3-89` — [MODEL] Gmail OAuth bootstrap + `gmail_client.py` (Phase 1 + 2 trong doc này)
> - `ACW3-90` — [MODEL] Gmail poller `gmail_poller.py` (Phase 3 — deploy poller)
> - `ACW3-73` — [QA] Tests cho Gmail poller (Phase 4 — verify)
> - Cross-ref: [Sprint_2_Tickets.md §5.5, §5.2](Sprint_2_Tickets.md), [PROJECT_CONTEXT.md §9.4](PROJECT_CONTEXT.md).

---

## Phase 0 — (Khuyến nghị) Tạo Gmail account riêng để test

**Lý do**: poller áp label vào tất cả email mới đến inbox 24/7. Test trên
mail chính rủi ro:
- Bug logic → label sai hàng loạt, khó dọn (Gmail không có undo bulk)
- Token rò rỉ → kẻ khác đọc được toàn bộ inbox
- Quên revoke token sau POC → poller chạy ngầm sau khi đã quên
- Mail công ty (Workspace) → admin có thể chặn 3rd-party app

Account riêng: hỏng cũng không sao, xóa account vẫn ổn.

### Cách tạo (~2 phút)

1. Mở **incognito window** (để khỏi đăng xuất account chính)
2. → https://accounts.google.com/signup
3. Điền:
   - First name: `Spam` (hoặc gì cũng được)
   - Last name: `Classifier`
   - Username: tự chọn, vd `<tên bạn>.spam.poc`
4. Password + phone verify (Google bắt buộc số điện thoại)
5. **Skip** recovery email nếu hỏi
6. → https://mail.google.com để confirm inbox hoạt động

Ghi nhớ địa chỉ email mới — sẽ dùng ở step 1.3 (Test user) và để gửi email
test ở Phase 4.

---

## Phase 1 — Google Cloud Console (~5 phút)

> Login bằng **account Gmail mới** vừa tạo (hoặc dùng incognito + login).

### 1.1 Tạo Cloud Project

→ https://console.cloud.google.com/projectcreate

| Field | Value |
|---|---|
| Project name | `email-spam-classifier` |
| Organization | (để mặc định nếu có) |

Bấm **CREATE** → đợi vài giây → **Select project** ở thanh trên cùng để
chọn project vừa tạo.

### 1.2 Enable Gmail API

→ https://console.cloud.google.com/apis/library/gmail.googleapis.com

(Nếu URL prompt chọn project, chọn `email-spam-classifier`.)

Bấm **ENABLE**. Đợi ~10 giây.

### 1.3 Configure OAuth consent screen

→ https://console.cloud.google.com/apis/credentials/consent

#### Step 1 — User type

- Chọn **External** (không có Workspace org)
- Bấm **CREATE**

#### Step 2 — App information

| Field | Value |
|---|---|
| App name | `Email Spam Classifier POC` |
| User support email | email Gmail bạn vừa tạo |
| Developer contact information | email Gmail bạn vừa tạo |

Bấm **SAVE AND CONTINUE**.

#### Step 3 — Scopes

Bấm **ADD OR REMOVE SCOPES** → search box → tick **3 scope sau**:

```
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.modify
https://www.googleapis.com/auth/gmail.labels
```

(`gmail.readonly` để đọc email, `gmail.modify` để áp label, `gmail.labels`
để tạo label `AI_SPAM` / `AI_HAM` lần đầu.)

→ Bấm **UPDATE** → **SAVE AND CONTINUE**.

> Cảnh báo "sensitive scopes" có thể hiện — kệ, vì app đang ở Testing mode
> nên không cần Google review.

#### Step 4 — Test users

Bấm **ADD USERS** → nhập **chính email Gmail bạn vừa tạo** → **ADD** →
**SAVE AND CONTINUE**.

Chỉ user trong list này login được. Quên add → không login được.

#### Step 5 — Summary

Bấm **BACK TO DASHBOARD**. Xong consent screen.

> ⚠️ App đang ở **Testing mode**: token sẽ **expire mỗi 7 ngày**. Đủ cho
> POC. Sau 7 ngày phải re-run OAuth bootstrap (1 click). Refresh token vẫn
> valid trong 7 ngày → poller tự refresh OK.
>
> Nếu muốn token bền vĩnh viễn: phải **PUBLISH APP** + qua Google review
> (~2 tuần). KHÔNG cần cho POC.

### 1.4 Create OAuth 2.0 credentials (Desktop app)

→ https://console.cloud.google.com/apis/credentials

- Bấm **+ CREATE CREDENTIALS** ở thanh trên → chọn **OAuth client ID**
- **Application type**: **Desktop app** (rất quan trọng — không phải Web app)
- **Name**: `email-spam-classifier-desktop`
- Bấm **CREATE**

Modal popup hiện → bấm **DOWNLOAD JSON**. File tải về dạng:

```
client_secret_<long-id>.apps.googleusercontent.com.json
```

→ vào folder Downloads.

### 1.5 Đặt file credentials đúng chỗ trong project

Mở terminal trên Mac:

```bash
cd /Users/tienphan/VanTien/AIO/2026/Conquer/warmup3/email-spam-classification

mkdir -p app/secrets
mv ~/Downloads/client_secret_*.json app/secrets/gmail_credentials.json

ls -la app/secrets/gmail_credentials.json
```

Phải thấy file ~500-800 bytes ở đường dẫn `app/secrets/gmail_credentials.json`.

> File `app/secrets/` đã được [`.gitignore`](../.gitignore) cover (line:
> `app/secrets/`) → an toàn không leak lên Git. Vẫn đừng share công khai
> (chứa OAuth client_id + client_secret).

---

## Phase 1 — Checklist (đánh dấu khi xong)

- [ ] (optional) Đã tạo Gmail account riêng cho test
- [ ] 1.1 Tạo Cloud Project `email-spam-classifier`
- [ ] 1.2 Enable Gmail API
- [ ] 1.3 OAuth consent screen — Testing mode, 3 scopes, test user là chính bạn
- [ ] 1.4 Tạo OAuth client ID (Desktop app), download JSON
- [ ] 1.5 Move JSON về `app/secrets/gmail_credentials.json` trong project

→ Khi tick hết 5 mục → báo "done phase 1" + paste **email mới** (để mình
note + thêm vào docs/PROJECT_CONTEXT.md), mình bắt đầu Phase 2.

---

## Phase 2 — Local: chạy OAuth bootstrap (~2 phút, mình lo)

Sau khi bạn xong Phase 1, mình sẽ chạy:

```bash
cd email-spam-classification
.venv/bin/pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
.venv/bin/python scripts/gmail_oauth_bootstrap.py
```

→ Browser sẽ tự mở, hỏi bạn:
1. Chọn Google account → **chọn email mới**
2. Cảnh báo "Google hasn't verified this app" → bấm **Advanced** → **Go to
   Email Spam Classifier POC (unsafe)** (an toàn vì là app của chính bạn)
3. Allow 3 scopes (read / modify / labels)

→ Tạo file `app/secrets/gmail_token.json` (đã gitignored).

---

## Phase 3 — Deploy poller lên EC2 (~5 phút, mình lo)

Mình sẽ:
1. `scp` `app/secrets/gmail_token.json` lên EC2
2. `pip install` 4 gói google-* trong venv EC2
3. Tạo systemd unit `email-spam-gmail-poller.service` (đã có template trong
   [DEPLOY_BASELINE_EC2.md section 12](DEPLOY_BASELINE_EC2.md))
4. `systemctl enable --now`
5. `journalctl -u email-spam-gmail-poller -f` để theo dõi log lần đầu

---

## Phase 4 — Verify end-to-end (~3 phút, bạn + mình)

1. Bạn từ **mail chính** → gửi email vào **mail test** với subject + body
   spam (vd: `Subject: Free iPhone — Body: Click here to claim your prize now!`)
2. Đợi ~30 giây (`GMAIL_POLL_INTERVAL_SECONDS=30`)
3. Vào https://mail.google.com với mail test → mở email vừa gửi → thấy
   label `AI_SPAM` hoặc `AI_HAM` được gắn ở góc phải

Verify thêm:
- Trên Mac: `ssh ec2 'tail -f ~/email-spam-classification/logs/gmail_predictions.csv'`
  → thấy row mới mỗi lần email được score

---

## Reference — Files sẽ được tạo / dùng

| File | Đâu | Status |
|---|---|---|
| `app/secrets/gmail_credentials.json` | local Mac | Bạn tạo ở Phase 1.5 (Download → move) |
| `app/secrets/gmail_token.json` | local Mac → scp lên EC2 | Mình tạo ở Phase 2 (browser OAuth) |
| `logs/gmail_state.json` | EC2 | Tự tạo lần poller chạy đầu (lưu `last_history_id`) |
| `logs/gmail_predictions.csv` | EC2 | Tự tạo, append mỗi prediction |
| `/etc/systemd/system/email-spam-gmail-poller.service` | EC2 | Mình tạo ở Phase 3 |

---

## Troubleshooting (cho lần sau)

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| Bootstrap script báo `client secret not found` | File credentials chưa đúng đường dẫn | `ls app/secrets/gmail_credentials.json` — phải tồn tại |
| Browser báo "Access blocked: Authorization Error" | Email login khác với test user đã add ở 1.3 | Add đúng email vào Test users |
| Browser báo "This app is blocked" | App ở Production mode mà chưa qua review | Quay về Testing mode ở consent screen |
| Token expire sau 7 ngày | App ở Testing mode (expected) | Re-run bootstrap script |
| Poller không thấy email mới | `last_history_id` lưu ở `logs/gmail_state.json` quá cũ → Gmail trả `Sync required` | `rm logs/gmail_state.json` rồi restart service — sẽ pin lại từ historyId hiện tại (không backfill) |
| `403 insufficient_scope` khi áp label | Bootstrap chưa request đủ 3 scopes | Re-run bootstrap với credentials mới |

---

## Cleanup khi xong POC

Để revoke quyền truy cập Gmail của app này:

1. → https://myaccount.google.com/permissions (login bằng email test)
2. Tìm **Email Spam Classifier POC** → bấm **Remove access**

Hoặc xóa hẳn project Google Cloud:
1. → https://console.cloud.google.com/cloud-resource-manager
2. Select `email-spam-classifier` → **DELETE**

Trên EC2:

```bash
sudo systemctl disable --now email-spam-gmail-poller.service
sudo rm /etc/systemd/system/email-spam-gmail-poller.service
sudo systemctl daemon-reload
rm -f ~/email-spam-classification/app/secrets/gmail_token.json \
     ~/email-spam-classification/logs/gmail_{state,predictions}.{json,csv}
```

Trên Mac:

```bash
rm -rf app/secrets/
```
