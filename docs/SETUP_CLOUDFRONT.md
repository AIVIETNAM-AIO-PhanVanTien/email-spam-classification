# Setup CloudFront cho Streamlit (HTTPS + custom domain + lockdown EC2)

> **Mục đích**: đặt CloudFront distribution trước Streamlit (port 8501) để có:
> 1. HTTPS miễn phí (cert `*.cloudfront.net` mặc định)
> 2. Custom domain (vd `spam.tien.com` → CloudFront → EC2)
> 3. Giấu public IP EC2: Security Group chỉ accept request từ CloudFront IP range
> 4. CDN cache (lợi ích thấp với Streamlit dynamic, nhưng có sẵn)
>
> FastAPI (port 8000) **không** đi qua CloudFront — Streamlit sẽ gọi
> `http://127.0.0.1:8000/predict` qua loopback nội bộ EC2. Port 8000 sẽ
> được đóng khỏi Internet (chỉ giữ access từ My IP để debug).
>
> **Liên quan tới Sprint 2 tickets**:
> - `ACW3-65` — [TECH LEAD] Provision EC2 + systemd (prerequisite cho doc này)
> - `ACW3-67` — [TECH LEAD] Final demo prep (HTTPS URL từ CloudFront sẽ dùng cho buổi demo)
> - Cross-ref: [DEPLOY_BASELINE_EC2.md](DEPLOY_BASELINE_EC2.md) (deploy EC2 + Streamlit gốc), [Sprint_2_Tickets.md §5.1](Sprint_2_Tickets.md).
>
> **Lưu ý phạm vi**: doc này là *optional* cho Sprint 2. Nếu thời gian gấp, demo có thể chạy thẳng `http://<ec2-dns>:8501`. Nên làm khi có domain custom + buổi demo có audience ngoài team.

---

## Architecture sau khi setup xong

```
Browser (HTTPS)
    │
    ▼
*.cloudfront.net hoặc spam.tien.com (HTTPS, cert miễn phí)
    │  (port 443)
    ▼
┌─────────── CloudFront edge ───────────┐
│  Cache policy: CachingDisabled         │
│  Origin protocol: HTTP                 │
│  Origin: ec2-...:8501                  │
└──────────────┬─────────────────────────┘
               │  (port 8501, HTTP)
               ▼
        EC2 Security Group
        ┌──────────────────────────────┐
        │ 8501  ← only CloudFront IPs  │
        │ 8000  ← only My IP           │
        │ 22    ← My IP                │
        └──────────────┬───────────────┘
                       ▼
        Streamlit :8501
                │  (loopback)
                ▼
        FastAPI :8000  → /predict
```

---

## Phase 0 — Quyết định trước

### 0.1 Custom domain hay không?

Có 2 cách dùng CloudFront:

| Mode | URL | Cost | Setup |
|---|---|---|---|
| **Default cloudfront.net** | `https://d3xxxxxxxx.cloudfront.net` | $0 | Phase 2 thẳng |
| **Custom domain** | `https://spam.tien.com` (vd) | Domain ~$10/năm | Phase 1 + 2 + 4 |

**Yêu cầu cho custom domain:**
- Domain bạn đã sở hữu (Route 53, Namecheap, GoDaddy, vv)
- Quyền add CNAME / Alias record vào DNS

Nếu chưa có domain → bắt đầu với default `cloudfront.net` trước, lúc nào có domain thì add sau.

### 0.2 Region của ACM cert

**Quan trọng**: ACM cert dùng cho CloudFront **bắt buộc tạo ở region `us-east-1`**, bất kể EC2 ở đâu. EC2 đang ở us-east-1 nên trùng region — tiện. Nếu EC2 ở region khác cũng vẫn phải tạo cert ở us-east-1.

---

## Phase 1 — (Chỉ khi dùng custom domain) Tạo ACM cert

→ https://us-east-1.console.aws.amazon.com/acm/home?region=us-east-1#/certificates/request

> Đảm bảo region top-right hiển thị **N. Virginia (us-east-1)**.

1. **Request a public certificate** → Next
2. **Fully qualified domain name**: nhập domain bạn muốn (vd `spam.tien.com`)
   - Có thể add wildcard: `*.tien.com` để cover nhiều subdomain
3. **Validation method**: chọn **DNS validation (recommended)**
4. **Key algorithm**: RSA 2048
5. **Tags**: skip
6. **Request**

→ Certificate hiển thị trạng thái **Pending validation**.

7. Click vào cert → tab **Domains** → thấy 1 row với `CNAME name` + `CNAME value`
8. Bấm **Create records in Route 53** (nếu domain ở Route 53) — auto add record
   - Nếu domain không ở Route 53: copy CNAME name + value, add manually vào DNS provider của bạn (Namecheap/GoDaddy/Cloudflare DNS, etc.)
9. Đợi 5-30 phút → status đổi thành **Issued** ✓

**Ghi nhớ ARN của cert** — sẽ dùng ở Phase 2:
```
arn:aws:acm:us-east-1:<account-id>:certificate/<cert-id>
```

---

## Phase 2 — Tạo CloudFront distribution

→ https://us-east-1.console.aws.amazon.com/cloudfront/v4/home

Bấm **Create distribution**.

### 2.1 Origin

| Field | Value |
|---|---|
| **Origin domain** | Paste public DNS của EC2: `ec2-54-164-98-169.compute-1.amazonaws.com` |
| Name | (auto-fill — kệ) |
| **Protocol** | **HTTP only** |
| **HTTP port** | **8501** |
| HTTPS port | (kệ, không dùng) |
| Minimum origin SSL protocol | (kệ, không dùng) |
| **Origin path** | (để trống) |
| Add custom header | skip |
| **Enable Origin Shield** | No |

### 2.2 Default cache behavior

| Field | Value |
|---|---|
| **Path pattern** | Default (`*`) |
| **Compress objects automatically** | Yes |
| **Viewer protocol policy** | **Redirect HTTP to HTTPS** |
| **Allowed HTTP methods** | **GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE** (Streamlit cần POST cho form submit) |
| **Restrict viewer access** | No |
| **Cache key and origin requests** | **Use legacy cache settings** (đơn giản hơn cho POC) |

Khi chọn legacy cache settings:

| Field | Value |
|---|---|
| **Headers** | **Include the following headers** → tick: `Host`, `Origin`, `Referer`, `User-Agent`, `Sec-WebSocket-Key`, `Sec-WebSocket-Version`, `Sec-WebSocket-Extensions`, `Sec-WebSocket-Protocol`, `Sec-WebSocket-Accept` |
| **Query strings** | All |
| **Cookies** | All |
| **Object caching** | **Customize** — Min TTL `0`, Max TTL `0`, Default TTL `0` (disable cache vì Streamlit dynamic) |

> Hoặc dùng **Cache policy and origin request policy (recommended)** với:
> - Cache policy: **CachingDisabled** (managed)
> - Origin request policy: **AllViewer** (managed) — forward tất cả headers/cookies/query
>
> Cách này clean hơn nhưng UI bước này hơi khác — cứ chọn 1 trong 2 cách trên.

### 2.3 Settings (Distribution-level)

| Field | Value |
|---|---|
| **Price class** | **Use only North America and Europe** (rẻ nhất, đủ cho POC). Nếu user ở VN thì pick "All edge locations" cho latency thấp. |
| **Alternate domain names (CNAMEs)** | **Để trống** nếu dùng default `cloudfront.net`. **Nhập** `spam.tien.com` (domain của bạn) nếu dùng custom domain. |
| **Custom SSL certificate** | (Để trống) cho default. **Chọn ACM cert** ở Phase 1 cho custom domain. |
| **Supported HTTP versions** | HTTP/2 + HTTP/1.1 |
| **Default root object** | (để trống) |
| **Standard logging** | Off (POC) |
| **IPv6** | On |
| **Description** | `Streamlit Email Spam UI` |

Bấm **Create distribution**.

### 2.4 Đợi distribution deploy

CloudFront sẽ deploy ra ~200 edge worldwide → mất **5-15 phút**.
Status: **Deploying** → **Enabled** ✓

Trong khi đợi, **copy distribution domain name**, vd:

```
d3abc12def34g5.cloudfront.net
```

---

## Phase 3 — Update Streamlit để chạy ngon sau proxy CloudFront

Streamlit có XSRF protection check origin header — sẽ fail khi browser
gửi request từ CloudFront (`https://xxx.cloudfront.net`) đến Streamlit
(nghĩ rằng nó host ở `http://ec2-...:8501`). Disable check để cho POC
chạy được.

```bash
EC2=ec2-user@ec2-54-164-98-169.compute-1.amazonaws.com
KEY=~/Desktop/email-spam-key.pem

ssh -i $KEY $EC2 'sudo tee /etc/systemd/system/email-spam-streamlit.service > /dev/null <<EOF
[Unit]
Description=Email Spam Streamlit UI
After=network.target email-spam-api.service
Wants=email-spam-api.service

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/email-spam-classification
Environment="API_URL=http://127.0.0.1:8000/predict"
ExecStart=/home/ec2-user/email-spam-classification/.venv/bin/streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl restart email-spam-streamlit.service
sudo systemctl status email-spam-streamlit.service --no-pager | head -10
'
```

---

## Phase 4 — (Chỉ khi dùng custom domain) DNS

### Nếu domain ở Route 53

→ https://us-east-1.console.aws.amazon.com/route53/v2/hostedzones

1. Chọn hosted zone của domain
2. **Create record**
3. Record name: `spam` (sub) hoặc để trống cho apex
4. Record type: **A**
5. **Alias**: bật ON
6. Route traffic to: **Alias to CloudFront distribution** → chọn distribution vừa tạo
7. **Create records**

### Nếu domain ở Namecheap / GoDaddy / Cloudflare

DNS provider không hỗ trợ ALIAS / ANAME → dùng CNAME:

| Type | Host | Value | TTL |
|---|---|---|---|
| CNAME | `spam` (vd) | `d3abc12def34g5.cloudfront.net` | 300 |

Lưu ý: CNAME không dùng được cho apex domain (`tien.com`). Nếu muốn apex
trỏ vào CloudFront → phải chuyển DNS sang Route 53 (free) hoặc dùng
Cloudflare CNAME flattening.

Đợi 5-15 phút DNS propagate → test:

```bash
dig +short spam.tien.com
# → trả ra IP của CloudFront edge (vd 18.x, 52.x)
```

---

## Phase 5 — Lockdown EC2 Security Group

Mục tiêu: chỉ CloudFront mới gọi được port 8501; port 8000 chỉ My IP truy cập (để debug).

### 5.1 Lấy CloudFront managed prefix list ID

CloudFront publish 1 AWS-managed prefix list chứa tất cả IP edge của họ.
Reference qua prefix list ID thay vì hardcode IP.

→ https://us-east-1.console.aws.amazon.com/vpcconsole/home?region=us-east-1#ManagedPrefixLists

Tìm row có Name = **`com.amazonaws.global.cloudfront.origin-facing`** → copy **Prefix list ID**, dạng:

```
pl-3b927c52   (us-east-1)
```

### 5.2 Update Security Group inbound

→ EC2 Console → **Instances** → click instance → tab **Security** → click vào Security Group → **Edit inbound rules**

**Xóa các rule cũ:**
- Custom TCP `8501` from `0.0.0.0/0` ← xóa
- Custom TCP `8000` from `0.0.0.0/0` ← xóa

**Add các rule mới:**

| Type | Port | Source | Description |
|---|---|---|---|
| SSH | 22 | My IP (đã có) | SSH |
| Custom TCP | **8501** | **Prefix list `pl-3b927c52`** | Streamlit (CloudFront only) |
| Custom TCP | **8000** | **My IP** | FastAPI (debug only — Streamlit gọi qua loopback) |

→ **Save rules**.

### 5.3 Verify SG đã chặn được public

Từ máy ngoài (Mac của bạn):

```bash
# Trước: HTTP trực tiếp lên 8501 phải timeout / refuse
curl --max-time 5 http://ec2-54-164-98-169.compute-1.amazonaws.com:8501
# Expected: timeout (curl: (28) Connection timed out)

# Trước: HTTP trực tiếp lên 8000 phải vẫn OK (My IP)
curl --max-time 5 http://ec2-54-164-98-169.compute-1.amazonaws.com:8000/health
# Expected: {"status":"ok",...}

# Trước: HTTPS qua CloudFront phải OK
curl --max-time 10 https://d3abc12def34g5.cloudfront.net/_stcore/health
# Expected: ok
```

Nếu CloudFront vẫn không reach origin (502 Bad Gateway), kiểm tra:
- Prefix list ID đúng region (us-east-1 = pl-3b927c52)
- Distribution status = Enabled (không Deploying)
- Origin protocol = HTTP, port = 8501

---

## Phase 6 — Verify end-to-end

1. Mở browser: `https://d3abc12def34g5.cloudfront.net` (hoặc `https://spam.tien.com`)
2. Phải thấy padlock 🔒 + Streamlit UI Email Spam Classifier
3. Nhập subject + body → bấm Predict → kết quả hiện ra

Test thêm:
- Trên EC2: `tail -f ~/email-spam-classification/logs/predictions.csv` → thấy row mới mỗi lần predict qua UI

---

## Cost estimate

| Resource | Free tier (12 tháng đầu) | Sau free tier |
|---|---|---|
| CloudFront data out | 1 TB / tháng | $0.085 / GB (NA-EU) |
| CloudFront requests | 10M HTTP/HTTPS / tháng | $0.0075 / 10k req |
| ACM cert (CloudFront) | Free luôn | Free luôn |
| Route 53 hosted zone | $0.50 / zone / tháng | giống |
| Domain registrar | $10-15 / năm | giống |
| **Total cho POC (1-2 user/ngày)** | **$0** | gần như $0 |

CloudFront có free tier vĩnh viễn cho 1TB+10M req nếu bạn dùng AWS account
12 tháng đầu — POC này dùng <0.1% nên không tốn gì.

---

## Phase 7 — Cleanup khi xong POC

1. **CloudFront**: Console → Distribution → Disable → đợi Deploying xong → Delete
   - Lưu ý: Distribution phải Disable trước, đợi 5-10 phút, mới Delete được.
2. **ACM cert** (custom domain): Console → ACM → cert → Delete (chỉ delete được khi không có CloudFront/ALB nào dùng)
3. **DNS record** (custom domain): xóa CNAME / A record trong DNS provider
4. **Restore SG**: nếu tắt CloudFront mà vẫn muốn truy cập Streamlit/FastAPI direct → mở lại 0.0.0.0/0 cho 8000/8501

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| **502 Bad Gateway** từ CloudFront | Origin EC2 SG không cho CloudFront vào 8501 | Verify prefix list `pl-3b927c52` đúng region; verify distribution Enabled |
| **403 Forbidden Streamlit XSRF token expired** | Chưa disable XSRF check | Phase 3 — thêm `--server.enableXsrfProtection false` |
| **WebSocket disconnect liên tục** (UI không respond) | Headers WebSocket không forward | Cache behavior — verify `Sec-WebSocket-*` headers in whitelist; hoặc dùng managed AllViewer policy |
| **`https://` không load nhưng `http://` ok** | Distribution chưa deploy xong | Đợi status Enabled; có thể mất 15 phút |
| **Custom domain trả ERR_TOO_MANY_REDIRECTS** | DNS trỏ sai (vd CNAME → EC2 thay vì CloudFront) | Verify `dig spam.tien.com` ra IP CloudFront, không phải EC2 |
| **ACM cert ở wrong region** | CloudFront chỉ dùng cert us-east-1 | Phase 1 — đảm bảo region top-right = N. Virginia |
| **Streamlit báo "Please wait"** mãi | App chưa kết nối WebSocket | Check browser DevTools → Network → WS tab xem có frame nào không. Phải có "Sec-WebSocket-*" headers forward. |
| **Browser cache HTTPS sang IP cũ** | Old DNS cached | `Cmd+Shift+R` hard refresh; hoặc thử incognito |

---

## Checklist tổng

- [ ] (Optional) Phase 1: ACM cert ở **us-east-1** Issued
- [ ] Phase 2: CloudFront distribution Enabled, copy `*.cloudfront.net` URL
- [ ] Phase 3: Streamlit systemd unit có `--server.enableXsrfProtection false` + `--server.enableCORS false`
- [ ] (Optional) Phase 4: DNS record CNAME / Alias trỏ vào CloudFront, `dig` xác nhận
- [ ] Phase 5: SG inbound: 8501 ← prefix list `pl-3b927c52`, 8000 ← My IP, xóa các rule 0.0.0.0/0 cũ
- [ ] Phase 6: Browser mở `https://...` thấy padlock + UI

---

## Câu hỏi mở khi user đến đây

Trước khi bắt đầu thực thi, mình cần biết:

1. **Có domain custom không?** → quyết định có làm Phase 1 + 4 hay không.
   - Có: cho mình biết domain (vd `spam.tien.com`) + DNS provider (Route 53? Namecheap?)
   - Không: skip, dùng `*.cloudfront.net` luôn.

2. **AWS account có ở free tier 12 tháng đầu không?** → cost estimate.
