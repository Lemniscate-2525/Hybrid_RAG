# Production Deployment Guide — Hybrid RAG System

This guide outlines deployment procedures for both containerized environments (Docker Compose / Kubernetes) and virtual machine bare-metal hosts.

---

## 1. Quickstart: Local Production Stack with Docker Compose

The simplest way to run the complete production stack (Qdrant + Redis + FastAPI Backend + Streamlit UI) is using Docker Compose:

```bash
# 1. Clone repository and navigate to root
cd Hybrid_RAG

# 2. Create environment file from template
cp .env.example .env
# Edit .env and supply your MISTRAL_API_KEY and SECRET_KEY

# 3. Build and launch all services
docker compose up -d --build

# 4. Verify service health
docker compose ps
curl http://localhost:8000/api/v1/health/ready
```

### Access Points
- **FastAPI API & OpenAPI Docs:** `http://localhost:8000/docs`
- **Streamlit Web Application:** `http://localhost:8501`
- **Qdrant Vector DB Console:** `http://localhost:6333/dashboard`

---

## 2. Bare-Metal / Virtual Machine Deployment

### Step 1: Install System Dependencies
Ensure Python 3.12, Redis, and build tools are installed:
```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv redis-server
```

### Step 2: Set Up Virtual Environment & Dependencies
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Production Environment
```bash
cp .env.example .env
# Set ENVIRONMENT=production
# Generate a secure 64-character SECRET_KEY:
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 4: Run Database Initialization & Health Probes
```bash
python -c "from src.db.database import init_db; init_db()"
pytest tests/ -v
```

### Step 5: Configure Systemd Services

Create `/etc/systemd/system/rag-backend.service`:
```ini
[Unit]
Description=Hybrid RAG FastAPI Backend
After=network.target redis-server.service

[Service]
User=www-data
WorkingDirectory=/var/www/Hybrid_RAG
ExecStart=/var/www/Hybrid_RAG/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
EnvironmentFile=/var/www/Hybrid_RAG/.env

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/rag-frontend.service`:
```ini
[Unit]
Description=Hybrid RAG Streamlit Frontend
After=rag-backend.service

[Service]
User=www-data
WorkingDirectory=/var/www/Hybrid_RAG
ExecStart=/var/www/Hybrid_RAG/venv/bin/streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-backend rag-frontend
```

---

## 3. Reverse Proxy Configuration (Nginx + SSL)

Terminate SSL and route traffic securely to FastAPI and Streamlit:

```nginx
server {
    listen 443 ssl http2;
    server_name rag.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/rag.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.yourdomain.com/privkey.pem;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Backend API routing
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Streamlit UI routing
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 4. Rollback Strategy
If an update causes regressions:
1. Revert code via Git: `git checkout <previous-tag-or-commit>`.
2. Restart systemd services: `sudo systemctl restart rag-backend rag-frontend`.
3. Vector data in Qdrant and relational tables in SQLite/PostgreSQL remain completely intact since point IDs are deterministic and partitioned by `user_id`.

