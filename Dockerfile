# Dockerfile cho REST API cua PVehicle-AI.
#
# Dung build nhieu giai doan: giai doan builder cai thu vien, giai doan
# runtime chi chep ket qua sang — anh cuoi cung khong chua trinh bien dich
# nen nho hon va it be mat tan cong hon.

# =====================================================================
# Giai doan 1: cai thu vien
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Cac goi can de bien dich mot so thu vien Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-api.txt ./

# Cai vao thu muc rieng de de chep sang giai doan sau.
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# =====================================================================
# Giai doan 2: anh chay that
# =====================================================================
FROM python:3.11-slim

# libgl1 va libglib2.0-0: OpenCV can, ngay ca ban headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Chay bang tai khoan thuong, khong dung root.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Chi chep nhung gi can de chay — khong chep notebook, tests, docs.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Kiem tra suc khoe: dung /health (tien trinh con song) chu khong dung
# /ready — /ready tra 503 khi thieu mo hinh, se lam container bi khoi
# dong lai lien tuc du van phuc vu duoc cac endpoint tu van.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
