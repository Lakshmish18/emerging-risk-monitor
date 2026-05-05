# AI service — Emerging Risk Monitor
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Chroma persistence and optional Redis: set env and/or mount volumes at run time
CMD sh -c 'gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 120 --access-logfile - --error-logfile - "app:app"'
