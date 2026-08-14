# 3.12 to match requires-python in pyproject.toml; the image previously pinned
# 3.11, so the container ran on a version the project declares it does not support.
FROM python:3.12-slim

WORKDIR /app

# Install deps first so application edits do not invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop root. The default image runs as uid 0, which is unnecessary for serving HTTP.
RUN useradd --create-home --uid 10001 aura && chown -R aura:aura /app
USER aura

# Render (and most PaaS) inject the port to bind; hardcoding 8000 meant the
# container listened somewhere the platform was not routing to.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
