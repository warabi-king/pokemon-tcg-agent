FROM python:3.11-slim AS agent-assets
COPY agents /assets/agents
COPY tools/prepare_cloud_assets.py /prepare_cloud_assets.py
RUN python /prepare_cloud_assets.py /assets \
    && find /assets/agents -type d -empty -delete

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv/poketcg

WORKDIR /srv/poketcg

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt
RUN python -c "import torch; print('Cloud PyTorch:', torch.__version__)"

COPY app app
COPY data data
COPY --from=agent-assets /assets/agents agents

EXPOSE 8080
CMD exec gunicorn --bind "0.0.0.0:${PORT:-8080}" --workers 1 --threads 20 --timeout 60 app.cloud_server:app
