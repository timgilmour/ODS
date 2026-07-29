# Build from repo root:  docker build -f ods/extensions/services/node-agent/Dockerfile .
FROM python:3.12-slim
WORKDIR /app
COPY ods/extensions/services/node-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Vendor the single-source GPU collectors from dashboard-api
COPY ods/extensions/services/dashboard-api/gpu.py ./gpu.py
COPY ods/extensions/services/node-agent/*.py ./
EXPOSE 7720
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${NODE_AGENT_PORT:-7720}"]
