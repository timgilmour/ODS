FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/ ./
RUN if [ -f package.json ]; then npm ci && npm run build; else mkdir -p dist; fi
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY --from=ui /ui/dist ui/dist
EXPOSE 3015
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3015"]
