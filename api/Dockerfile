# Build from the repo root: docker build -f api/Dockerfile -t semantic-search-api .
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY semantic_search ./semantic_search
COPY api ./api
COPY data ./data
COPY mynn_model_weights_final.pkl ./mynn_model_weights_final.pkl

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
