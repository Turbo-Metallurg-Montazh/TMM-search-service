FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/biencoder_model
ENV INDEX_DIR=/data/index_data
ENV PRICE_LIST_DIR=/data/price_lists

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .
COPY biencoder_model ./biencoder_model
COPY index_data ./index_data
COPY price_lists ./price_lists

RUN mkdir -p /data/index_data /data/price_lists \
    && useradd --system --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
