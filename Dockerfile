# CogniFlow API (and optional Streamlit) — multi-stage not required for prototype
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default: API. Override in docker-compose for Streamlit.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
