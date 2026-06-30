FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python -m pip install --no-cache-dir pip==24.0 && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app/app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
