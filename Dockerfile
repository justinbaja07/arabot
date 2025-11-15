# Use a smaller base image
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only minimal deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install lightweight CPU-only torch
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "arabic_bot_full.py"]
