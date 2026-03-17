FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to avoid cold-start timeout
COPY main.py .
RUN python -c "from main import download_model; download_model()"

# Expose port
EXPOSE 8000

# Run with uvicorn (shell form is required to evaluate $PORT)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT