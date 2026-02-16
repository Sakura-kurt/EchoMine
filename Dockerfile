FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    webrtcvad \
    faster-whisper \
    aio-pika

COPY stt_server.py .
COPY rabbitmq_config.py .

EXPOSE 8001

CMD ["uvicorn", "stt_server:app", "--host", "0.0.0.0", "--port", "8001"]
