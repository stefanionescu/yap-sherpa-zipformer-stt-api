FROM alphacep/kaldi-vosk-server-gpu:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ARG MODEL_URL=https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
ARG MODEL_NAME=vosk-model-en-us-0.22

# Base already contains CUDA-enabled libvosk; install runtime tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1 git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Important: do NOT install PyPI 'vosk'; keep the GPU lib from the base image
RUN sed -i '/^vosk[=<>]/d' /app/requirements.txt && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /models/en /tmp/model && \
    curl -L -o /tmp/model.zip "$MODEL_URL" && \
    unzip -q /tmp/model.zip -d /tmp/model && \
    mv "/tmp/model/$MODEL_NAME"/* /models/en/ && \
    rm -rf /tmp/model /tmp/model.zip

COPY server /app/server
COPY tests /app/tests
COPY samples /app/samples

ENV MODEL_DIR=/models/en \
    HOST=0.0.0.0 \
    PORT=8000 \
    CONCURRENCY=64 \
    LOG_LEVEL=INFO

EXPOSE 8000
ENTRYPOINT []
CMD ["python3", "-m", "server.ws_server"]
