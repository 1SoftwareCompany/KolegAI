FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Minimal Python + tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 and then the rest
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        "torch==2.5.1+cu124" \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# App code
COPY embed.py .
COPY security.py .

# Default envs – override at runtime if needed
ENV QDRANT_URL=http://localhost:6333 \
    LLM_API=http://localhost:5000/v1/chat/completions \
    COLLECTION=unicom \
    NOTION_EXPORT_DIR=Notion-Export

EXPOSE 8000

CMD ["python3", "embed.py", "--mode", "serve", "--port", "8000"]
