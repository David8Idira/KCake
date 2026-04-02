# KCake Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install KCake
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy KCake source
COPY . .

# Install KCake
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 kcake && \
    mkdir -p /home/kcake/.cache/huggingface && \
    chown -R kcake:kcake /app

USER kcake

# Default command
CMD ["python", "-m", "kcake", "serve", "--model", "meta-llama/Llama-3.1-8B", "--cluster-key", "secret123"]

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
