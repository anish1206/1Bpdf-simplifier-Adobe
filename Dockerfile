# Use Python 3.9 slim image for better performance
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script and run it
COPY download_model.py .
RUN python download_model.py

# Copy application code
COPY main.py .
COPY utils.py .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Default command
CMD ["python", "main.py"]
