# YOLOv8 Object Detection - Docker Image
# Supports CPU and GPU (NVIDIA CUDA)

# CPU-only base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY README.md ./

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for web interface (if added later)
EXPOSE 8080

# Default command - run demo
CMD ["python", "src/demo.py"]
