# Use slim Python image
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/

# Configure environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MALLOC_ARENA_MAX=1
ENV PYTHONMALLOC=malloc

# Run the service
CMD ["python", "-m", "src.main"] 