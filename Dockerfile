FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Configure environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set resource limits
ENV MALLOC_ARENA_MAX=2
ENV PYTHONMALLOC=malloc

# Run the service
CMD ["python", "-m", "src.main"] 