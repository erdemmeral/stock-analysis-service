FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    netcat \
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

# Add health check script
COPY healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Add startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /healthcheck.sh

# Run the service
CMD ["/start.sh"] 