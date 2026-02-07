FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Create data directory for RAG
RUN mkdir -p data/chromadb logs

# Expose ports
EXPOSE 8000 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
