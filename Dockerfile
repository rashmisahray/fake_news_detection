# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for memory & CPU optimization
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive
ENV MALLOC_ARENA_MAX 2
ENV OMP_NUM_THREADS 1
ENV MKL_NUM_THREADS 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Use CPU PyTorch to save memory and size)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt averaged_perceptron_tagger vader_lexicon

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Create log file
RUN touch app.log && chmod 666 app.log

# Run single-worker Uvicorn for production on low-memory environments (prevents OOM)
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "180"]
