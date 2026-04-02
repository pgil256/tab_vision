FROM python:3.11-slim

# Install ffmpeg (needed for audio extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port (Railway sets PORT env var)
EXPOSE ${PORT:-5000}

# Start with gunicorn
CMD gunicorn run:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 1 --preload
