FROM python:3.11-slim

# Install system GIS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose backend (8000) and Streamlit frontend (8501)
EXPOSE 8000 8501

# Start both FastAPI backend and Streamlit frontend
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
