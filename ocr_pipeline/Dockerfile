FROM python:3.10-slim

# Instalacija sistemskih biblioteka potrebnih za Paddle i OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "ocr_pipeline:app", "--host", "0.0.0.0", "--port", "80"]