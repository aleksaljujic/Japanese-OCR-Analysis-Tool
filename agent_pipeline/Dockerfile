FROM python:3.10-slim

WORKDIR /app

# Kopiramo requirements iz agent foldera
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiramo kod agenta
COPY . .

EXPOSE 80

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "80"]