# Fase 7: Dockerización del servicio de predicción

FROM python:3.11-slim

# Creamos directorio de trabajo
WORKDIR /app

# Copiamos dependencias e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el proyecto dentro de la imagen
COPY . .

# Exponemos el puerto donde correrá la API
EXPOSE 8000

# Comando por defecto: lanzar la API con uvicorn
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
