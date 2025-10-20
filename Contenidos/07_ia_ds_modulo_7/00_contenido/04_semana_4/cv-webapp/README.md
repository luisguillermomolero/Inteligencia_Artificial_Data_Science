CV WebApp - Demo

Estructura:

backend/: FastAPI app
frontend/: archivos estáticos (index.html, app.js)
docker-compose.yml: orquesta backend + frontend (nginx)

Instrucciones rápidas:

1. Crear entorno virtual e instalar dependencias para backend:

    cd backend
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
    pip install -r requirements.txt

2. Ejecutar backend localmente:

    uvicorn backend.app.main:app --reload --port 8000

3. Abrir frontend (index.html) directamente en el navegador o usar docker-compose:

    docker compose up --build

4. Exportar modelo TorchScript (opcional):

    python scripts/export_torchscript.py

5. Ejecutar tests:

    pytest tests/

Notas:
- El script export_torchscript.py genera models/resnet18_traced.pt si lo ejecutas.
- Para producción use certificados HTTPS, autenticación y límites de subida apropiados.
