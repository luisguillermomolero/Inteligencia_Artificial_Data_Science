# api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator.orchestrator import handle_generation

app = FastAPI(title="API de generación de texto", version="1.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir el esquema de entrada usando Pydantic
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    """
    Endpoint de prueba para verificar que el servidor funciona.
    """
    return {"message": "Servidor funcionando correctamente"}

@app.post("/generate")
async def generate(request: PromptRequest):
    """
    Endpoint para generar texto a partir de un prompt.
    """
    try:
        result = await handle_generation(request.prompt)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del servidor")
