from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator.orchestrator import handler_generator

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"]
                   )

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"mensaje": "Servicio funcionando correctamente"}

@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        result = await handler_generator(request.prompt)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except:
        raise HTTPException(status_code=500, detail="Error interno del servidor")