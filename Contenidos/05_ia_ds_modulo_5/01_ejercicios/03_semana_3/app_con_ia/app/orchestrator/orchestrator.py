# orchestrator/orchestrator.py
from provider.provider_gemini import generate_text_gemini
from config import MAX_PROMPT_LENGTH

async def handle_generation(prompt: str) -> str:
    if len(prompt) == 0:
        raise ValueError("El prompt no puede estar vacío.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("El prompt excede el tamaño máximo permitido.")
    
    return await generate_text_gemini(prompt)
