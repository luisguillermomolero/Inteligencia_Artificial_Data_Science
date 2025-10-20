# orchestrator/orchestrator.py
from provider.provider_local import generate_text_local

MAX_PROMPT_LENGTH = 200  # Seguridad: limitar entrada

async def handle_generation(prompt: str) -> str:
    """
    Lógica de negocio para generar texto usando el generador local.
    """
    if len(prompt) == 0:
        raise ValueError("El prompt no puede estar vacío.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("El prompt excede el tamaño máximo permitido.")
    
    # Usar el generador local para texto creativo
    return await generate_text_local(prompt)
