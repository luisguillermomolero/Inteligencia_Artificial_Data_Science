from provider.provider_local import generate_text_local

MAX_PROMPT_LENGTH = 200

async def handler_generator(prompt: str) -> str:
    if len(prompt) == 0:
        raise ValueError("El prompt no puede estar vacio")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("El prompt excede el tamaño máximo permitido")
    
    return await generate_text_local(prompt)