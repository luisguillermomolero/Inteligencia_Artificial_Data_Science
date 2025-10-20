# config.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Configuración de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuración de la aplicación
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "200"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))

# Verificar que la API key esté configurada
if not GEMINI_API_KEY:
    print("ADVERTENCIA: GEMINI_API_KEY no está configurada")
    print("Crea un archivo .env con tu API key de Gemini")
