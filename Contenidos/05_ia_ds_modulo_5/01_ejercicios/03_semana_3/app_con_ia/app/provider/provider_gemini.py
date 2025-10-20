# provider/provider_gemini.py
import google.generativeai as genai
from config import GEMINI_API_KEY, MAX_TOKENS

genai.configure(api_key=GEMINI_API_KEY)

async def generate_text_gemini(prompt: str) -> str:
    try:
        if not GEMINI_API_KEY:
            return "Error: API key de Gemini no configurada. Revisa tu archivo .env"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        full_prompt = f"""
        Eres un asistente creativo que genera texto en español. 
        Responde de manera creativa, amigable y útil.
        
        Prompt del usuario: {prompt}
        
        Genera una respuesta creativa y útil:
        """
        
        response = model.generate_content(full_prompt)
        
        if response.text:
            return response.text.strip()
        else:
            return f"Respuesta generada para: {prompt}"
        
    except Exception as e:
        return f"Error al generar texto: {str(e)}"
