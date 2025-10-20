import os
import streamlit as st
from datetime import datetime
import json

from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="App con Gemini API", layout="centered")
st.title("Aplicación generadora de texto con Gemini API")
st.markdown(
    "Esta app utiliza la API de Gemini para crear contenido a partir de un PROMPT."
    "  Si no hay API disponible, no generará texto real."
)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Necesitas configurar el Token API de Gemini (GEMINI_API_KEY).")
    st.stop()

if not GEMINI_AVAILABLE:
    st.error("No se encontró la libreria `google.genai` o similar. Debe instalarse primero")
    st.stop()

client = genai.Client(api_key=API_KEY)

prompt = st.text_area(
    "Ingrese un prompt o instrucción",
    height=100,
    placeholder="Hay algo en lo que pueda ayudarte?..."
)

max_token = st.sidebar.slider(
    "Longitud máxima (token)",
    50,
    100,
    150
)

temperature = st.sidebar.slider(
    "Temperatura",
    0.0,
    2.0,
    1.0,
    step=0.1
)

if st.button("Generar contenido"):
    if not prompt.strip():
        st.warning("Escribe algo antes de generar")
    else:
        with st.spinner("Generando..."):
            try:
               response = client.models.generate_content(
                   model=MODEL_NAME,
                   contents=prompt,
                   config=genai.types.GenerateContentConfig(
                       temperature=temperature,
                       max_output_tokens=max_token
                   )
               )
               text = response.text
            except Exception as e:
                st.error(f"Error al generar contenido (Texto): {e}")
                text =""
        
        if text:
            st.subheader("Texto generado:")
            st.write(text)
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "output": text,
                "model": MODEL_NAME,
                "temperature": temperature,
                "max_token": max_token
            })

if st.session_state.get("history"):
    st.markdown("---")
    st.subheader("Historial de generación:")
    for entry in st.session_state.history:
        st.write(F"{entry['timestamp']} - Modelo: {entry['model']}")
        st.write("**Prompt:**", entry['prompt'])
        st.write("**Output:**", entry['output'])
        st.write("---")
                