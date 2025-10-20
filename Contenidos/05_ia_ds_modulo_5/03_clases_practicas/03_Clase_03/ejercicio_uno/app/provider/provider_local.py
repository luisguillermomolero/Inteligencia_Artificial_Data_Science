import random
import re

async def generate_text_local(prompt: str) -> str:
    try:
       prompt_lower = prompt.lower()
       if any(word in prompt_lower for word in ['historia', 'cuento', 'story', 'narrar']):
           return generate_story(prompt)
       elif any(word in prompt_lower for word in ['poema', 'poem', 'verso', 'rima']):
           return generate_poem(prompt)
       elif any(word in prompt_lower for word in ['explica', 'explain', 'qué es', 'what is']):
           return generate_explanation(prompt)
       elif any(word in prompt_lower for word in ['describe', 'describe', 'cómo es', 'how is']):
           return generate_description(prompt)
       else:
           return generate_general_response(prompt)
    except Exception as e:
        return f"Error al generar texto {str(e)}"

def generate_story(prompt: str) -> str:
    """Genera una historia corta basada en el prompt."""
    stories = [
        f"Érase una vez, en un lugar muy especial, donde {prompt.lower()} se convirtió en el protagonista de una aventura increíble. La historia comenzó cuando...",
        f"En un mundo lleno de posibilidades, {prompt.lower()} marcó el inicio de algo extraordinario. Los personajes se encontraron y...",
        f"La magia de {prompt.lower()} se desplegó ante los ojos de todos. Era como si el universo entero conspirara para crear algo hermoso..."
    ]
    return random.choice(stories)

def generate_poem(prompt: str) -> str:
    """Genera un poema corto basado en el prompt."""
    poems = [
        f"En el silencio de la noche,\n{prompt.lower()} brilla con luz propia,\ncomo una estrella fugaz\nque ilumina el camino.",
        f"Entre versos y rimas,\n{prompt.lower()} se convierte en arte,\nun suspiro del alma\nque toca el corazón.",
        f"La poesía de {prompt.lower()}\nfluye como un río,\nlleno de emociones\nque nunca se agotan."
    ]
    return random.choice(poems)

def generate_explanation(prompt: str) -> str:
    """Genera una explicación basada en el prompt."""
    explanations = [
        f"{prompt} es un concepto fascinante que representa la creatividad y la innovación en su máxima expresión. Es algo que trasciende los límites convencionales.",
        f"Para entender {prompt}, debemos explorar sus múltiples dimensiones. Es como un diamante con muchas facetas, cada una revelando algo nuevo y sorprendente.",
        f"{prompt} es la manifestación de ideas que transforman la realidad. Es el puente entre lo que imaginamos y lo que podemos crear."
    ]
    return random.choice(explanations)

def generate_description(prompt: str) -> str:
    """Genera una descripción basada en el prompt."""
    descriptions = [
        f"{prompt} se presenta como una experiencia única y memorable. Sus características lo hacen especial y digno de ser recordado.",
        f"Al observar {prompt}, uno puede apreciar la belleza en los detalles. Es como un cuadro que revela nuevos matices con cada mirada.",
        f"{prompt} tiene una presencia que llena el espacio con su energía. Es imposible ignorarlo, y una vez que lo conoces, nunca lo olvidas."
    ]
    return random.choice(descriptions)

def generate_general_response(prompt: str) -> str:
    """Genera una respuesta general basada en el prompt."""
    responses = [
        f"El prompt '{prompt}' ha inspirado una respuesta creativa que combina imaginación y lógica. Es fascinante cómo las palabras pueden abrir puertas a nuevos mundos.",
        f"Cuando pienso en '{prompt}', mi mente se llena de posibilidades infinitas. Es como tener una llave que abre múltiples puertas de la creatividad.",
        f"'{prompt}' es más que solo palabras; es un portal a la imaginación. Cada vez que lo considero, descubro algo nuevo y emocionante."
    ]
    return random.choice(responses) 