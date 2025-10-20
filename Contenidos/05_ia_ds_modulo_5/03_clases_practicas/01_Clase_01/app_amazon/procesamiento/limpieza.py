import string

def limpiar_texto(texto):
    
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    tokens_limpios = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens_limpios)

    