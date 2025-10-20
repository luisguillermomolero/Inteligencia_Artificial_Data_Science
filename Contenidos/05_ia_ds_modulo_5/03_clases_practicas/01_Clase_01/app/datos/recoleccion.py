import pandas as pd

def cargar_datos():
    reseñas = [
        "Me encantó el producto, lo recomiendo totalmente",            # Positiva
        "Es una pérdida de dinero, muy malo",                          # Negativa
        "Excelente calidad y entrega rápida",                          # Positiva
        "Horrible, llegó dañado y sin caja",                           # Negativa
        "Producto decente, aunque esperaba más",                       # Positiva
        "Muy contento con la compra, funciona perfecto",               # Positiva
        "No sirve, se rompió el primer día",                           # Negativa
        "Satisfecho con la calidad, lo usaré nuevamente"               # Positiva
    ]
    
    etiquetas = [1, 0, 1, 0, 1, 1, 0, 1]
    
    return pd.DataFrame({'texto': reseñas, 'sentimiento': etiquetas})