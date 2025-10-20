from datos.recoleccion import cargar_datos
from procesamiento.limpieza import limpiar_texto
from procesamiento.exploracion import graficar_distribucion
from procesamiento.vectorizacion import vectorizar_texto
from modelo.entrenamiento import entrenar_y_evaluar

if __name__ == "__main__":
    df = cargar_datos()
    df['texto_limpio'] = df['texto'].apply(limpiar_texto)
    graficar_distribucion(df)
    X, _ = vectorizar_texto(df['texto_limpio'])
    y = df['sentimiento']
    entrenar_y_evaluar(X, y)
    