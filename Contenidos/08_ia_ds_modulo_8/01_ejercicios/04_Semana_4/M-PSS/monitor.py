# =============================
# MONITOR.PY - MONITOREO DE RENDIMIENTO DEL MODELO
# =============================
# Este script evalúa el rendimiento del modelo en producción usando datos nuevos.
# Se ejecuta periódicamente para detectar si el modelo necesita reentrenamiento.

# Importamos las librerías necesarias
import pandas as pd  # Para manejar DataFrames y leer CSV
import pickle  # Para cargar el modelo y el preprocesador guardados
from sklearn.metrics import accuracy_score  # Para calcular la precisión del modelo

# =============================
# ETAPA 1: CARGA DE DATOS NUEVOS DE PRODUCCIÓN
# =============================
# Cargamos los datos nuevos que han llegado del sistema productivo.
# Estos datos representan casos reales que el modelo debe predecir.
df_new = pd.read_csv("data/new_data.csv")

# Separamos las características (features) que el modelo usa para predecir.
# Estas son las variables independientes: edad, visitas, tiempo en sitio y valor del carrito.
X_new = df_new[["age", "visits", "time_on_site", "cart_value"]]

# Obtenemos los valores reales (labels) que sabemos que ocurrieron.
# Estos son los resultados verdaderos que usaremos para comparar con las predicciones.
y_true = df_new["buy"]

# =============================
# ETAPA 2: CARGA DEL MODELO Y PREPROCESADOR
# =============================
# Cargamos el modelo entrenado que fue guardado previamente.
# Este modelo contiene los parámetros aprendidos durante el entrenamiento.
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargamos el scaler (preprocesador) que normaliza los datos.
# Es importante usar el mismo scaler que se usó durante el entrenamiento
# para mantener la consistencia en el escalado de los datos.
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================
# ETAPA 3: PREPROCESAMIENTO DE DATOS NUEVOS
# =============================
# Normalizamos los datos nuevos usando el mismo scaler del entrenamiento.
# Esto asegura que los datos nuevos estén en la misma escala que los datos de entrenamiento.
X_scaled = scaler.transform(X_new)

# =============================
# ETAPA 4: PREDICCIÓN CON EL MODELO
# =============================
# Usamos el modelo para hacer predicciones sobre los datos nuevos.
# El modelo predice si cada usuario compró (1) o no compró (0).
y_pred = model.predict(X_scaled)

# =============================
# ETAPA 5: EVALUACIÓN DEL RENDIMIENTO
# =============================
# Calculamos la precisión (accuracy) comparando las predicciones con los valores reales.
# La precisión indica qué porcentaje de predicciones fueron correctas.
acc = accuracy_score(y_true, y_pred)

# Mostramos la precisión obtenida en producción.
print(f"Accuracy en producción: {acc}")

# =============================
# ETAPA 6: DECISIÓN Y ALERTA
# =============================
# Evaluamos si el modelo sigue funcionando correctamente.
# Si la precisión es menor al 70%, significa que el modelo ha perdido rendimiento
# y necesita ser reentrenado con datos más recientes.
if acc < 0.70:
    print("ALERTA: el modelo requiere reentrenamiento.")
else:
    print("El modelo funciona correctamente.")
