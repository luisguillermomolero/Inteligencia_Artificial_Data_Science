import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Generar los datos simulados
np.random.seed(42)
antiguedad = np.random.randint(1, 10, 120)
uso_servicio = np.random.randint(1, 10, 120)

# definición de la etiqueta(variable objetivo): abandono del cliente
# Reglas del negocio
# Un cliente tiene alta probabilidad de abandonar si:
# - Uso mínimo del servicio (uso < 4)
# - Tiene poca antigüedad (antigüedad < 12)
# El resultado es:
#   - 1 (abandona)
#   - 0 (No abandona)
abandono = ((uso_servicio < 4) & (antiguedad < 12)).astype(int)

data = pd.DataFrame({
    'Antiguedad_Meses': antiguedad,
    'Use_Servicio': uso_servicio,
    'Abandono': abandono
})

# Definición de variables

# Variable independiente
X = data [['Antiguedad_Meses', 'Use_Servicio']]

# Variable dependiente (objetivo)
y = data['Abandono']

# Separación de datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Instancia del modelo de regresión logística
modelo_logistico = LogisticRegression()

# Entrenar el modelo
modelo_logistico.fit(X_train, y_train)

# Predicción y la evaluación del modelo
# 1 -> Abandono; 0 -> No abandono
y_pred = modelo_logistico.predict(X_test)

# Imprimir la matriz de confusión
print("Matriz de confusión")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación
print(f"\nReporte de Clasificación")
print(classification_report(y_test, y_pred))
