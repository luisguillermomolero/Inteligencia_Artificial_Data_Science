import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generar datos simulados
np.random.seed(42)
inversion_marketing = np.random.uniform(5, 50, 60)
ventas = 4.2 * inversion_marketing + np.random.normal(0, 8, 60)
data = pd.DataFrame({
    'Inversion_Marketing': inversion_marketing,
    'Ventas_Mensuales': ventas
})

# Definir las variables independientes y la variable objetivo(target, Etiqueta, dependiente)
# Variable Independiente
X = data[['Inversion_Marketing']]
# Variable dependiente
y = data['Ventas_Mensuales']

# Separación de datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Instancia del modelo de regresión lineal
modelo_regresion = LinearRegression()

# Entrenamiento del modelo de manera que aprenda la relación de "y" con respecto a "X"
modelo_regresion.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = modelo_regresion.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Absoluto Medio (MAE): {mae:.2f}") # ej: 25.38
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (r2): {r2:.2f}")

# Visualización de datos
plt.scatter(X, y, label="Datos reales")
plt.plot(X, modelo_regresion.predict(X), color="red", label="Modelo")
plt.xlabel("Inversión en Marketing")
plt.ylabel("Ventas mensuales")

plt.legend()
plt.show()

