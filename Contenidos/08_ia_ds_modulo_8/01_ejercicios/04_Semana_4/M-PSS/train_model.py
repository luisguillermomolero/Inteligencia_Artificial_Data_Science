# =============================
# ETAPA 1: CARGA DE DATOS
# =============================

# Importamos pandas para manejar el DataFrame.
import pandas as pd

# sklearn ofrece herramientas para dividir datos en train/test.
from sklearn.model_selection import train_test_split

# StandardScaler se usa para normalizar las características numéricas.
from sklearn.preprocessing import StandardScaler

# Modelo base de clasificación: Regresión Logística.
from sklearn.linear_model import LogisticRegression

# pickle permite guardar modelos y objetos de Python en archivos.
import pickle

# Cargamos el archivo CSV con los datos de entrenamiento.
# Aquí comienza la etapa de INGESTA.
df = pd.read_csv("data/sales_data.csv")


# =============================
# ETAPA 2: PREPROCESAMIENTO
# =============================

# Seleccionamos las columnas que serán las FEATURES (variables independientes).
# Estas características se usarán para predecir si el usuario comprará.
X = df[["age", "visits", "time_on_site", "cart_value"]]

# Seleccionamos la variable objetivo (LABEL), que indica si compró (1) o no (0).
y = df["buy"]

# Creamos un objeto StandardScaler para normalizar los datos.
# Esto mejora el desempeño de muchos modelos, incluida la Regresión Logística.
scaler = StandardScaler()

# Ajustamos el scaler a los datos y transformamos X.
# fit_transform() aprende las medias y desviaciones estándar y luego escala los datos.
X_scaled = scaler.fit_transform(X)

# Dividimos los datos en entrenamiento y prueba.
# test_size=0.2 significa que el 20% se usa para evaluar.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# =============================
# ETAPA 3: ENTRENAMIENTO DEL MODELO
# =============================

# Creamos una instancia del modelo de Regresión Logística.
model = LogisticRegression()

# Entrenamos el modelo usando los datos escalados.
model.fit(X_train, y_train)

# Mensaje informativo al finalizar la fase de entrenamiento.
print("Entrenamiento finalizado.")


# =============================
# ETAPA 4: GUARDAR MODELO Y PREPROCESADOR
# =============================

# Guardamos el modelo entrenado en un archivo binario usando pickle.
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Guardamos también el scaler, ya que lo necesitaremos para la API y nuevas predicciones.
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Mensaje final indicando que los artefactos del modelo están listos.
print("Modelo y preprocesador guardados.")
