import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/sales_data.csv")

# Preprocesamiento

X = df[["age","visits","time_on_site","cart_value"]]

y = df["buy"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Entrenamiento

model = LogisticRegression()

model.fit(X_train, y_train)

print("Entrenamiento finalizado")

# Guardar el modelo

# Guardar el modelo entrenado
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
    
# Guardar el scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modelo y preprocesador guardados.")