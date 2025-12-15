import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

df_new = pd.read_csv("data/new_data.csv")

X_new = df_new[["age","visits","time_on_site","cart_value"]]

y_true = df_new["buy"]

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
    
# Guardar el scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X_new)

y_pred = model.predict(X_scaled)

acc = accuracy_score(y_true, y_pred)

print(f"Precisión en producción: {acc}")

if acc < 0.70:
    print("ALERTA: El modelo requiere reentrenamiento")
else:
    print("El modelo funciona correctamente.")