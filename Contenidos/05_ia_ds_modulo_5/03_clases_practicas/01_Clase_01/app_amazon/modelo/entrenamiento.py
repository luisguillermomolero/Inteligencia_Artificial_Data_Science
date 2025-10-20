from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def entrenar_y_evaluar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, pred, target_names=["Negativo", "Positivo"], zero_division=0))
    matriz = confusion_matrix(y_test, pred)
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()