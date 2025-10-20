import matplotlib.pyplot as plt
import seaborn as sns

def graficar_distribucion(df):
    sns.countplot(data=df, x='sentimiento', hue='sentimiento', palette='Set2', legend=False)
    plt.title("Distribución de sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.xticks([0, 1], ['Negativo', 'Positivo'])
    plt.show()

