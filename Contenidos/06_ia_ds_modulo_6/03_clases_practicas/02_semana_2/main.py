import time
import warnings
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_PREPLEXITY = 30
DEFAULT_LEARNING_RATE = 200
DEFAULT_N_ITER = 1000
DEFAULT_FIGURE_SIZE = (8, 6)

def validar_datos(X, y=None):
    if X is None or len(X) == 0:
        raise ValueError("Los datos de entrada no pueden estar vacios")
    
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    if y is not None and len(y) != len(X):
        raise ValueError("El número de etiquetas debe coincidir con el número de muestras.")
    
    return True

def aplicar_pca(X, n_componentes=2):
    try:
        validar_datos(X)
        if n_componentes > X.shape[1]:
            warnings.warn(
                f"n_componentes ({n_componentes}) es mayor que el número de caracteristicas ({X.shape[1]})."
                f"Se ajustará a {X.shape[1]}"
                )
            n_componentes = X.shape[1]
        
        inicio = time.time()
        
        pca = PCA(n_components=n_componentes, random_state=RANDOM_STATE)
        
        X_pca = pca.fit_transform(X)
        
        tiempo_ejecucion = time.time() - inicio
        
        print(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
        print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.4f}")
        print(f"Tiempo de ejecución PCA: {tiempo_ejecucion:.2f} segundos")
        
        return X_pca, tiempo_ejecucion
        
    except Exception as e:
        print(f"Error en PCA: {e}")
        raise

def aplica_tsne(X, n_componentes=2, perplejidad= DEFAULT_PREPLEXITY, learning_rate=DEFAULT_LEARNING_RATE, n_iter=DEFAULT_N_ITER):
    try:
        validar_datos(X)
        
        if perplejidad >= len(X):
            perplejidad = min(perplejidad, len(X) - 1)
            warnings.warn(f"Perplejidad ajustada a {perplejidad} (máximo) permitido {len(X) - 1}")
        
        inicio = time.time()
        
        tsne = TSNE(
            n_components=n_componentes,
            perplexity=perplejidad,
            learning_rate=learning_rate,
            random_state=RANDOM_STATE,
            n_iter=n_iter
        )
        
        X_tsne = tsne.fit_transform(X)
        
        tiempo_ejecucion = time.time() - inicio
        
        print(f"Tiempo de ejecución de t-SNE: {tiempo_ejecucion:.2f} segundos")
        
        return X_tsne, tiempo_ejecucion
    
    except Exception as e:
        print(f"Error en t-SNE: {e}")
        raise

def graficar_resultados(X_transformado, y, titulo, figura_size=DEFAULT_FIGURE_SIZE):
    """
    Genera un gráfico de dispersión coloreado por etiqueta.
    :param X_transformado: Datos reducidos.
    :param y: Etiquetas de clase.
    :param titulo: Título del gráfico.
    :param figura_size: Tamaño de la figura (ancho, alto).
    """
    try:
        # Validar que los datos de entrada sean correctos
        validar_datos(X_transformado, y)
        
        # Obtener el número de componentes (dimensiones) de los datos transformados
        n_componentes = X_transformado.shape[1]
        
        if n_componentes == 2:
            # Visualización 2D: gráfico de dispersión simple
            plt.figure(figsize=figura_size)  # Crear nueva figura con tamaño especificado
            sns.scatterplot(
                x=X_transformado[:, 0],      # Eje X: primera componente
                y=X_transformado[:, 1],      # Eje Y: segunda componente
                hue=y,                       # Colorear por etiqueta de clase
                palette="tab10",             # Paleta de colores para las clases
                legend="full",               # Mostrar leyenda completa
                alpha=0.7                    # Transparencia de los puntos
            )
            plt.title(titulo, fontsize=14)   # Título del gráfico
            plt.xlabel("Componente 1")       # Etiqueta del eje X
            plt.ylabel("Componente 2")       # Etiqueta del eje Y
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posición de la leyenda
            
        elif n_componentes == 3:
            # Visualización 3D: gráfico de dispersión tridimensional
            fig = plt.figure(figsize=figura_size)  # Crear figura
            ax = fig.add_subplot(111, projection='3d')  # Agregar subplot 3D
            
            # Crear gráfico de dispersión 3D
            scatter = ax.scatter(
                X_transformado[:, 0],        # Coordenada X
                X_transformado[:, 1],        # Coordenada Y
                X_transformado[:, 2],        # Coordenada Z
                c=y,                         # Colorear por etiqueta
                cmap='tab10',                # Mapa de colores
                alpha=0.7                    # Transparencia
            )
            ax.set_title(titulo, fontsize=14)  # Título
            ax.set_xlabel("Componente 1")      # Etiqueta eje X
            ax.set_ylabel("Componente 2")      # Etiqueta eje Y
            ax.set_zlabel("Componente 3")      # Etiqueta eje Z
            
        else:
            # Para más de 3 dimensiones, mostrar solo las primeras 2
            plt.figure(figsize=figura_size)
            sns.scatterplot(
                x=X_transformado[:, 0],      # Primera componente
                y=X_transformado[:, 1],      # Segunda componente
                hue=y,                       # Colorear por clase
                palette="tab10",
                legend="full",
                alpha=0.7
            )
            # Título indicando que solo se muestran 2 de N componentes
            plt.title(f"{titulo} (primeras 2 componentes de {n_componentes})", fontsize=14)
            plt.xlabel("Componente 1")
            plt.ylabel("Componente 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()  # Ajustar layout para evitar solapamientos
        plt.show()          # Mostrar el gráfico
        
    except Exception as e:
        # Si hay algún error, mostrarlo y re-lanzarlo
        print(f"Error en visualización: {e}")
        raise

def main():
    
    try:
        print("=" * 50)
        print("ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD")
        
        print("\n1. Cargando dataset de dígitos manuscritos...")
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        print(f"Dimensiones originales: {X.shape}")
        print(f"Número de clases: {len(np.unique(y))}")
        
        print("\n2. Escalando datos...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Escalado completado")
        
        print("\n3. Aplicando PCA... ")
        X_pca, tiempo_pca = aplicar_pca(X_scaled, n_componentes=2)
        graficar_resultados(X_pca, y, "Visualización con PCA")
        
        print("\n4. Aplicando t-SNE")
        X_tsne, tiempo_tsne = aplica_tsne(
            X_scaled, 
            n_componentes=2,
            perplejidad=DEFAULT_PREPLEXITY,
            learning_rate=DEFAULT_LEARNING_RATE
        )
        
        graficar_resultados(X_tsne, y, "Visualización con t-SNE")
        
        print("\n" + "=" * 50)
        print("RESUMEN DE RENDIMIENTOS")
        print("=" * 50)
        print(f"Tiempo total PCA: {tiempo_pca} segundos")
        print(f"Tiempo total t-SNE: {tiempo_tsne} segundos")
        print(f"Tiempo total {tiempo_pca + tiempo_tsne:.2f} segundos")
        print("\nAnálisis completado")
        
    except Exception as e:
        print(f"Error en el análisis principal: {e}")
    
if __name__ == "__main__":
    main()
