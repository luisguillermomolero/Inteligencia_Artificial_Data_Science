import numpy as np

real_data = np.random.normal(loc=95, scale=5, size=1000)

def generador(n):
    ruido = np.random.normal(0, 1, n)
    return 80 + ruido * 5

def discriminador(x):
    media_real = np.mean(real_data)
    sigma = 5.0
    return np.exp(-((x -media_real) ** 2) / (2 * sigma ** 2))

for epoch in range(10):
    falsas = generador(5)
    prob_reales = discriminador(np.random.choice(real_data, 5))
    prob_falsas = discriminador(falsas)
    mejora = np.mean(prob_falsas)
    falsas = falsas + (95 - falsas) * mejora * 0.1

    print(f"Epoch {epoch + 1}")
    print(f"Tallas falsas generadas {falsas.round(2)}")
    print(f"Prob. de ser reales (falsas): {prob_falsas.round(2)}")
    print("-" * 40)
    
    