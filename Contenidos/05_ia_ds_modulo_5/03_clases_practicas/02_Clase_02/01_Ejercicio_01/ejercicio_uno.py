import random

sabores = ["chocolate", "vainilla", "fresa", "limón", "café"]
toppings = ["crema", "fresas", "chispas", "caramelo", "nueces"]

def generar_cupcake():
    sabor = random.choice(sabores)
    topping = random.choice(toppings)
    return f"Cupcake de {sabor} con topping de {topping}"

for _ in range(5):
    print(generar_cupcake())