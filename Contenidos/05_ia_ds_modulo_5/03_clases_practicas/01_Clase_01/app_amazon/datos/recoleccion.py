import pandas as pd
from datasets import load_dataset

def cargar_datos(n=100):
    dataset = load_dataset("amazon_polarity", split="train")
    textos = dataset[:n]["content"]
    etiquetas = dataset[:n]["label"]
    df  = pd.DataFrame({"texto": textos, "sentimiento": etiquetas})
    return df
    