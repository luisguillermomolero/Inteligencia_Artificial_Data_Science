from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_texto(corpus):
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(corpus)
    return X, vectorizador