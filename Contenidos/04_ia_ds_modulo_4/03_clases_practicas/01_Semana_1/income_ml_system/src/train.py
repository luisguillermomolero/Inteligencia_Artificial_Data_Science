import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_dataset
from src.preprocessing import build_preprocessor, save_preprocessor
from src.model import build_model
from src.config import TEST_SIZE, SEED

def set_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def train():
    set_seed(SEED)
    os.makedirs("artifacts", exist_ok=True)
    X, y = load_dataset()
    if y.dtype == "category" or y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y).astype(np.float32)
    else:
        y = np.asarray(y, dtype=np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)    
    X_test = preprocessor.transform(X_test)
    
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    X_train = np.asarray(X_train, dtype=np.float32)
    
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    X_test = np.asarray(X_test, dtype=np.float32)
    
    save_preprocessor(preprocessor)
    
    model = build_model(X_train.shape[1])
    
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64
    )
    
    model.save("artifacts/model.keras")
    
if __name__ == "__main__":
    train()
    
    