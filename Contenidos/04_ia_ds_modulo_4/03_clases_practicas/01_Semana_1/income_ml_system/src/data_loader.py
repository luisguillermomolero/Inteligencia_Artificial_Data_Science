import openml
import pandas as pd

DATASET_ID = 1590

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = openml.datasets.get_dataset(DATASET_ID)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y

