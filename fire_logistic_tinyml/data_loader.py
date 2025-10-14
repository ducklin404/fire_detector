import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    assert 'fireStatus' in df.columns, "CSV must contain 'fireStatus' column"
    X = df.drop('fireStatus', axis=1).values.astype(np.float32)
    y = df['fireStatus'].values.astype(np.float32)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)