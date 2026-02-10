import os

os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import torch
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

os.environ["KERAS_BACKEND"] = "torch"


def load_fusion_embeddings(seq1_path, seq2_path, response_path):
    T1 = pd.read_csv(seq1_path, header=None)
    T2 = pd.read_csv(seq2_path, header=None)
    T1.columns = [f"{col}_T1" for col in T1.columns]
    T2.columns = [f"{col}_T2" for col in T2.columns]
    x = pd.concat([T1, T2], axis=1)

    # Load and encode labels
    y = pd.read_csv(response_path, header=None).iloc[:, 0]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Convert to torch tensors
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)

    return x_tensor, y_tensor
