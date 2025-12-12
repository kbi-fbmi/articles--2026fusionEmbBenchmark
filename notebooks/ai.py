import os

os.environ["KERAS_BACKEND"] = "torch"
import fmlib.fm as fm
import fmlib.io as io
import keras as kr
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from keras import ops
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def evaluate_model(model, data_test):
    x_test, y_test = data_test
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true) * 100
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    # ROC AUC requires probability scores, and works best for binary or one-vs-rest multiclass
    try:
        auc_score = roc_auc_score(
            y_true,
            y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba,
            multi_class="ovo",
        )
    except ValueError:  # Handle cases where there's only one class in y_true or y_pred
        auc_score = np.nan
    conf_matrix = confusion_matrix(y_true, y_pred)

    # ROC curve values
    if y_pred_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    else:
        fpr, tpr, thresholds = None, None, None

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc_score,
        "Confusion Matrix": conf_matrix,
        "FPR": fpr,
        "TPR": tpr,
        "Thresholds": thresholds,
    }

    # Load and concatenate features


def load_fusion_actg(data_path, response_path):
    """Load one-hot encoded sequences from a file."""

    fusion_data = io.load_fusions_from_fusionaitxt(data_path)

    sq1a = fm.extr_key(fusion_data, "sequence1")
    sq1 = np.array([io.convert_sequence_to_onehot_ACGT(seq) for seq in tqdm(sq1a, desc="Encoding sq1")])

    sq2a = fm.extr_key(fusion_data, "sequence2")
    sq2 = np.array([io.convert_sequence_to_onehot_ACGT(seq) for seq in tqdm(sq2a, desc="Encoding sq2")])

    # Concatenate sq1 and sq2 along axis=1 (assuming shape: (N, 20000, 4))
    x_array = np.concatenate([sq1, sq2], axis=1)  # shape: (N, 40000, 4)

    # Load and encode labels
    y = pd.read_csv(response_path, header=None).iloc[:, 0]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Convert to torch tensors for get_fusionai_model input
    x_tensor = torch.tensor(x_array[..., np.newaxis], dtype=torch.float32)  # shape: (N, 40000, 4, 1)
    y_tensor = torch.tensor(y_onehot[: x_array.shape[0]], dtype=torch.float32)

    return x_tensor, y_tensor


def load_fusion_embedings(seq1_path, seq2_path, response_path):
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


def get_fully_connected(input_dim, num_classes=2):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def get_fusionai_model(input_dim=20000, num_classes=2):
    model = Sequential(
        [
            Input(shape=(input_dim, 4, 1)),
            Conv2D(256, (20, 4), activation="relu"),
            Conv2D(32, (200, 1), activation="relu"),
            MaxPooling2D(pool_size=(20, 1), strides=(20, 1)),
            Dropout(0.25),
            Flatten(),
            Dense(32, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def evaluate_model(model, data_test):
    x_test, y_test = data_test
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true) * 100
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    # ROC AUC requires probability scores, and works best for binary or one-vs-rest multiclass
    try:
        auc_score = roc_auc_score(
            y_true,
            y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba,
            multi_class="ovo",
        )
    except ValueError:  # Handle cases where there's only one class in y_true or y_pred
        auc_score = np.nan
    conf_matrix = confusion_matrix(y_true, y_pred)

    # ROC curve values
    if y_pred_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    else:
        fpr, tpr, thresholds = None, None, None

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc_score,
        "Confusion Matrix": conf_matrix,
        "FPR": fpr,
        "TPR": tpr,
        "Thresholds": thresholds,
    }


def split_test(test_tuple, split_ratio=0.5, random_state=42):
    X, y = test_tuple
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )
    return (X_train, y_train), (X_val, y_val)


def train_adam(data_train, data_val, model, num_epochs=200, batch_size=256, verbose=False):
    x_train, y_train = data_train
    x_val, y_val = data_val
    # optimizer=Adam(learning_rate=3e-4, beta_2=0.99),
    model.compile(
        optimizer="adadelta",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
        # callbacks=[early_stop]
    )

    val_acc = history.history["val_accuracy"]
    best_epoch = np.argmax(val_acc) + 1
    best_acc = val_acc[best_epoch - 1] * 100

    print(f"\nValidation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    return model, history


def train_and_evaluate_svm(
    train_data,
    test_data,
    val_data=None,
    max_samples=10000,
    kernels=None,
    random_state=42,
    cache_size=500,
    verbose=True,
    n_jobs=-1,
):
    """
    Train and evaluate SVM with different kernels.

    Args:
        train_data: Tuple of (X_train, y_train) where y_train is one-hot encoded
        test_data: Tuple of (X_test, y_test) where y_test is one-hot encoded
        val_data: Optional tuple of (X_val, y_val) - currently unused but kept for compatibility
        max_samples: Maximum number of training samples to use
        kernels: List of kernel configurations to try
        random_state: Random state for reproducibility
        cache_size: SVM cache size in MB
        verbose: Whether to print progress

    Returns:
        dict: Evaluation results with best SVM metrics
    """

    # Default kernels if none provided
    if kernels is None:
        kernels = [
            {"name": "Linear", "kernel": "linear", "C": 1.0},
            {"name": "RBF", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
        ]

    # Prepare data (convert from tensors if needed and flatten y to 1D labels)
    x_train_np = train_data[0].numpy() if hasattr(train_data[0], "numpy") else train_data[0]
    y_train_np = (
        train_data[1].numpy().argmax(axis=1) if hasattr(train_data[1], "numpy") else train_data[1].argmax(axis=1)
    )

    x_test_np = test_data[0].numpy() if hasattr(test_data[0], "numpy") else test_data[0]
    y_test_np = test_data[1].numpy().argmax(axis=1) if hasattr(test_data[1], "numpy") else test_data[1].argmax(axis=1)

    # Subsample training data if too large
    if x_train_np.shape[0] > max_samples:
        indices = np.random.choice(x_train_np.shape[0], max_samples, replace=False)
        x_train_subset = x_train_np[indices]
        y_train_subset = y_train_np[indices]
    else:
        x_train_subset = x_train_np
        y_train_subset = y_train_np

    if verbose:
        print(f"Training SVM with {len(y_train_subset)} samples (from {len(y_train_np)} total)...")

    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_subset)
    x_test_scaled = scaler.transform(x_test_np)

    # Try different kernels
    best_score = 0
    best_svm = None
    best_name = None

    for config in kernels:
        config_copy = config.copy()  # Don't modify original
        name = config_copy.pop("name")

        if verbose:
            print(f"Training {name} SVM...")

        svm = SVC(
            probability=True,
            random_state=random_state,
            cache_size=cache_size,
            **config_copy,
        )
        svm.fit(x_train_scaled, y_train_subset)

        score = svm.score(x_test_scaled, y_test_np)
        if verbose:
            print(f"{name} SVM accuracy: {score:.3f}")

        if score > best_score:
            best_score = score
            best_svm = svm
            best_name = name

    if verbose:
        print(f"\nBest performing kernel: {best_name} with accuracy: {best_score:.3f}")

    # Evaluate best model
    y_pred = best_svm.predict(x_test_scaled)
    y_proba = best_svm.predict_proba(x_test_scaled)[:, 1]

    # Calculate all metrics
    svm_accuracy = accuracy_score(y_test_np, y_pred)
    svm_precision = precision_score(y_test_np, y_pred)
    svm_recall = recall_score(y_test_np, y_pred)
    svm_f1 = f1_score(y_test_np, y_pred)
    svm_roc_auc = roc_auc_score(y_test_np, y_proba)
    svm_conf_matrix = confusion_matrix(y_test_np, y_pred)
    svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test_np, y_proba)

    results = {
        "Accuracy": svm_accuracy,
        "Precision": svm_precision,
        "Recall": svm_recall,
        "F1 Score": svm_f1,
        "ROC AUC": svm_roc_auc,
        "Confusion Matrix": svm_conf_matrix,
        "FPR": svm_fpr,
        "TPR": svm_tpr,
        "Thresholds": svm_thresholds,
        "Best Kernel": best_name,
        "Model": best_svm,
        "Scaler": scaler,
    }

    if verbose:
        print(f"SVM training completed. Final accuracy: {svm_accuracy:.3f}")

    return results
