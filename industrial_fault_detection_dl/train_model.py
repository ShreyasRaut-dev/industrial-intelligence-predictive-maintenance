from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DATA_PATH = DATA_DIR / "ai4i2020.csv"
MODEL_PATH = MODEL_DIR / "fault_mlp.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_PATH = REPORT_DIR / "confusion_matrix.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
CLASS_REPORT_PATH = REPORT_DIR / "classification_report.txt"
HISTORY_PATH = REPORT_DIR / "training_history.csv"
HISTORY_PLOT_PATH = REPORT_DIR / "training_curves.png"
CONFUSION_PLOT_PATH = REPORT_DIR / "confusion_matrix.png"
ROC_PLOT_PATH = REPORT_DIR / "roc_curve.png"
PR_PLOT_PATH = REPORT_DIR / "precision_recall_curve.png"

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURE = "Type"
# Stable one-hot columns keep the training and dashboard feature order identical.
ENCODED_TYPE_FEATURES = ["Type_H", "Type_L", "Type_M"]
MODEL_FEATURES = NUMERIC_FEATURES + ENCODED_TYPE_FEATURES

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_dataset():
    """Load local official CSV, or fetch the UCI dataset through ucimlrepo."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=601)
            X = ds.data.features.copy()
            y = ds.data.targets.copy()
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            df.to_csv(DATA_PATH, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI dataset. Check your internet connection, or "
                "download the official ai4i2020.csv from "
                "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
                "and save it as data/ai4i2020.csv. Original error: " + str(exc)
            ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    column_aliases = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
        "Machine Failure": "Machine failure",
    }
    df = df.rename(columns=column_aliases)
    # Save normalized columns back to CSV
    df.to_csv(DATA_PATH, index=False)
    required = [CATEGORICAL_FEATURE] + NUMERIC_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset missing expected columns: " + ", ".join(missing) +
            ". Use the official UCI ai4i2020.csv file."
        )

    df = df[required].copy()
    df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].astype(str).str.strip().str.upper()
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df[CATEGORICAL_FEATURE].isin(["L", "M", "H"])]
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Dataset contains too few valid records.")
    return df

def normalize_ai4i_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column name variations for AI4I features."""
    mapping = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
        "Machine Failure": "Machine failure",
    }
    renamed = df.copy()
    renamed.columns = [str(c).strip() for c in renamed.columns]
    for orig, target in mapping.items():
        if orig in renamed.columns and target not in renamed.columns:
            renamed = renamed.rename(columns={orig: target})
    return renamed

def encode_features(df):
    """One-hot encode Type and ensure deterministic feature columns/order."""
    df_norm = normalize_ai4i_columns(df)
    missing = [c for c in [CATEGORICAL_FEATURE] + NUMERIC_FEATURES if c not in df_norm.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    numeric = df_norm[NUMERIC_FEATURES].astype(float).reset_index(drop=True)
    type_dummies = pd.get_dummies(
        df_norm[CATEGORICAL_FEATURE].astype(str).str.upper(), prefix="Type"
    )
    for col in ENCODED_TYPE_FEATURES:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
    type_dummies = type_dummies[ENCODED_TYPE_FEATURES].astype(float).reset_index(drop=True)
    return pd.concat([numeric, type_dummies], axis=1)[MODEL_FEATURES]

def build_model(input_dim):
    """A genuine feed-forward deep neural network (MLP) for binary classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="sensor_features"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="failure_probability"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        ],
    )
    return model

# AUREL Industrial Robotics Color Tokens
COLOR_ORANGE = "#E99A35"
COLOR_GRAY = "#777777"
COLOR_CHARCOAL = "#242424"
COLOR_BORDER = "#E7E7E4"
COLOR_BG = "#FFFFFF"

def style_axes(ax):
    """Apply AUREL clean industrial styling to matplotlib axes."""
    ax.set_facecolor(COLOR_BG)
    ax.tick_params(colors=COLOR_CHARCOAL, labelsize=9)
    ax.xaxis.label.set_color(COLOR_CHARCOAL)
    ax.yaxis.label.set_color(COLOR_CHARCOAL)
    ax.title.set_color(COLOR_CHARCOAL)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.grid(True, linestyle="-", alpha=0.6, color=COLOR_BORDER)

def save_training_curves(history_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=COLOR_BG)
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Training loss", color=COLOR_ORANGE, linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", color=COLOR_GRAY, linestyle="--", linewidth=1.8)
    axes[0].set_title("Loss across epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend(frameon=True, facecolor=COLOR_BG, edgecolor=COLOR_BORDER)
    style_axes(axes[0])

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Training accuracy", color=COLOR_ORANGE, linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy", color=COLOR_GRAY, linestyle="--", linewidth=1.8)
    axes[1].set_title("Accuracy across epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(frameon=True, facecolor=COLOR_BG, edgecolor=COLOR_BORDER)
    style_axes(axes[1])

    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)

def save_confusion_plot(cm):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=COLOR_BG)
    aurel_cmap = sns.light_palette(COLOR_ORANGE, as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=aurel_cmap, cbar=False, ax=ax,
        xticklabels=["Predicted normal", "Predicted failure"],
        yticklabels=["Actual normal", "Actual failure"],
        annot_kws={"size": 11, "weight": "bold", "color": COLOR_CHARCOAL}
    )
    ax.set_title("Test-set confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)

def train_and_save():
    set_seeds(42)
    for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X = encode_features(df)
    y = df[TARGET].astype(int).to_numpy()

    # Hold out the test set first. Validation data is used for early stopping and threshold selection.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=43, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(X_train_scaled.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, min_delta=0.0001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0
        ),
    ]
    hist = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = pd.DataFrame(hist.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(HISTORY_PATH, index=False)
    save_training_curves(history_df)

    # Pick the F1-maximizing threshold on validation data only; test set remains untouched.
    val_prob = model.predict(X_val_scaled, verbose=0).ravel()
    thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_scores = [
        f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(threshold_scores))])

    test_prob = model.predict(X_test_scaled, verbose=0).ravel()
    test_pred = (test_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    metrics = {
        "dataset_rows": int(len(df)),
        "normal_records": int((y == 0).sum()),
        "failure_records": int((y == 1).sum()),
        "failure_rate_percent": round(float(y.mean() * 100), 3),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "epochs_trained": int(len(history_df)),
        "threshold": round(best_threshold, 3),
        "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "average_precision": round(float(average_precision_score(y_test, test_prob)), 4),
        "random_state": 42,
        "model": "TensorFlow/Keras MLP",
        "features": MODEL_FEATURES,
        "note": "Test metrics use a held-out stratified test set. The decision threshold was selected on validation data only.",
        "dataset_note": "AI4I 2020 is synthetic data, not live factory sensor data."
    }

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        cm,
        index=["Actual: Normal", "Actual: Failure"],
        columns=["Predicted: Normal", "Predicted: Failure"],
    ).to_csv(CONFUSION_PATH)
    CLASS_REPORT_PATH.write_text(
        classification_report(
            y_test, test_pred, labels=[0, 1],
            target_names=["Normal", "Failure"], zero_division=0
        ),
        encoding="utf-8",
    )
    save_confusion_plot(cm)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=COLOR_BG)
    ax.plot(fpr, tpr, color=COLOR_ORANGE, linewidth=2, label=f"MLP (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_GRAY, label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — held-out test set")
    ax.legend(frameon=True, facecolor=COLOR_BG, edgecolor=COLOR_BORDER)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(ROC_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=COLOR_BG)
    ax.plot(recall, precision, color=COLOR_ORANGE, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve — held-out test set")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(PR_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)

    # Permutation importance is model-agnostic: shuffle one test feature at a time.
    # It is an explanatory approximation, not causal evidence.
    rng = np.random.default_rng(42)
    baseline = f1_score(y_test, test_pred, zero_division=0)
    importances = []
    for col_index, feature in enumerate(MODEL_FEATURES):
        shuffled = X_test_scaled.copy()
        shuffled[:, col_index] = rng.permutation(shuffled[:, col_index])
        shuffled_prob = model.predict(shuffled, verbose=0).ravel()
        shuffled_pred = (shuffled_prob >= best_threshold).astype(int)
        shuffled_f1 = f1_score(y_test, shuffled_pred, zero_division=0)
        importances.append({"feature": feature, "f1_drop_when_shuffled": baseline - shuffled_f1})
    pd.DataFrame(importances).sort_values(
        "f1_drop_when_shuffled", ascending=False
    ).to_csv(IMPORTANCE_PATH, index=False)

    return model, scaler, metrics, df

def load_artifacts():
    needed = [MODEL_PATH, SCALER_PATH, METRICS_PATH, DATA_PATH]
    if not all(p.exists() for p in needed):
        return train_and_save()
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_PATH)
    return model, scaler, metrics, df

if __name__ == "__main__":
    model, scaler, metrics, df = train_and_save()
    print("\nTraining completed.")
    for key in ["dataset_rows", "train_rows", "validation_rows", "test_rows", "epochs_trained",
                "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "threshold"]:
        print(f"{key}: {metrics[key]}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Training graphs and evaluation reports: {REPORT_DIR}")
