from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent
CUSTOM_MODELS_DIR = ROOT / "models" / "custom_models"
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(name: str) -> str:
    """Sanitize user-provided model name for safe filesystem storage."""
    clean = re.sub(r"[^\w\-]", "_", str(name).strip())
    return clean[:50] if clean else "custom_model_v1"


def inspect_dataset_for_training(df: pd.DataFrame, target_col: str):
    """
    Inspect a user-uploaded dataset for binary classification readiness.
    Returns:
        is_valid (bool): Whether the dataset is suitable for training.
        issues (list): List of error/warning strings.
        meta (dict): Detected metadata (classes, counts, missing values, column types).
    """
    issues = []
    meta = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "numeric_cols": [],
        "categorical_cols": [],
        "target_classes": [],
        "class_counts": {},
        "target_col": target_col,
    }

    # 1. Row count validation
    if len(df) < 30:
        issues.append(f"Dataset has only {len(df)} rows. Deep learning requires at least 30 rows (recommended 100+).")

    # 2. Target presence
    if not target_col or target_col not in df.columns:
        issues.append(f"Selected target column '{target_col}' was not found in the dataset.")
        return False, issues, meta

    # 3. Target values check
    clean_target = df[target_col].dropna()
    unique_classes = clean_target.unique().tolist()
    meta["target_classes"] = [str(c) for c in unique_classes]
    meta["class_counts"] = {str(k): int(v) for k, v in clean_target.value_counts().items()}

    if len(unique_classes) == 0:
        issues.append(f"Target column '{target_col}' contains only null values.")
        return False, issues, meta

    if len(unique_classes) == 1:
        issues.append(
            f"Target column '{target_col}' contains only 1 class ({unique_classes[0]}). "
            "Supervised learning requires both Normal and Failure examples."
        )
        return False, issues, meta

    if len(unique_classes) > 2:
        issues.append(
            f"Target column '{target_col}' contains {len(unique_classes)} classes: {unique_classes[:5]}. "
            "This project's Deep Learning trainer supports Binary Classification (2 classes: e.g. Normal vs Failure, 0 vs 1). "
            "Please upload a binary labelled dataset or map your labels to 2 classes before training."
        )
        return False, issues, meta

    # Check minority class count for stratified splitting
    min_count = min(clean_target.value_counts().values)
    if min_count < 4:
        issues.append(
            f"Minority class in '{target_col}' has only {min_count} samples. "
            "At least 4 samples per class are required for train/val/test splitting."
        )
        return False, issues, meta

    # 4. Feature analysis
    candidate_features = [c for c in df.columns if c != target_col]
    for col in candidate_features:
        # Check if ID-like column (all unique or sequential integers)
        if df[col].nunique() == len(df) and len(df) > 20:
            continue  # Likely an ID column, recommend skipping
        if pd.api.types.is_numeric_dtype(df[col]):
            meta["numeric_cols"].append(col)
        else:
            meta["categorical_cols"].append(col)

    if not meta["numeric_cols"] and not meta["categorical_cols"]:
        issues.append("No valid feature columns found for training.")
        return False, issues, meta

    return True, issues, meta


def train_custom_neural_network(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    model_name: str,
    epochs: int = 50,
    batch_size: int = 32,
    progress_callback=None
):
    """
    Train a genuine TensorFlow/Keras MLP on a user's custom labelled dataset.
    Saves the model and preprocessing artifacts into models/custom_models/<model_name>/
    NEVER touches the original AI4I model.
    """
    clean_name = sanitize_model_name(model_name)
    save_dir = CUSTOM_MODELS_DIR / clean_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean and prepare data
    work_df = df[feature_cols + [target_col]].dropna().copy()
    if len(work_df) < 25:
        raise ValueError("Too few non-null rows remaining after dropping missing values.")

    # 2. Binary target mapping
    unique_targets = list(work_df[target_col].unique())
    # Sort or heuristic: if one looks like 'fail', '1', 'fault', map it to 1
    def is_positive(val):
        s = str(val).lower().strip()
        return s in ["1", "true", "failure", "fail", "fault", "defect", "abnormal", "yes", "y", "positive"]

    if any(is_positive(u) for u in unique_targets):
        pos_class = next(u for u in unique_targets if is_positive(u))
        neg_class = next(u for u in unique_targets if u != pos_class)
    else:
        neg_class, pos_class = sorted(unique_targets, key=str)

    class_mapping = {neg_class: 0, pos_class: 1}
    reverse_mapping = {"0": str(neg_class), "1": str(pos_class)}
    y = work_df[target_col].map(class_mapping).to_numpy(dtype=int)

    # 3. Categorize features
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # 4. Train / Val / Test Split (Stratified 70 / 15 / 15)
    indices = np.arange(len(work_df))
    try:
        idx_trainval, idx_test = train_test_split(
            indices, test_size=0.15, random_state=42, stratify=y
        )
        y_trainval = y[idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.176, random_state=43, stratify=y_trainval  # 0.176 of 0.85 ~= 0.15
        )
    except Exception:
        # Fallback to non-stratified if minority class count is too small for 3-way split
        idx_trainval, idx_test = train_test_split(indices, test_size=0.15, random_state=42)
        idx_train, idx_val = train_test_split(idx_trainval, test_size=0.176, random_state=43)

    train_df = work_df.iloc[idx_train].copy()
    val_df = work_df.iloc[idx_val].copy()
    test_df = work_df.iloc[idx_test].copy()

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    # 5. Fit Preprocessing ONLY on training data
    scaler = None
    encoder = None
    processed_feature_names = []

    # Process numeric
    X_train_num, X_val_num, X_test_num = None, None, None
    if numeric_cols:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(train_df[numeric_cols])
        X_val_num = scaler.transform(val_df[numeric_cols])
        X_test_num = scaler.transform(test_df[numeric_cols])
        processed_feature_names.extend(numeric_cols)

    # Process categorical
    X_train_cat, X_val_cat, X_test_cat = None, None, None
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = encoder.fit_transform(train_df[categorical_cols].astype(str))
        X_val_cat = encoder.transform(val_df[categorical_cols].astype(str))
        X_test_cat = encoder.transform(test_df[categorical_cols].astype(str))
        cat_names = encoder.get_feature_names_out(categorical_cols).tolist()
        processed_feature_names.extend(cat_names)

    # Combine parts
    def assemble(num_part, cat_part):
        if num_part is not None and cat_part is not None:
            return np.hstack([num_part, cat_part])
        return num_part if num_part is not None else cat_part

    X_train_proc = assemble(X_train_num, X_train_cat)
    X_val_proc = assemble(X_val_num, X_val_cat)
    X_test_proc = assemble(X_test_num, X_test_cat)

    input_dim = X_train_proc.shape[1]

    # 6. Deep Learning Architecture (MLP dynamically adapted to input dimension)
    units_1 = min(64, max(16, input_dim * 4))
    units_2 = min(32, max(8, input_dim * 2))
    units_3 = min(16, max(4, input_dim))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="custom_features"),
        tf.keras.layers.Dense(units_1, activation="relu", name="dense_1"),
        tf.keras.layers.BatchNormalization(name="batch_norm"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        tf.keras.layers.Dense(units_2, activation="relu", name="dense_2"),
        tf.keras.layers.Dropout(0.15, name="dropout_2"),
        tf.keras.layers.Dense(units_3, activation="relu", name="dense_3"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_probability"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Compute class weights for imbalance handling
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]

    # Train
    history = model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=min(epochs, 80),
        batch_size=min(batch_size, max(8, len(train_df) // 4)),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    # 7. Select Decision Threshold on Validation Set
    val_probs = model.predict(X_val_proc, verbose=0).ravel()
    thresholds = np.arange(0.10, 0.91, 0.02)
    best_thresh = 0.50
    best_f1 = -1.0
    for t in thresholds:
        score = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    # 8. Evaluate on Untouched Test Set
    test_probs = model.predict(X_test_proc, verbose=0).ravel()
    test_preds = (test_probs >= best_thresh).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))

    # ROC-AUC if both classes exist in test set
    roc_auc = None
    if len(np.unique(y_test)) == 2:
        try:
            roc_auc = round(float(roc_auc_score(y_test, test_probs)), 4)
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1]).tolist()

    test_metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "test_records": int(len(test_df)),
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "threshold": round(best_thresh, 3),
    }

    # 9. Save all artifacts in isolated folder
    model.save(save_dir / "model.keras")
    if scaler is not None:
        joblib.dump(scaler, save_dir / "scaler.joblib")
    if encoder is not None:
        joblib.dump(encoder, save_dir / "encoder.joblib")

    config = {
        "model_name": clean_name,
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_col": target_col,
        "positive_class_name": str(pos_class),
        "negative_class_name": str(neg_class),
        "class_mapping": {str(k): int(v) for k, v in class_mapping.items()},
        "reverse_mapping": reverse_mapping,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "processed_feature_names": processed_feature_names,
        "input_dim": input_dim,
        "threshold": round(best_thresh, 3),
        "metrics": test_metrics,
        "history": {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "loss": [round(float(v), 5) for v in history.history["loss"]],
            "val_loss": [round(float(v), 5) for v in history.history.get("val_loss", [])],
            "accuracy": [round(float(v), 5) for v in history.history["accuracy"]],
            "val_accuracy": [round(float(v), 5) for v in history.history.get("val_accuracy", [])],
        },
    }

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config, save_dir


def list_custom_models():
    """Scan and list all successfully saved custom models."""
    models = []
    if not CUSTOM_MODELS_DIR.exists():
        return models

    for sub in sorted(CUSTOM_MODELS_DIR.iterdir()):
        cfg_file = sub / "config.json"
        model_file = sub / "model.keras"
        if sub.is_dir() and cfg_file.exists() and model_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["dir_path"] = str(sub)
                models.append(cfg)
            except Exception:
                continue
    return models


def predict_with_custom_model(model_name: str, input_df: pd.DataFrame):
    """
    Run batch inference on new data using a previously saved custom model.
    Applies the exact fitted scaler, encoder, and threshold stored in config.
    """
    clean_name = sanitize_model_name(model_name)
    model_dir = CUSTOM_MODELS_DIR / clean_name
    cfg_file = model_dir / "config.json"
    model_file = model_dir / "model.keras"

    if not cfg_file.exists() or not model_file.exists():
        raise FileNotFoundError(f"Custom model '{model_name}' not found at {model_dir}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Validate required feature columns
    req_features = cfg["feature_cols"]
    missing = [c for c in req_features if c not in input_df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing features expected by model '{model_name}': {missing}\n"
            f"Expected features: {req_features}"
        )

    # Extract required columns
    df_clean = input_df[req_features].copy()
    numeric_cols = cfg.get("numeric_cols", [])
    categorical_cols = cfg.get("categorical_cols", [])

    # Apply scaler if present
    X_num = None
    if numeric_cols:
        scaler_file = model_dir / "scaler.joblib"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
            for c in numeric_cols:
                df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)
            X_num = scaler.transform(df_clean[numeric_cols])

    # Apply categorical encoder if present
    X_cat = None
    if categorical_cols:
        encoder_file = model_dir / "encoder.joblib"
        if encoder_file.exists():
            encoder = joblib.load(encoder_file)
            X_cat = encoder.transform(df_clean[categorical_cols].astype(str))

    # Combine
    if X_num is not None and X_cat is not None:
        X_proc = np.hstack([X_num, X_cat])
    elif X_num is not None:
        X_proc = X_num
    else:
        X_proc = X_cat

    # Load neural network and predict
    model = tf.keras.models.load_model(model_file)
    raw_probs = model.predict(X_proc, verbose=0).ravel()

    threshold = float(cfg.get("threshold", 0.50))
    preds = (raw_probs >= threshold).astype(int)

    pos_name = cfg.get("positive_class_name", "Failure")
    neg_name = cfg.get("negative_class_name", "Normal")

    output_df = input_df.copy()
    output_df["Predicted_Condition"] = np.where(preds == 1, "Failure", "Normal")
    output_df["Predicted_Class"] = np.where(preds == 1, pos_name, neg_name)
    output_df["Failure_Score_%"] = np.round(raw_probs * 100.0, 2)
    output_df["Threshold_Applied"] = round(threshold, 3)

    return output_df, cfg
