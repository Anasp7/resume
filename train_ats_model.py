"""
Train ATS Scorer Model
======================
Run this manually when you're ready to activate the model:

    python train_ats_model.py

Requirements:
    pip install xgboost scikit-learn

Reads: data/resume_dataset.jsonl
Saves: models/ats_scorer.pkl
"""

import json
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_ats")

DATASET_PATH = Path("data/resume_dataset.jsonl")
MODEL_PATH   = Path("models/ats_scorer.pkl")
MIN_SAMPLES  = 20  # lowered for faster first train; 50+ for production quality

TASKS = {
    "ats_parseable":         "binary",
    "content_quality":       "regression",
    "skill_match_potential": "regression",
    "hallucination_risk":    "binary",
    "improvement_delta":     "regression",
    "role_fit":              "multiclass",
}


def load_dataset():
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        sys.exit(1)

    records = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info("Loaded %d records", len(records))
    if len(records) < MIN_SAMPLES:
        logger.error("Need at least %d samples, have %d. Keep collecting!", MIN_SAMPLES, len(records))
        sys.exit(1)
    return records


def build_features(records):
    """Extract features and labels from dataset records."""
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    from core.ats_model import extract_features

    X_struct, X_texts, Y = [], [], {task: [] for task in TASKS}

    for r in records:
        inp = r.get("input_resume", "")
        out = r.get("output_resume", "")
        role = r.get("target_role", "")
        labels = r.get("labels", {})

        feats = extract_features(inp, out, role)
        X_struct.append(feats)
        X_texts.append(inp)

        for task in TASKS:
            Y[task].append(labels.get(task, 0))

    return X_struct, X_texts, Y


def train():
    try:
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
    except ImportError:
        logger.error("Missing dependencies. Run: pip install xgboost scikit-learn")
        sys.exit(1)

    records = load_dataset()
    X_struct, X_texts, Y = build_features(records)

    # TF-IDF on input text (500 features to stay lightweight)
    logger.info("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X_texts).toarray()

    X = np.hstack([np.array(X_struct), X_tfidf])
    logger.info("Feature matrix: %s", X.shape)

    models = {}
    for task, kind in TASKS.items():
        y = Y[task]
        logger.info("Training %s (%s)...", task, kind)

        if kind == "binary":
            m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42)
            m.fit(X, [int(v) for v in y])

        elif kind == "regression":
            m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=42)
            m.fit(X, [float(v) for v in y])

        elif kind == "multiclass":
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="mlogloss",
                              random_state=42)
            m.fit(X, y_enc)
            m.classes_ = le.classes_  # attach for inverse-transform in predict

        models[task] = m
        logger.info("  ✓ %s done", task)

    # Save bundle
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"models": models, "tfidf": tfidf}
    with MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    logger.info("✅ Model saved to %s", MODEL_PATH)
    logger.info("Activate by restarting the server — model will load automatically.")


if __name__ == "__main__":
    train()
