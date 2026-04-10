"""
Train Parser Model
==================
Run this manually when you're ready to improve parsing accuracy:

    python train_parser_model.py

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm

Reads: data/parser_dataset.jsonl
Saves: models/parser_model/ (spaCy model directory)

The model activates automatically on next restart after training.
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_parser")

DATASET_PATH = Path("data/parser_dataset.jsonl")
MODEL_DIR    = Path("models/parser_model")
MIN_SAMPLES  = 30


def load_dataset():
    if not DATASET_PATH.exists():
        logger.error("Parser dataset not found: %s", DATASET_PATH)
        logger.error("Use the system normally to accumulate parse samples first.")
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

    logger.info("Loaded %d parser records", len(records))
    if len(records) < MIN_SAMPLES:
        logger.error("Need at least %d samples, have %d. Keep using the system!", MIN_SAMPLES, len(records))
        sys.exit(1)
    return records


def build_training_data(records):
    """
    Convert (raw_text, parsed) pairs → spaCy NER training format.
    Tags: NAME, EMAIL, PHONE, SKILL, ORG (company/institution)
    """
    import re
    training = []

    for r in records:
        text   = r.get("raw_text", "")
        parsed = r.get("parsed", {})
        if not text:
            continue

        entities = []

        # Tag name
        name = parsed.get("name", "")
        if name and name in text:
            start = text.index(name)
            entities.append((start, start + len(name), "PERSON"))

        # Tag email
        email = parsed.get("email", "")
        if email and email in text:
            start = text.index(email)
            entities.append((start, start + len(email), "EMAIL"))

        # Tag individual skills
        for skill in parsed.get("skills", [])[:10]:
            for m in re.finditer(re.escape(skill), text, re.I):
                entities.append((m.start(), m.end(), "SKILL"))
                break  # first occurrence only

        if entities:
            training.append((text[:3000], {"entities": entities}))

    logger.info("Built %d training examples", len(training))
    return training


def train():
    try:
        import spacy
        from spacy.training import Example
    except ImportError:
        logger.error("Missing spaCy. Run: pip install spacy && python -m spacy download en_core_web_sm")
        sys.exit(1)

    records      = load_dataset()
    training_data = build_training_data(records)

    if len(training_data) < MIN_SAMPLES:
        logger.error("Not enough valid training examples (%d). Need %d+.", len(training_data), MIN_SAMPLES)
        sys.exit(1)

    logger.info("Loading base model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    # Add/get NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add custom labels
    for _, ann in training_data:
        for _, _, label in ann["entities"]:
            ner.add_label(label)

    # Train
    logger.info("Training NER on %d examples...", len(training_data))
    optimizer = nlp.resume_training()

    for epoch in range(15):
        import random
        random.shuffle(training_data)
        losses = {}
        examples = []
        for text, ann in training_data:
            try:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, ann))
            except Exception:
                continue
        nlp.update(examples, sgd=optimizer, losses=losses)
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d — NER loss: %.4f", epoch + 1, losses.get("ner", 0))

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(str(MODEL_DIR))
    logger.info("✅ Parser model saved to %s", MODEL_DIR)
    logger.info("Restart the server to activate the learned parser fallback.")


if __name__ == "__main__":
    train()
