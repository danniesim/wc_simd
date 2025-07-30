#!/usr/bin/env python3
"""
Minimal Flask service for Dedupe operations.
Handles the heavy lifting of Dedupe processing in a separate process.
"""

import dedupe
import logging
import pandas as pd
import time
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global Dedupe instance (loaded once on startup)
dedupe = None
train_data = None

# File paths
dedup_data_file = "data/dn_labels_dedup_data.csv"
training_file = "data/dn_label_dedupe_training.json"
settings_file = "data/dn_label_dedupe_settings.bin"

fields = [
    dedupe.variables.String("label"),
    dedupe.variables.String("type")
]


def initialize_dedupe():
    """Initialize the Dedupe instance"""
    global dedupe, train_data

    logger.info("Initializing Dedupe...")
    dedupe = dedupe.Dedupe(fields, num_cores=8)

    # Load data
    dedup_data = pd.read_csv(dedup_data_file, index_col=0)
    # Ensure label and type columns are strings
    dedup_data['label'] = dedup_data['label'].astype(str)
    dedup_data['type'] = dedup_data['type'].astype(str)
    train_data = {idx: {"label": row.label, "type": row.type}
                  for idx, row in dedup_data.iterrows()}

    # Load existing training data if available
    import os
    if os.path.exists(training_file):
        logger.info(f"Loading existing training data from {training_file}")
        with open(training_file, "r") as tf:
            start_time = time.time()
            dedupe.prepare_training(train_data, training_file=tf)
            duration = time.time() - start_time
            logger.info(
                f"prepare_training with existing data completed in {duration:.2f} seconds")

    else:
        logger.info("No existing training data found, starting fresh")
        start_time = time.time()
        dedupe.prepare_training(train_data)
        duration = time.time() - start_time
        logger.info(f"prepare_training completed in {duration:.2f} seconds")

    logger.info("Dedupe initialization complete")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "initialized": dedupe is not None})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get training statistics"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    n_match = len(dedupe.training_pairs.get("match", []))
    n_distinct = len(dedupe.training_pairs.get("distinct", []))

    return jsonify({
        "matches": n_match,
        "distinct": n_distinct,
        "total": n_match + n_distinct
    })


@app.route('/uncertain_pair', methods=['GET'])
def get_uncertain_pair():
    """Get next uncertain pair for labeling"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    try:
        # Find an uncertain pair
        uncertain_pairs = list(dedupe.uncertain_pairs())
        if uncertain_pairs:
            pair = uncertain_pairs[0]  # Get the first available pair
            # Convert pair to serializable format
            record1, record2 = pair
            return jsonify({
                "record1": dict(record1),
                "record2": dict(record2),
                "has_pair": True
            })

        return jsonify(
            {"has_pair": False,
             "message": "No more uncertain pairs available"})

    except Exception as e:
        logger.error(f"Error getting uncertain pair: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/existing_pairs', methods=['GET'])
def get_existing_pairs():
    """Get all existing labeled pairs"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    pairs = []

    # Add match pairs
    for pair in dedupe.training_pairs["match"]:
        record1, record2 = pair
        pairs.append({
            "record1": dict(record1),
            "record2": dict(record2),
            "label": "match"
        })

    # Add distinct pairs
    for pair in dedupe.training_pairs["distinct"]:
        record1, record2 = pair
        pairs.append({
            "record1": dict(record1),
            "record2": dict(record2),
            "label": "distinct"
        })

    return jsonify({"pairs": pairs})


@app.route('/label_pair', methods=['POST'])
def label_pair():
    """Label a pair as match/distinct/unsure"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        record1 = data["record1"]
        record2 = data["record2"]
        label = data["label"]  # "match", "distinct", or "unsure"

        # Convert back to tuple format
        record_pair = (record1, record2)

        # Handle labeling
        examples = {"distinct": [], "match": []}
        if label == "unsure":
            examples["match"].append(record_pair)
            examples["distinct"].append(record_pair)
        else:
            examples[label].append(record_pair)

        dedupe.mark_pairs(examples)

        return jsonify(
            {"success": True, "message": f"Pair labeled as {label}"})

    except Exception as e:
        logger.error(f"Error labeling pair: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/remove_pair', methods=['POST'])
def remove_pair():
    """Remove a pair from training data"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        record1 = data["record1"]
        record2 = data["record2"]
        record_pair = (record1, record2)

        original_label = None

        # Check in match pairs
        if record_pair in dedupe.training_pairs["match"]:
            dedupe.training_pairs["match"].remove(record_pair)
            original_label = "match"

        # Check in distinct pairs
        if record_pair in dedupe.training_pairs["distinct"]:
            dedupe.training_pairs["distinct"].remove(record_pair)
            original_label = "distinct"

        return jsonify(
            {"success": True, "original_label": original_label,
             "message": f"Pair removed from {original_label} category"
             if original_label else "Pair not found"})

    except Exception as e:
        logger.error(f"Error removing pair: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/save_training', methods=['POST'])
def save_training():
    """Save training data to file"""
    if dedupe is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    try:
        with open(training_file, "w") as tf:
            dedupe.write_training(tf)

        return jsonify(
            {"success": True,
             "message": f"Training data saved to {training_file}"})

    except Exception as e:
        logger.error(f"Error saving training data: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize Dedupe on startup
    initialize_dedupe()

    # Run Flask app
    app.run(host='0.0.0.0', port=5051, debug=False)
