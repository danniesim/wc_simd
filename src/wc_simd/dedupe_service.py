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
dedupe_instance = None
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
    global dedupe_instance, train_data

    logger.info("Initializing Dedupe...")
    dedupe_instance = dedupe.Dedupe(fields, num_cores=8)

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
            dedupe_instance.prepare_training(train_data, training_file=tf)
            duration = time.time() - start_time
            logger.info(
                f"prepare_training with existing data completed in {duration:.2f} seconds")

    else:
        logger.info("No existing training data found, starting fresh")
        start_time = time.time()
        dedupe_instance.prepare_training(train_data)
        duration = time.time() - start_time
        logger.info(f"prepare_training completed in {duration:.2f} seconds")

    logger.info("Dedupe initialization complete")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "initialized": dedupe_instance is not None})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get training statistics"""
    if dedupe_instance is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    n_match = len(dedupe_instance.training_pairs.get("match", []))
    n_distinct = len(dedupe_instance.training_pairs.get("distinct", []))

    return jsonify({
        "matches": n_match,
        "distinct": n_distinct,
        "total": n_match + n_distinct
    })


@app.route('/uncertain_pair', methods=['GET'])
def get_uncertain_pair():
    """Get next uncertain pair for labeling"""
    if dedupe_instance is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    try:
        # Find an uncertain pair
        uncertain_pairs = list(dedupe_instance.uncertain_pairs())
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


@app.route('/existing_pairs', methods=['GET', 'POST'])
def get_existing_pairs():
    """Get all existing labeled pairs with optional filtering"""
    if dedupe_instance is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    # Support both GET and POST, with POST allowing filtering
    label_filter = 'all'  # default
    if request.method == 'POST':
        data = request.get_json()
        if data:
            label_filter = data.get('label_filter', 'all')

    pairs = []

    # Add match pairs (if not filtered out)
    if label_filter in ['all', 'match']:
        for pair in dedupe_instance.training_pairs["match"]:
            record1, record2 = pair
            pairs.append({
                "record1": dict(record1),
                "record2": dict(record2),
                "label": "match"
            })

    # Add distinct pairs (if not filtered out)
    if label_filter in ['all', 'distinct']:
        for pair in dedupe_instance.training_pairs["distinct"]:
            record1, record2 = pair
            pairs.append({
                "record1": dict(record1),
                "record2": dict(record2),
                "label": "distinct"
            })

    return jsonify({
        "pairs": pairs,
        "label_filter": label_filter
    })


@app.route('/label_pair', methods=['POST'])
def label_pair():
    """Label a pair as match/distinct/unsure"""
    if dedupe_instance is None:
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

        dedupe_instance.mark_pairs(examples)

        return jsonify(
            {"success": True, "message": f"Pair labeled as {label}"})

    except Exception as e:
        logger.error(f"Error labeling pair: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/remove_pair', methods=['POST'])
def remove_pair():
    """Remove a pair from training data"""
    if dedupe_instance is None:
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
        if record_pair in dedupe_instance.training_pairs["match"]:
            dedupe_instance.training_pairs["match"].remove(record_pair)
            original_label = "match"

        # Check in distinct pairs
        if record_pair in dedupe_instance.training_pairs["distinct"]:
            dedupe_instance.training_pairs["distinct"].remove(record_pair)
            original_label = "distinct"

        return jsonify(
            {"success": True, "original_label": original_label,
             "message": f"Pair removed from {original_label} category"
             if original_label else "Pair not found"})

    except Exception as e:
        logger.error(f"Error removing pair: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/search_records', methods=['POST'])
def search_records():
    """Search records by text query with AND/OR operators"""
    if train_data is None:
        return jsonify({"error": "Training data not loaded"}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No search query provided"}), 400

    try:
        query = data['query'].strip()
        limit = data.get('limit', 50)  # Default limit of 50 results

        results = []
        count = 0

        for idx, record in train_data.items():
            if count >= limit:
                break

            # Check if record matches the query
            match_result = evaluate_search_query(query, record)

            if match_result['matches']:
                results.append({
                    'idx': idx,
                    'record': record,
                    'match_in': match_result['match_fields'],
                    'matched_terms': match_result['matched_terms']
                })
                count += 1

        return jsonify({
            "results": results,
            "total_found": len(results),
            "limited": len(results) == limit,
            "query_parsed": parse_search_query(query)
        })

    except Exception as e:
        logger.error(f"Error searching records: {e}")
        return jsonify({"error": str(e)}), 500


def parse_search_query(query):
    """Parse search query into tokens and operators"""
    import re

    # Replace multiple spaces with single space and strip
    query = re.sub(r'\s+', ' ', query.strip())

    # Split by AND and OR while preserving the operators
    tokens = []
    current_token = ""
    i = 0

    while i < len(query):
        if query[i:].upper().startswith(' AND '):
            if current_token.strip():
                tokens.append(current_token.strip())
            tokens.append('AND')
            current_token = ""
            i += 5  # length of ' AND '
        elif query[i:].upper().startswith(' OR '):
            if current_token.strip():
                tokens.append(current_token.strip())
            tokens.append('OR')
            current_token = ""
            i += 4  # length of ' OR '
        else:
            current_token += query[i]
            i += 1

    if current_token.strip():
        tokens.append(current_token.strip())

    return tokens


def evaluate_search_query(query, record):
    """Evaluate search query against a record"""
    tokens = parse_search_query(query)

    if not tokens:
        return {'matches': False, 'match_fields': [], 'matched_terms': []}

    # If no operators, treat as simple search
    if 'AND' not in tokens and 'OR' not in tokens:
        return simple_search_match(query.lower(), record)

    # Evaluate the expression
    result = evaluate_tokens(tokens, record)
    return result


def simple_search_match(query, record):
    """Simple search without operators"""
    label_text = record['label'].lower()
    type_text = record['type'].lower()

    label_match = query in label_text
    type_match = query in type_text

    match_fields = []
    matched_terms = []

    if label_match:
        match_fields.append('label')
        matched_terms.append(query)
    if type_match:
        match_fields.append('type')
        matched_terms.append(query)

    return {
        'matches': label_match or type_match,
        'match_fields': match_fields,
        'matched_terms': matched_terms
    }


def evaluate_tokens(tokens, record):
    """Evaluate parsed tokens against record"""
    if not tokens:
        return {'matches': False, 'match_fields': [], 'matched_terms': []}

    # Convert to postfix notation for easier evaluation
    postfix = infix_to_postfix(tokens)

    # Evaluate postfix expression
    stack = []
    all_match_fields = set()
    all_matched_terms = set()

    for token in postfix:
        if token in ['AND', 'OR']:
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()

                if token == 'AND':
                    result = left['matches'] and right['matches']
                else:  # OR
                    result = left['matches'] or right['matches']

                # Combine match fields and terms
                match_fields = list(
                    set(left['match_fields'] + right['match_fields']))
                matched_terms = list(
                    set(left['matched_terms'] + right['matched_terms']))

                stack.append({
                    'matches': result,
                    'match_fields': match_fields,
                    'matched_terms': matched_terms
                })

                if result:
                    all_match_fields.update(match_fields)
                    all_matched_terms.update(matched_terms)
        else:
            # It's a search term
            match_result = simple_search_match(token.lower(), record)
            stack.append(match_result)

            if match_result['matches']:
                all_match_fields.update(match_result['match_fields'])
                all_matched_terms.update(match_result['matched_terms'])

    if stack:
        final_result = stack[0]
        return {
            'matches': final_result['matches'],
            'match_fields': list(all_match_fields) if final_result['matches'] else [],
            'matched_terms': list(all_matched_terms) if final_result['matches'] else []
        }

    return {'matches': False, 'match_fields': [], 'matched_terms': []}


def infix_to_postfix(tokens):
    """Convert infix notation to postfix for easier evaluation"""
    precedence = {'OR': 1, 'AND': 2}
    output = []
    operators = []

    for token in tokens:
        if token in precedence:
            while (operators and
                   operators[-1] in precedence and
                   precedence[operators[-1]] >= precedence[token]):
                output.append(operators.pop())
            operators.append(token)
        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output


@app.route('/search_existing_pairs', methods=['POST'])
def search_existing_pairs():
    """Search existing labeled pairs by text query"""
    if dedupe_instance is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No search query provided"}), 400

    try:
        query = data['query'].strip()
        limit = data.get('limit', 50)
        # 'all', 'match', 'distinct'
        label_filter = data.get('label_filter', 'all')

        results = []
        count = 0

        # Search through match pairs (if not filtered out)
        if label_filter in ['all', 'match']:
            for pair in dedupe_instance.training_pairs.get("match", []):
                if count >= limit:
                    break

                record1, record2 = pair
                record1_dict = dict(record1)
                record2_dict = dict(record2)

                # Check if either record matches the query
                match1 = evaluate_search_query(query, record1_dict)
                match2 = evaluate_search_query(query, record2_dict)

                if match1['matches'] or match2['matches']:
                    results.append({
                        'record1': record1_dict,
                        'record2': record2_dict,
                        'label': 'match',
                        'match_info': {
                            'record1_matches': match1['matches'],
                            'record2_matches': match2['matches'],
                            'record1_match_fields': match1['match_fields'],
                            'record2_match_fields': match2['match_fields'],
                            'record1_matched_terms': match1['matched_terms'],
                            'record2_matched_terms': match2['matched_terms']
                        }
                    })
                    count += 1

        # Search through distinct pairs (if not filtered out)
        if label_filter in ['all', 'distinct']:
            for pair in dedupe_instance.training_pairs.get("distinct", []):
                if count >= limit:
                    break

                record1, record2 = pair
                record1_dict = dict(record1)
                record2_dict = dict(record2)

                # Check if either record matches the query
                match1 = evaluate_search_query(query, record1_dict)
                match2 = evaluate_search_query(query, record2_dict)

                if match1['matches'] or match2['matches']:
                    results.append({
                        'record1': record1_dict,
                        'record2': record2_dict,
                        'label': 'distinct',
                        'match_info': {
                            'record1_matches': match1['matches'],
                            'record2_matches': match2['matches'],
                            'record1_match_fields': match1['match_fields'],
                            'record2_match_fields': match2['match_fields'],
                            'record1_matched_terms': match1['matched_terms'],
                            'record2_matched_terms': match2['matched_terms']
                        }
                    })
                    count += 1

        return jsonify({
            "results": results,
            "total_found": len(results),
            "limited": len(results) == limit,
            "query_parsed": parse_search_query(query),
            "label_filter": label_filter
        })

    except Exception as e:
        logger.error(f"Error searching existing pairs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_record_pairs', methods=['POST'])
def get_record_pairs():
    """Get potential pairs for selected records"""
    if train_data is None:
        return jsonify({"error": "Training data not loaded"}), 500

    data = request.get_json()
    if not data or 'record_indices' not in data:
        return jsonify({"error": "No record indices provided"}), 400

    try:
        record_indices = data['record_indices']
        pairs = []

        # Generate pairs from selected records
        for i, idx1 in enumerate(record_indices):
            for idx2 in record_indices[i + 1:]:
                if idx1 in train_data and idx2 in train_data:
                    pairs.append({
                        'record1': {'idx': idx1, **train_data[idx1]},
                        'record2': {'idx': idx2, **train_data[idx2]}
                    })

        return jsonify({
            "pairs": pairs,
            "total_pairs": len(pairs)
        })

    except Exception as e:
        logger.error(f"Error generating pairs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/save_training', methods=['POST'])
def save_training():
    """Save training data to file"""
    if dedupe_instance is None:
        return jsonify({"error": "Dedupe not initialized"}), 500

    try:
        with open(training_file, "w") as tf:
            dedupe_instance.write_training(tf)

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
