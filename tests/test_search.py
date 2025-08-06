#!/usr/bin/env python3
"""
Test script for the enhanced search functionality
"""


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


# Test cases
if __name__ == "__main__":
    # Sample test records
    test_records = [
        {'label': 'Children and criminal law', 'type': 'subject'},
        {'label': 'Criminal law', 'type': 'subject'},
        {'label': 'Police meeting', 'type': 'Meeting'},
        {'label': 'Youth services', 'type': 'subject'},
        {'label': 'Legal consultation', 'type': 'Meeting'},
    ]

    # Test queries
    test_queries = [
        "children AND criminal",
        "criminal OR police",
        "meeting",
        "children AND law",
        "legal OR youth",
        "nonexistent"
    ]

    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        tokens = parse_search_query(query)
        print(f"Parsed tokens: {tokens}")

        for i, record in enumerate(test_records):
            result = evaluate_search_query(query, record)
            if result['matches']:
                print(f"Record {i+1}: MATCH - {record} -> {result}")
            else:
                print(f"Record {i+1}: no match - {record}")
