# Enhanced Search & Cherry-pick Feature for Dedupe Training

## Overview

The dedupe training interface has been enhanced with powerful search capabilities that support both creating new training pairs and re-labeling existing ones. The system now supports Boolean search operators (AND/OR) and automatically adapts its behavior based on the labeling mode.

## New Features

### 1. Boolean Search Operators

The search functionality now supports:

- **AND operator**: Find records containing ALL search terms
  - Example: `children AND criminal` - finds records with both "children" and "criminal"
- **OR operator**: Find records containing ANY search terms  
  - Example: `criminal OR police` - finds records with either "criminal" or "police"
- **Combined operators**: Mix AND/OR with precedence
  - Example: `children AND (law OR legal)` - finds records with "children" and either "law" or "legal"
- **Simple search**: Single terms or phrases (no operators)
  - Example: `meeting` - finds records containing "meeting"

### 2. Mode-Aware Search

The search interface automatically detects the current labeling mode:

#### New Pairs Mode (Default)

- Searches through raw data (dn_labels_dedup_data.csv)
- Shows individual records for selection
- Allows generating training pairs from selected records
- Interface shows: "🆕 **New pairs mode**: Searching raw data to create new training pairs"

#### Re-label Mode

- Searches through existing labeled pairs (both match and distinct)
- Shows complete pairs with current labels
- Allows selecting existing pairs for re-labeling
- Interface shows: "🔄 **Re-label mode**: Searching existing labeled pairs"

### 3. Enhanced Result Display

#### For Raw Data (New Pairs Mode)

- Shows individual records with highlighting of matched terms
- Field indicators: 📝 (label match), 🏷️ (type match)
- Displays which terms were matched
- Checkbox selection for creating pairs

#### For Existing Pairs (Re-label Mode)

- Shows complete pairs side-by-side
- Current label with color coding: 🟢 (match), 🔴 (distinct)
- Highlights matched terms in both records
- Shows which record(s) contained the search terms
- Checkbox selection for re-labeling

### 4. Improved Labeling Interface

- Shows original labels when re-labeling pairs
- Provides feedback on label changes (e.g., "re-labeled from match to distinct")
- Maintains session state across searches and modes
- Handles removal of old labels before applying new ones

## API Endpoints

### New Flask Service Endpoints

#### `/search_records` (Enhanced)

- **Method**: POST
- **Purpose**: Search raw data with Boolean operators
- **Parameters**:
  - `query`: Search string with optional AND/OR operators
  - `limit`: Maximum results (default 50)
- **Returns**: Records with highlighting and match information

#### `/search_existing_pairs` (New)

- **Method**: POST  
- **Purpose**: Search existing labeled pairs
- **Parameters**:
  - `query`: Search string with optional AND/OR operators
  - `limit`: Maximum results (default 50)
- **Returns**: Existing pairs with match information for both records

## Search Algorithm

The search system uses a sophisticated parsing and evaluation algorithm:

1. **Query Parsing**: Splits search string into tokens and operators
2. **Infix to Postfix**: Converts to postfix notation for proper precedence
3. **Expression Evaluation**: Uses stack-based evaluation with short-circuit logic
4. **Result Highlighting**: Marks matched terms in the display

### Precedence Rules

- `AND` has higher precedence than `OR`
- Use parentheses for explicit grouping (future enhancement)

## Usage Examples

### Creating New Training Pairs

1. Stay in "Interactive Labeling" tab (default mode)
2. Switch to "Search & Cherry-pick" tab
3. Enter search query: `children AND law`
4. Select relevant records
5. Click "Generate Training Pairs"
6. Label the generated pairs

### Re-labeling Existing Pairs

1. Go to "Interactive Labeling" tab
2. Select "Re-label existing pairs" mode in sidebar
3. Switch to "Search & Cherry-pick" tab
4. Enter search query: `criminal OR police`
5. Select pairs you want to re-examine
6. Click "Load Pairs for Re-labeling"
7. Review and update labels as needed

## Technical Implementation

### Backend Changes (`dedupe_service.py`)

- Added `search_existing_pairs()` endpoint
- Enhanced `search_records()` with Boolean operators
- Added query parsing functions:
  - `parse_search_query()`
  - `evaluate_search_query()`
  - `infix_to_postfix()`
  - `evaluate_tokens()`

### Frontend Changes (`dn_label_train_client.py`)

- Refactored `search_and_label()` to be mode-aware
- Added `display_raw_data_results()` for new pair creation
- Added `display_existing_pairs_results()` for re-labeling
- Added `display_cherry_picked_pairs()` for unified labeling interface
- Enhanced `label_cherry_picked_pair()` to handle re-labeling

## Benefits

1. **Precision**: Boolean operators allow precise searches
2. **Efficiency**: Find specific types of records quickly
3. **Flexibility**: Works for both new pairs and re-labeling
4. **User-Friendly**: Visual indicators and helpful tooltips
5. **Comprehensive**: Searches both label and type fields
6. **Scalable**: Handles large datasets with configurable limits

This enhancement significantly improves the training workflow by allowing users to strategically select training examples rather than relying on random uncertain pairs.
