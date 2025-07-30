# Dedupe Service Architecture

The Dedupe training functionality has been split into two components for better performance:

## Components

### 1. Dedupe Service (`dedupe_service.py`)

A Flask backend service that handles the heavy Dedupe training operations:

- Loads and initializes the Dedupe training once at startup
- Provides REST API endpoints for labeling operations
- Maintains Dedupe training state in memory
- No duplicate handling needed (service manages state)

### 2. Streamlit Client (`dn_label_train_client.py`)

A lightweight Streamlit frontend that communicates with the service:

- Provides the same UI for labeling pairs
- Makes HTTP requests to the Flask service
- No longer needs to load/manage Dedupe training directly
- Simplified logic without hashing/deduplication
- No longer needs to load/manage Dedupe training directly

## Usage

### Start the Service

```bash
# Start the Flask Dedupe training service (runs on port 5001)
cd /home/ubuntu/wc_simd
python src/wc_simd/dedupe_service.py
```

The service will initialize the Dedupe training on startup (this takes time) and then be ready to serve requests quickly.

### Run the Streamlit Client

```bash
# In a separate terminal, start the Streamlit app
cd /home/ubuntu/wc_simd
streamlit run src/wc_simd/dn_label_train_client.py
```

### Test the Service

```bash
# Test that the service is working
python test_deduper_service.py
```

## API Endpoints

- `GET /health` - Health check and initialization status
- `GET /stats` - Get training statistics (matches, distinct, total)
- `GET /uncertain_pair` - Get next uncertain pair for labeling
- `GET /existing_pairs` - Get all existing labeled pairs
- `POST /label_pair` - Label a pair as match/distinct/unsure
- `POST /remove_pair` - Remove a pair from training data
- `POST /save_training` - Save training data to file

## Benefits

1. **Faster UI responsiveness** - Streamlit app doesn't need to load Dedupe training
2. **Better resource utilization** - Service can stay running and serve multiple requests
3. **Separation of concerns** - Heavy processing separated from UI
4. **Scalability** - Service could potentially serve multiple clients

## Files

- `src/wc_simd/dedupe_service.py` - Flask backend service
- `src/wc_simd/dn_label_train_client.py` - Streamlit frontend client  
- `test_deduper_service.py` - Service testing script
