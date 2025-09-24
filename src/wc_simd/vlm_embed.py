from io import BytesIO
from PIL import Image
import requests
import torch
from transformers.utils.versions import require_version
from transformers import AutoModel
import pandas as pd
import glob
import os
import argparse
import hashlib
from typing import Dict, List, Optional
import time
import threading
from queue import Queue
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

os.environ["TOKENIZERS_PARALLELISM"] = "false"


############################################################
# Argparse for distributed sharding across instances
############################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed images with Qwen2-VL model (sharded by hash)")
    parser.add_argument(
        "--instances", type=int, default=1,
        help="Total number of instances participating")
    parser.add_argument(
        "--instance-no", type=int, default=0,
        help="Zero-based index of this instance")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional max rows per file (debug / sampling)")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for embedding inference")
    parser.add_argument(
        "--input-glob", type=str,
        default="data/works_with_images_no_text_partitioned.parquet/*.parquet",
        help="Glob pattern for input parquet files")
    parser.add_argument(
        "--output-dir", type=str,
        default="data/works_with_images_no_text_partitioned_embedded.parquet",
        help="Directory to write output parquet files")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files")
    parser.add_argument(
        "--allow-shared-gpu", action="store_true",
        help="Allow multiple instances to share a single GPU (disabled by default)")
    parser.add_argument(
        "--prefetch-workers", type=int, default=8,
        help="Number of worker threads for background image prefetch (default: 8)")
    parser.add_argument(
        "--prefetch-buffer", type=int, default=64,
        help="Max number of prefetched (fetched or errored) items buffered ahead of embedding (default: 64)")
    parser.add_argument(
        "--no-prefetch", action="store_true",
        help="Disable background prefetch pipeline and fall back to sequential fetch (debug)")
    parser.add_argument(
        "--fetch-max-inflight", type=int, default=None,
        help="Max simultaneous in-flight HTTP fetch tasks (default: 4 * prefetch-workers)")
    return parser.parse_args()


args = parse_args()
if args.instance_no < 0 or args.instance_no >= args.instances:
    raise ValueError(
        f"instance-no ({args.instance_no}) must be in [0, {args.instances - 1}]")

# Enforce one-instance-per-GPU policy on single GPU systems unless
# explicitly overridden
if torch.cuda.is_available() and torch.cuda.device_count() == 1:
    if args.instances > 1 and args.instance_no > 0 and not args.allow_shared_gpu:
        raise RuntimeError(
            "Single GPU detected (1). instance-no > 0 is not allowed without --allow-shared-gpu. "
            "Either run only instance-no 0, reduce --instances to 1, or pass --allow-shared-gpu to override.")


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL with proper headers to avoid 403 errors."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Python script)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


############################################################
# Device & Model
############################################################
def select_device_for_instance() -> Dict[str, Optional[str]]:
    """Select device (optionally GPU index) based on instance number."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            gpu_index = args.instance_no % num_gpus
            device = f"cuda:{gpu_index}"
            print(f"Using CUDA GPU {gpu_index} / {num_gpus}")
        else:
            device = "cuda"
            print("Using single CUDA GPU")
        return {"device": device, "torch_dtype": torch.float16}
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon MPS")
        return {"device": "mps", "torch_dtype": torch.float16}
    else:
        print("Using CPU")
        return {"device": "cpu", "torch_dtype": torch.float32}


dev_conf = select_device_for_instance()
device = dev_conf["device"]
torch_dtype = dev_conf["torch_dtype"]

# Load model on a single target device (avoid device_map complexities for
# sharded runs)
gme = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype=torch_dtype,
    trust_remote_code=True
)
gme.to(device)
gme.eval()


# glob input parquet files from
# data/works_with_images_no_test_partitioned.parquets
input_files = sorted(glob.glob(args.input_glob))


def file_belongs_to_instance(
        path: str, instances: int, instance_no: int) -> bool:
    """Hash file path (basename) and modulo by instances to decide assignment."""
    h = int(hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest(), 16)
    return (h % instances) == instance_no


selected_files = [
    f for f in input_files if file_belongs_to_instance(
        f, args.instances, args.instance_no)]
print(f"Instance {args.instance_no}/{args.instances} selected {len(selected_files)} / {len(input_files)} files")
if not selected_files:
    print("No files assigned to this instance. Exiting.")
    exit(0)

############################################################
# Embedding Processing (order-preserving & robust)
############################################################

output_files_dir = args.output_dir
os.makedirs(output_files_dir, exist_ok=True)

MAX_ROWS: Optional[int] = args.max_rows
BATCH_SIZE = args.batch_size

for file in selected_files:
    output_file = os.path.join(output_files_dir, os.path.basename(file))
    if os.path.exists(output_file) and not args.overwrite:
        print(
            f"Output file {output_file} already exists, skipping (use --overwrite to force)...")
        continue
    print(f"Processing file {file}")

    df = pd.read_parquet(file)
    total_rows = len(df)
    if MAX_ROWS is not None:
        total_rows = min(total_rows, MAX_ROWS)
    print(f"Loaded {len(df)} rows (processing first {total_rows})")
    print(f"Columns: {df.columns.tolist()}")

    errors: Dict[int, str] = {}

    if args.no_prefetch:
        # Original sequential path (for debug / reproducibility)
        success_indices: List[int] = []
        success_images: List[Image.Image] = []
        print("Fetching images sequentially (no prefetch)...")
        for idx, url in enumerate(df['image_id']):
            if idx >= total_rows:
                break
            try:
                img = load_image_from_url(url)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                success_indices.append(idx)
                success_images.append(img)
            except requests.exceptions.RequestException as e:
                errors[idx] = f"HTTP ERROR: {e}"
            except Exception as e:
                errors[idx] = f"EXCEPTION ERROR: {e}"

        print(
            f"Successfully loaded {len(success_images)} images; {len(errors)} errors")

        embeddings: List[torch.Tensor] = []
        if success_images:
            with torch.inference_mode():
                for start in range(0, len(success_images), BATCH_SIZE):
                    batch_images = success_images[start:start + BATCH_SIZE]
                    e_batch = gme.get_image_embeddings(images=batch_images)
                    assert e_batch.shape[0] == len(
                        batch_images), "Embedding batch size mismatch"
                    embeddings.append(e_batch)
                    print(
                        f"Processed batch {(start // BATCH_SIZE) + 1} / {(len(success_images) - 1)//BATCH_SIZE + 1}")

        if embeddings:
            embeddings_tensor = torch.cat(embeddings, dim=0).cpu()
        else:
            embeddings_tensor = torch.empty(
                (0, gme.config.hidden_size),
                dtype=torch.float32)

        index_to_pos = {
            orig_idx: pos for pos,
            orig_idx in enumerate(success_indices)}
        result_rows = []
        for idx in range(total_rows):
            if idx in errors:
                result_rows.append({
                    'row_index': idx,
                    'image_id': df.iloc[idx]['image_id'],
                    'embedding': None,
                    'error': errors[idx]
                })
            else:
                pos = index_to_pos[idx]
                emb = embeddings_tensor[pos].numpy(
                ) if pos < embeddings_tensor.shape[0] else None
                result_rows.append({
                    'row_index': idx,
                    'image_id': df.iloc[idx]['image_id'],
                    'embedding': emb,
                    'error': None
                })
    else:
        # Background prefetch path
        print(
            f"Prefetching images in background with {args.prefetch_workers} workers (buffer={args.prefetch_buffer})...")
        t_prefetch_start = time.time()

        # Queue items: (idx:int, image:Image|None, error:str|None) ; sentinel
        # idx == -1
        q: Queue = Queue(maxsize=args.prefetch_buffer)
        SENTINEL = (-1, None, None)

        # Shared HTTP session with retry + connection pooling
        retry_conf = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=retry_conf,
            pool_connections=args.prefetch_workers,
            pool_maxsize=(
                args.fetch_max_inflight or (
                    args.prefetch_workers * 4)))
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        def fetch_single(idx: int, url: str):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; Python script)'}
                resp = session.get(
                    url, headers=headers, timeout=(
                        3, 20))  # (connect, read)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return (idx, img, None)
            except Exception as e:  # consolidated (network + decode)
                return (idx, None, f"FETCH ERROR: {e}")

        def producer():
            max_inflight = args.fetch_max_inflight or (
                args.prefetch_workers * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
                next_idx = 0
                futures: Dict[concurrent.futures.Future, int] = {}
                # prime
                while next_idx < total_rows and len(futures) < max_inflight:
                    url = df['image_id'].iloc[next_idx]
                    fut = executor.submit(fetch_single, next_idx, url)
                    futures[fut] = next_idx
                    next_idx += 1
                while futures:
                    done, _pending = concurrent.futures.wait(
                        futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        q.put(fut.result())
                        del futures[fut]
                        # backfill to maintain inflight level
                        if next_idx < total_rows:
                            url = df['image_id'].iloc[next_idx]
                            new_fut = executor.submit(
                                fetch_single, next_idx, url)
                            futures[new_fut] = next_idx
                            next_idx += 1
                # all done
            q.put(SENTINEL)

        threading.Thread(target=producer, daemon=True).start()

        state = {
            'embeddings_map': {},  # idx -> tensor
            'success_count': 0,
            'processed_batches': 0,
            'batch_images': [],  # type: List[Image.Image]
            'batch_indices': []  # type: List[int]
        }

        def embed_current_batch():
            if not state['batch_images']:
                return
            with torch.inference_mode():
                e_batch = gme.get_image_embeddings(
                    images=state['batch_images'])
            assert e_batch.shape[0] == len(
                state['batch_images']), "Embedding batch size mismatch"
            for local_pos, orig_idx in enumerate(state['batch_indices']):
                state['embeddings_map'][orig_idx] = e_batch[local_pos].cpu()
            state['processed_batches'] += 1
            state['success_count'] += len(state['batch_images'])
            print(
                f"Processed batch {state['processed_batches']} (images so far: {state['success_count']})")
            # clear lists in-place
            state['batch_images'].clear()
            state['batch_indices'].clear()

        while True:
            idx_img_err = q.get()
            if idx_img_err == SENTINEL:
                # flush remainder
                embed_current_batch()
                break
            idx_i, img_i, err_i = idx_img_err
            if err_i:
                errors[idx_i] = err_i
            else:
                state['batch_images'].append(img_i)  # type: ignore[arg-type]
                state['batch_indices'].append(idx_i)
                if len(state['batch_images']) >= BATCH_SIZE:
                    embed_current_batch()

        elapsed = time.time() - t_prefetch_start
        embedded_count = len(state['embeddings_map'])
        imgs_per_sec = embedded_count / elapsed if elapsed > 0 else 0.0
        print(
            f"Finished prefetch+embed: {embedded_count} images embedded; {len(errors)} errors; "
            f"elapsed {elapsed:.2f}s; throughput {imgs_per_sec:.2f} img/s")

        # Build result rows using embeddings_map
        result_rows = []
        # type: ignore[assignment]
        embeddings_map: Dict[int, torch.Tensor] = state['embeddings_map']
        for idx in range(total_rows):
            if idx in errors:
                result_rows.append({
                    'row_index': idx,
                    'image_id': df.iloc[idx]['image_id'],
                    'embedding': None,
                    'error': errors[idx]
                })
            else:
                emb_tensor = embeddings_map.get(idx)
                emb = emb_tensor.numpy() if emb_tensor is not None else None
                result_rows.append({
                    'row_index': idx,
                    'image_id': df.iloc[idx]['image_id'],
                    'embedding': emb,
                    'error': None
                })

    out_df = pd.DataFrame(result_rows)

    # Atomic write: write to temp then replace
    tmp_path = output_file + ".tmp"
    out_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_file)
    print(f"Saved {output_file} (rows: {len(out_df)})")
