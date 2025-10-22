"""Aggregate individual embedded parquet shards into a single NumPy matrix.

This script scans a glob of parquet files that contain at least the columns:
    - image_id (or an override via --id-column)
    - embedding (array[float])

It will:
 1. Filter out null / malformed embeddings
 2. (Optionally) ensure all embeddings have the expected dimensionality
 3. Deduplicate by the id column (first occurrence kept by default)
 4. Produce:
        --output-npy : an *uncompressed* .npy file (shape [N, D])
        --output-index : a CSV / Parquet mapping row_index -> id

NOTE: Previously this script emitted a compressed NPZ (key 'embeddings').
Compression added substantial runtime for large matrices with little size gain,
so this now writes a raw .npy. If you pass a path ending in .npz it will be
silently converted to .npy (with a warning message).

Example:
  python -m wc_simd.vlm_embed_train_data \
      --input-glob "data/works_with_images_no_text_partitioned_embedded.parquet/*.parquet" \
      --output-npy data/vlm_embed/iiif_no_text_embedding_matrix.npy \
      --output-index data/vlm_embed/iiif_no_text_embedding_index.parquet

The script takes a two-pass approach (first counts valid rows, second writes)

"""

from __future__ import annotations

import os
import glob
import json
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import click
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
class Stats:
    total_rows: int = 0
    null_embeddings: int = 0
    bad_dim: int = 0
    duplicates: int = 0
    kept: int = 0


def normalise_embedding(e, dim: int) -> Optional[List[float]]:
    """Return a list[float] of length dim or None if invalid.

    Accepts list/tuple/np.ndarray; attempts coercion to float32. We avoid
    excessive copying by converting directly to np.asarray then tolist().
    """
    if e is None:
        return None
    try:
        arr = np.asarray(e, dtype=np.float32)
    except Exception:
        return None
    if arr.shape != (dim,):
        return None
    return arr.tolist()


def iter_valid_records(
        parquet_files: Sequence[str],
        id_col: str,
        dim: int,
        stats: Stats,
        dedupe: bool,
) -> Iterable[Tuple[str, List[float]]]:
    """Yield (id, embedding_list) for valid embeddings.

    Updates stats in-place. Dedupe uses a set to keep first id.
    """
    seen = set()
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, columns=[id_col, "embedding"])
        except Exception as e:
            click.echo(f"[WARN] Failed to read {file}: {e}", err=True)
            continue
        stats.total_rows += len(df)
        # Drop rows where embedding is null quickly
        if "embedding" not in df.columns:
            continue
        df = df[df["embedding"].notnull()]
        for rec in df.itertuples(index=False):
            _id = getattr(rec, id_col)
            emb = getattr(rec, "embedding")
            norm = normalise_embedding(emb, dim)
            if norm is None:
                if emb is None:
                    stats.null_embeddings += 1
                else:
                    stats.bad_dim += 1
                continue
            if dedupe and _id in seen:
                stats.duplicates += 1
                continue
            if dedupe:
                seen.add(_id)
            stats.kept += 1
            yield _id, norm


def build_matrix_two_pass(
        parquet_files: Sequence[str], dim: int, id_col: str, dedupe: bool, stats: Stats, memmap_dir: str
) -> Tuple[np.ndarray, List[str]]:
    # First pass: count valid vectors (with file-level progress)
    click.echo("First pass (counting) ...")
    count = 0
    seen_first: set = set()
    for file in tqdm(parquet_files, desc="Pass 1/2 files", unit="file"):
        try:
            df = pd.read_parquet(file, columns=[id_col, "embedding"])
        except Exception as e:
            click.echo(f"[WARN] Failed to read {file}: {e}", err=True)
            continue
        stats.total_rows += len(df)
        if "embedding" not in df.columns:
            continue
        # Filter null embeddings quickly
        df = df[df["embedding"].notnull()]
        for rec in df.itertuples(index=False):
            _id = getattr(rec, id_col)
            emb = getattr(rec, "embedding")
            norm = normalise_embedding(emb, dim)
            if norm is None:
                if emb is None:
                    stats.null_embeddings += 1
                else:
                    stats.bad_dim += 1
                continue
            if dedupe:
                if _id in seen_first:
                    continue
                seen_first.add(_id)
            count += 1
    click.echo(f"Valid vectors (after filtering & dedupe): {count}")

    # Second pass: write vectors (file progress + vector progress)
    os.makedirs(memmap_dir, exist_ok=True)
    mm_path = os.path.join(memmap_dir, f"embeddings_{uuid.uuid4().hex}.dat")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(count, dim))
    ids: List[str] = []
    stats.duplicates = 0  # recompute for second pass
    stats.kept = 0
    seen_second: set = set()
    filled = 0
    from_tqdm = tqdm(
        total=count,
        desc="Pass 2/2 vectors",
        unit="vec") if count > 0 else None
    for file in tqdm(parquet_files, desc="Pass 2/2 files", unit="file"):
        try:
            df = pd.read_parquet(file, columns=[id_col, "embedding"])
        except Exception:
            continue
        if "embedding" not in df.columns:
            continue
        df = df[df["embedding"].notnull()]
        for rec in df.itertuples(index=False):
            _id = getattr(rec, id_col)
            emb = getattr(rec, "embedding")
            norm = normalise_embedding(emb, dim)
            if norm is None:
                continue
            if dedupe and _id in seen_second:
                stats.duplicates += 1
                continue
            if dedupe:
                seen_second.add(_id)
            mm[filled] = norm
            ids.append(str(_id))
            filled += 1
            stats.kept += 1
            if from_tqdm is not None:
                from_tqdm.update(1)
    if from_tqdm is not None:
        from_tqdm.close()
    if filled != count:
        click.echo(
            f"[WARN] Filled vector count ({filled}) differs from first-pass estimate ({count}).",
            err=True)
        # Trim matrix if we over-allocated (shouldn't happen unless race)
    mat = np.asarray(mm)[:filled]
    del mm  # flush memmap
    return mat, ids


@click.command()
@click.option("--input-glob", required=True,
              help="Glob for parquet shards with embeddings.")
@click.option("--output-npy", "output_path", required=True,
              help="Path to write uncompressed .npy file (if endswith .npz it will be changed to .npy).")
@click.option("--output-index", required=True,
              help="Path to write index mapping (CSV or Parquet).")
@click.option("--dimensions", default=1536, show_default=True,
              help="Expected embedding dimensionality.")
@click.option("--id-column", default="image_id",
              show_default=True, help="Identifier column name.")
@click.option("--no-dedupe", is_flag=True, default=False,
              help="Do not deduplicate ids (keep all).")
@click.option("--memmap-dir", default="/tmp", show_default=True,
              help="Directory for memmap (always two-pass).")
def cli(
    input_glob: str,
    output_path: str,
    output_index: str,
    dimensions: int,
    id_column: str,
    no_dedupe: bool,
    memmap_dir: str,
):
    """Aggregate embedding shards into a single matrix + index mapping.

    OUTPUTS:
            - NPY file containing the full matrix
      - Index file (CSV or Parquet) with columns: row_index, <id_column>
    """
    parquet_files = sorted(glob.glob(input_glob))
    if not parquet_files:
        raise click.ClickException(
            f"No parquet files match glob: {input_glob}")
    click.echo(f"Found {len(parquet_files)} parquet files.")

    stats = Stats()
    dedupe = not no_dedupe

    # Always use two-pass strategy for predictable memory usage
    matrix, ids = build_matrix_two_pass(
        parquet_files, dimensions, id_column, dedupe, stats, memmap_dir)

    click.echo(f"Final matrix shape: {matrix.shape}")
    # Resolve output path & adjust extension if user supplied a legacy .npz
    if output_path.lower().endswith(".npz"):
        new_path = output_path[:-4] + ".npy"
        click.echo(
            f"[WARN] Requested output ends with .npz; switching to uncompressed .npy: {new_path}")
        output_path = new_path
    elif not output_path.lower().endswith('.npy'):
        # Force .npy extension for clarity
        output_path = output_path + '.npy'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # Write uncompressed .npy (fast, memory efficient given matrix already in
    # memory)
    np.save(output_path, matrix)
    click.echo(f"Wrote embeddings matrix (uncompressed .npy): {output_path}")

    # Build index dataframe
    index_df = pd.DataFrame(
        {"row_index": np.arange(len(ids), dtype=int), id_column: ids})
    out_lower = output_index.lower()
    os.makedirs(os.path.dirname(output_index) or '.', exist_ok=True)
    if out_lower.endswith(".parquet"):
        index_df.to_parquet(output_index, index=False)
    elif out_lower.endswith(".csv"):
        index_df.to_csv(output_index, index=False)
    else:
        # default to parquet
        output_index = output_index + ".parquet"
        index_df.to_parquet(output_index, index=False)
    click.echo(f"Wrote index mapping: {output_index}")

    # Print stats summary
    click.echo("--- Stats ---")
    click.echo(json.dumps({
        "total_rows": stats.total_rows,
        "null_embeddings": stats.null_embeddings,
        "bad_dim": stats.bad_dim,
        "duplicates_skipped": stats.duplicates,
        "kept": stats.kept,
        "dimensions": dimensions,
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    cli()
