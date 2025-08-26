import pyspark.sql.functions as F
import numpy as np
import pandas as pd
from typing import Iterator
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
from typing import List, Optional, Union
import time
import requests
import click
import logging
from datetime import datetime
from pyspark.sql import SparkSession

# See scripts/run_qwen3embed_multi.sh for running the endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Qwen3Embedding():
    def __init__(
            self, endpoint: str = "http://ec2-3-231-68-18.compute-1.amazonaws.com:8080/embed"):
        """
        Initialize the Qwen3Embedding class with the model endpoint.

        Args:
            endpoint (str): The URL of the embedding service endpoint.
        """
        self.endpoint = endpoint

    def get_detailed_instruct(
            self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = "Given a web search query, retrieve relevant passages that answer the query"
        return f'Instruct: {task_description}\nQuery: {query}'

    def get_embeddings(
            self, sentences: Union[List[str],
                                   str],
            is_query: bool = False, instruction=None, max_retries: int = 9999,
            max_backoff_secs: int = 60) -> Optional[List[List[float]]]:
        """
        Get embeddings from the embedding service endpoint with retry logic.

        Args:
            sentences (Union[List[str], str]): The input sentences or a single sentence to embed.
            is_query (bool): Whether the input is a query. If True, uses a different instruction.
            instruction (str): Custom instruction for the embedding service.
            max_retries (int): Maximum number of retries for failed requests.

        Returns:
            List[List[float]]: List of embedding vectors, or None if all retries failed
        """

        if (instruction is None) and (not is_query):
            sentences = [sentences]
        if is_query:
            sentences = [
                self.get_detailed_instruct(instruction, sent)
                for sent in sentences]
        elif instruction:
            sentences = [
                f'{instruction}{sent}'
                for sent in sentences]

        # Prepare the request payload
        payload = {
            "inputs": sentences
        }

        # Set headers
        headers = {
            "Content-Type": "application/json"
        }

        # Retryable status codes and exceptions
        # Server errors and rate limiting
        retryable_status_codes = {500, 502, 503, 504, 429}

        for attempt in range(
                max_retries + 1):  # +1 because we want max_retries actual retries
            try:
                # Make the POST request
                response = requests.post(
                    self.endpoint, json=payload, headers=headers, timeout=30)

                # If successful, extract embeddings from response
                if response.status_code == 200:
                    response_data = response.json()
                    try:
                        # Extract embedding vectors from response
                        # Assuming the API returns a structure with embeddings
                        if isinstance(response_data,
                                      dict) and 'embeddings' in response_data:
                            return response_data['embeddings']
                        elif isinstance(response_data, list):
                            return response_data  # Some APIs return embeddings as a direct list
                        else:
                            logger.error(
                                f"Unexpected response format: {response_data}")
                            return None
                    except Exception as e:
                        logger.error(
                            f"Failed to parse embedding response: {e}")
                        return None

                # Check if error is retryable
                if response.status_code in retryable_status_codes:
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = min(2 ** attempt, max_backoff_secs)
                        logger.warning(
                            f"Attempt {attempt + 1} failed with status {response.status_code}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"All {max_retries} retries failed. Last status: {response.status_code}")
                        return None
                else:
                    # Non-retryable error (e.g., 400, 401, 404)
                    logger.error(
                        f"Non-retryable error: {response.status_code} - {response.text}")
                    return None

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries:
                    wait_time = min(
                        2 ** attempt, max_backoff_secs)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"All {max_retries} retries failed. Last error: {e}")
                    return None

        return None


@click.command()
@click.option(
    '--input-table',
    default='plain_text_chunks',
    help='Input table name containing text chunks to embed'
)
@click.option(
    '--output-table-prefix',
    default='plain_text_chunks_with_embeddings',
    help='Prefix for output table names (will be suffixed with partition number)'
)
@click.option(
    '--endpoint',
    default='http://ec2-3-231-68-18.compute-1.amazonaws.com:8080/embed',
    help='Embedding service endpoint URL'
)
@click.option(
    '--instruction',
    default=None,
    help='Custom instruction for embedding generation (used with is_query=True)'
)
@click.option(
    '--batch-size',
    default=32,
    type=int,
    help='Batch size for processing embeddings (default: 32)'
)
def main(input_table: str, output_table_prefix: str,
         endpoint: str, instruction: str, batch_size: int):
    """Generate embeddings for text chunks using Qwen3Embedding service."""
    start_time = time.time()
    logger.info(f"Starting embedding generation for table: {input_table}")
    logger.info(f"Using batch size: {batch_size}")

    spark = (
        SparkSession.builder
        .appName("text_embedding")
        .master("local[100]")
        .config("spark.driver.memory", "100g")
        .config("spark.executor.memory", "100g")
        .config("spark.sql.orc.enableVectorizedReader", "false")
        .config("spark.sql.parquet.columnarReaderBatchSize", "256")
        .config("spark.sql.orc.columnarReaderBatchSize", "256")
        .config("spark.sql.shuffle.partitions", "1024")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # Define the embedding function that will be applied to each partition
    def embed_text_partition(
            iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """
        Function to generate embeddings for text chunks using Qwen3Embedding.
        This function will be applied to each partition of the DataFrame.

        Args:
            iterator: Iterator of pandas DataFrames (one per partition)

        Returns:
            Iterator of pandas DataFrames with embeddings added
        """
        # Initialize the embedding model
        embeddings_model = Qwen3Embedding(endpoint=endpoint)

        # Process each DataFrame in the partition
        for df in iterator:
            # Process in batches using the specified batch_size
            result_dfs = []

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                texts = batch['chunk_text'].tolist()

                # Get embeddings for the batch
                embedding_vectors = embeddings_model.get_embeddings(
                    texts, is_query=False, instruction=instruction)

                # Add embeddings to the batch if embeddings were returned
                batch = batch.copy()
                batch['embedding'] = embedding_vectors
                result_dfs.append(batch)

            yield pd.concat(result_dfs)

    # Define the output schema including embeddings
    embedding_schema = StructType([
        StructField("id", StringType(), True),
        StructField("chunk_text", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("total_chunks", IntegerType(), True),
        StructField("embedding", ArrayType(FloatType()), True)
    ])

    chunks_df = (
        spark.table(input_table)
        # .sample(withReplacement=False, fraction=0.0001, seed=42)
    )

    # Get the number of partitions
    original_partitions = chunks_df.rdd.getNumPartitions()
    logger.info(f"Original DataFrame has {original_partitions} partitions")

    # Create a list to store DataFrames for each partition
    partition_dataframes = []

    # Track timing for progress estimation
    partition_times = []
    processed_count = 0
    skipped_count = 0

    # Create a DataFrame for each partition and repartition to 64
    for i, partition_id in enumerate(range(original_partitions)):
        partition_start_time = time.time()

        # Check if table already exists
        table_name = f"{output_table_prefix}_{i}"
        if spark.catalog.tableExists(table_name):
            logger.info(f"Table {table_name} already exists, skipping...")
            skipped_count += 1
            continue

        # Filter data for this specific partition
        partition_df = chunks_df.filter(F.spark_partition_id() == partition_id)

        # Repartition to 64 partitions
        repartitioned_df = partition_df.repartition(64)

        logger.info(
            f"Created DataFrame for partition {partition_id} with {repartitioned_df.rdd.getNumPartitions()} partitions")

        logger.info(f"Processing partition DataFrame {i}...")
        embedded_df = repartitioned_df.mapInPandas(
            embed_text_partition,
            schema=embedding_schema
        )
        embedded_df.write.mode("overwrite").saveAsTable(table_name)

        # Calculate timing and progress
        partition_end_time = time.time()
        partition_duration = partition_end_time - partition_start_time
        partition_times.append(partition_duration)
        processed_count += 1

        # Calculate progress statistics
        total_remaining = original_partitions - processed_count - skipped_count
        if partition_times:
            avg_time_per_partition = sum(
                partition_times) / len(partition_times)
            estimated_time_left = avg_time_per_partition * total_remaining

            logger.info(
                f"Partition {i} completed in {partition_duration:.2f} seconds")
            logger.info(
                f"Progress: {processed_count + skipped_count}/{original_partitions} partitions "
                f"({processed_count} processed, {skipped_count} skipped)")
            logger.info(
                f"Average time per partition: {avg_time_per_partition:.2f} seconds")
            logger.info(
                f"Estimated time remaining: {estimated_time_left:.2f} seconds ({estimated_time_left/60:.1f} minutes)")

    # Log total time taken
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"=== PROCESSING COMPLETE ===")
    logger.info(f"Total partitions processed: {processed_count}")
    logger.info(f"Total partitions skipped: {skipped_count}")
    logger.info(
        f"Total time taken: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    if partition_times:
        logger.info(
            f"Average time per processed partition: {sum(partition_times)/len(partition_times):.2f} seconds")
        logger.info(f"Fastest partition: {min(partition_times):.2f} seconds")
        logger.info(f"Slowest partition: {max(partition_times):.2f} seconds")


if __name__ == "__main__":
    main()
