import pyspark.sql.functions as F
import numpy as np
import pandas as pd
from typing import Iterator
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
from typing import List, Optional, Union
import time
import requests
from pyspark.sql import SparkSession


class Qwen3Embedding():
    def __init__(
            self, endpoint: str = "http://ec2-18-134-162-140.eu-west-2.compute.amazonaws.com:8080/embed"):
        """
        Initialize the Qwen3Embedding class with the model endpoint.

        Args:
            endpoint (str): The URL of the embedding service endpoint.
        """
        self.endpoint = endpoint

    def get_detailed_instruct(
            self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
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

        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [
                self.get_detailed_instruct(instruction, sent)
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
                            print(
                                f"Unexpected response format: {response_data}")
                            return None
                    except Exception as e:
                        print(f"Failed to parse embedding response: {e}")
                        return None

                # Check if error is retryable
                if response.status_code in retryable_status_codes:
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = min(2 ** attempt, max_backoff_secs)
                        print(
                            f"Attempt {attempt + 1} failed with status {response.status_code}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(
                            f"All {max_retries} retries failed. Last status: {response.status_code}")
                        return None
                else:
                    # Non-retryable error (e.g., 400, 401, 404)
                    print(
                        f"Non-retryable error: {response.status_code} - {response.text}")
                    return None

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries:
                    wait_time = min(
                        2 ** attempt, max_backoff_secs)  # Exponential backoff
                    print(
                        f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} retries failed. Last error: {e}")
                    return None

        return None


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("test_pyspark") \
        .master("local[32]") \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "100g") \
        .config("spark.sql.orc.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "256") \
        .config("spark.sql.orc.columnarReaderBatchSize", "256") \
        .config("spark.sql.shuffle.partitions", "1024") \
        .getOrCreate()

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
        embeddings_model = Qwen3Embedding(
            endpoint="http://ec2-18-134-162-140.eu-west-2.compute.amazonaws.com:8080/embed")

        # Process each DataFrame in the partition
        for df in iterator:
            # Process in batches of 32 for efficiency
            batch_size = 32
            result_dfs = []

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                texts = batch['chunk_text'].tolist()

                # Get embeddings for the batch
                embedding_vectors = embeddings_model.get_embeddings(
                    texts, is_query=False)

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
        spark.table("plain_text_chunks")
        # .sample(withReplacement=False, fraction=0.0001, seed=42)
    )

    # Apply the embedding function to the sample
    sample_embedded_df = chunks_df.mapInPandas(
        embed_text_partition,
        schema=embedding_schema
    )

    # sample_embedded_df.show(truncate=False)

    sample_embedded_df.write.mode("overwrite").saveAsTable(
        "plain_text_chunks_with_embeddings")
