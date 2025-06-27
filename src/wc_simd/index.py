if __name__ == "__main__":
    import pyspark.sql.functions as F
    import numpy as np
    from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
    from typing import Iterator
    import pandas as pd
    from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
    import dotenv
    import os
    import time
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("pyspark_index") \
        .master("local[2]") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.sql.orc.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "256") \
        .config("spark.sql.orc.columnarReaderBatchSize", "256") \
        .config("spark.sql.shuffle.partitions", "1024") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Load environment variables from .env file at the driver level
    dotenv.load_dotenv()

    # Get environment variables on the driver node
    ES_CLOUD_ID = os.environ.get("ES_CLOUD_ID")
    ES_USERNAME = os.environ.get("ES_USERNAME")
    ES_PASSWORD = os.environ.get("ES_PASSWORD")

    # Define the embedding function that will be applied to each partition

    def index_embeddings_partition(
            iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:

        # Initialize ElasticsearchStore using the pre-loaded environment
        # variables
        db = ElasticsearchStore(
            # replace with your cloud ID
            es_cloud_id=ES_CLOUD_ID,
            index_name="vectorsearch_sharded",
            embedding=None,
            es_user=ES_USERNAME,
            es_password=ES_PASSWORD,
            # replace with your password
            strategy=DenseVectorStrategy(),  # strategy for dense vector search
        )

        # Process each DataFrame in the partition
        for df in iterator:
            # Skip empty DataFrames
            if df.empty:
                yield df
                continue

            # Combine "id" and "chunk_idx" to create a unique document ID
            df['es_doc_id'] = df['id'] + "_" + df['chunk_index'].astype(str)

            # Create list of texts
            texts = df['chunk_text'].tolist()
            # Create lists of embeddings from embedding column
            embeddings = df['embedding'].tolist()

            # Add embeddings with retry logic
            max_retries = 9999
            for attempt in range(max_retries + 1):
                try:
                    db.add_embeddings(
                        text_embeddings=zip(texts, embeddings),
                        metadatas=df[['contributor', 'date']].to_dict(
                            orient='records'),
                        ids=df['es_doc_id'].tolist())
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries:
                        # Exponential backoff, max 1 minutes
                        wait_time = min(2 ** min(attempt, 10), 60)
                        print(
                            f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(
                            f"All {max_retries} retries failed. Last error: {e}")
                        raise e  # Re-raise the exception after all retries

            # Return just the document ID
            yield df[['es_doc_id']]

    # Define the output schema including embeddings
    index_embedding_schema = StructType([
        StructField("es_doc_id", StringType(), True),
    ])

    work_contributor_w_date = spark.table("works").select(
        "id", F.explode("contributors").alias("contributors"),
        F.col("production").getItem(0).getField("dates").getItem(0).
        getField("label").alias("date"),).select(
        "id", "contributors.primary", F.col("contributors.agent.label").alias(
            "contributor"),
        "date").where(
        F.col("primary") == True).drop("primary")

    for df_idx in range(0, 287):
        table_name = f"plain_text_chunks_with_embeddings_{df_idx}"
        if not spark.catalog.tableExists(table_name):
            print(f"Table {table_name} does not exists, skipping...")
            continue

        output_table_name = table_name + "_indexed"
        if spark.catalog.tableExists(output_table_name):
            print(f"Table {output_table_name} already exists, skipping...")
            continue

        print(f"Indexing table {table_name}...")

        df_to_index = spark.table(table_name).join(
            work_contributor_w_date, on="id", how="left")

        indexed_df = (
            df_to_index
            .mapInPandas(
                index_embeddings_partition,
                schema=index_embedding_schema
            )
        )
        indexed_df.write.mode("overwrite").saveAsTable(
            output_table_name
        )
