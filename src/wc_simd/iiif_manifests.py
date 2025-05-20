from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, StructType, StructField
import requests
import os
from tqdm import tqdm
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from wc_simd.utility import spark_path
from typing import Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def download_iiif_manifests(
        subset_fraction: float = 0.01, download_dir: str = "."):
    """
    This script is used to extract IIIF manifest URLs from the works table in the database.
    It uses PySpark to process the data and saves the manifests to a specified directory.
    """

    # Create a Spark session
    spark = SparkSession.builder \
        .appName("test_pyspark") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.sql.orc.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024") \
        .config("spark.sql.orc.columnarReaderBatchSize", "1024") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    df_works_item_urls = (
        spark.table("works")
        # 1) explode each item so we have work‐level id + item struct
        .select(
            F.col("id"),
            F.explode(F.col("items")).alias("item")
        )
        # 2) from each item, explode only the locations whose URL matches
        .select(
            F.col("id"),
            F.col("item.id").alias("item_id"),
            F.explode(
                F.expr("filter(item.locations, l -> l.url LIKE '%/presentation/%')")
            ).alias("location")
        )
        # 3) pick out just the fields you want
        .select(
            F.col("id"),
            F.col("item_id"),
            F.col("location.url").alias("url")
        )
    )

    df_works_item_urls_subset = df_works_item_urls.sample(
        withReplacement=False,
        fraction=subset_fraction,
        seed=42
    )

    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Replace conversion to Pandas with Spark local iterator and tqdm
    total = df_works_item_urls_subset.count()
    row_iter = df_works_item_urls_subset.toLocalIterator()
    for row in tqdm(row_iter, total=total, desc="Downloading manifests"):
        url = row['url']
        last_part = url.rstrip('/').rsplit('/', 1)[-1]
        filename = f"{row['id']}_{last_part}.json"
        full_path = os.path.join(download_dir, filename)

        # Skip if file already exists
        if os.path.exists(full_path):
            continue

        # Try downloading up to 3 times, but skip retries if 404 is encountered
        success = False
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 404:
                    print(f"404 Not Found: {url}")
                    break  # Exit retry loop on 404
                resp.raise_for_status()
                with open(full_path, "w") as f:
                    f.write(resp.text)
                success = True
                break
            except Exception as e:
                if attempt < 2:  # Don't sleep after the last attempt
                    import time
                    time.sleep(1)  # Wait a bit before retrying

        # If all attempts failed, create an empty JSON file
        if not success:
            with open(full_path, "w") as f:
                f.write("{}")


def create_manifest_dataframe(spark, manifests_dir: str, limit: int = None):
    """
    Create a Spark DataFrame from all IIIF manifest JSON files in the given directory.
    Each file is expected to be a single JSON object.
    Args:
        spark: SparkSession object.
        manifests_dir: Directory containing IIIF manifest JSON files.
        limit: Optional integer to limit the number of files loaded (for testing).
    Returns:
        DataFrame containing the parsed manifest data, with an added 'filename' column.
    """
    import glob
    import os
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(manifests_dir, '*.json'))

    # use the spark_path function to convert to a file URL
    json_files = [spark_path(f) for f in json_files]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {manifests_dir}")
    if limit is not None:
        json_files = json_files[:limit]
    # Create a DataFrame from the JSON files
    df = spark.read.option("multiLine", True).json(json_files)
    # Add filename column for traceability
    from pyspark.sql.functions import input_file_name
    df = df.withColumn('filename', input_file_name())

    # extract id and manifest id from filename with a UDF

    def extract_ids(filename: str) -> Tuple[str, str]:
        # Extract the last part of the filename (before .json)
        base_name = os.path.basename(filename)
        id_part, manifest_id = base_name.split('_', 1)
        manifest_id = manifest_id.rsplit('.', 1)[0]  # Remove .json
        return id_part, manifest_id

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("manifest_id", StringType(), True)
    ])

    extract_ids_udf = udf(extract_ids, schema)

    df = spark.table("iiif_manifests")

    df = df.withColumn("ids", extract_ids_udf(F.col("filename")))

    df2 = (
        df
        .withColumn("id", F.col("ids.id"))
        .withColumn("manifest_id", F.col("ids.manifest_id"))
        # 2) drop the original struct
        .drop("ids")
    )
    return df2


if __name__ == "__main__":
    import argparse
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download IIIF manifests or create manifest DataFrame.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    parser_download = subparsers.add_parser(
        "download", help="Download IIIF manifests.")
    parser_download.add_argument(
        "--subset_fraction", type=float, default=0.01,
        help="Fraction of rows to sample from the works table.")
    parser_download.add_argument("--download_dir", type=str, default=".",
                                 help="Directory to download manifests.")

    # Create command
    parser_create = subparsers.add_parser(
        "create", help="Create manifest DataFrame from downloaded manifests.")
    parser_create.add_argument(
        "--manifests_dir", type=str, required=True,
        help="Directory containing IIIF manifest JSON files.")
    parser_create.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of files loaded (for testing).")

    args = parser.parse_args()

    if args.command == "download":
        download_iiif_manifests(
            subset_fraction=args.subset_fraction,
            download_dir=args.download_dir)
    elif args.command == "create":
        # Print time taken to create the DataFrame
        import time
        start_time = time.time()
        logging.info(
            f"Creating manifest DataFrame from {
                args.manifests_dir}...")
        # Create a Spark session
        spark = SparkSession.builder \
            .appName("test_pyspark") \
            .config("spark.driver.memory", "100g") \
            .config("spark.executor.memory", "100g") \
            .config("spark.sql.orc.enableVectorizedReader", "false") \
            .config("spark.sql.parquet.columnarReaderBatchSize", "256") \
            .config("spark.sql.orc.columnarReaderBatchSize", "256") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        df = create_manifest_dataframe(
            spark, args.manifests_dir, limit=args.limit)
        logging.info("Schema:")
        df.printSchema()
        df.write.saveAsTable(
            "iiif_manifests",
            mode="overwrite"
        )
        logging.info("DataFrame created and saved as table 'iiif_manifests'.")
        logging.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        logging.info(f"Count: {spark.table('iiif_manifests').count()}")
        spark.stop()
