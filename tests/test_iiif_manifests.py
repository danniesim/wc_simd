import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession

import wc_simd.iiif_manifests as iiif_manifests


@pytest.fixture
def temp_download_dir():
    """Create a temporary directory for downloads and clean it up after tests."""
    download_dir = os.path.join(os.path.dirname(__file__), "test_downloads")
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir)
    yield download_dir
    # Clean up
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)


@pytest.fixture
def mock_spark_session():
    """Create a mock Spark session with simulated data."""
    spark = SparkSession.builder \
        .appName("test_download_iiif_manifests") \
        .master("local[1]") \
        .getOrCreate()

    # Create test data with proper nested structure for the 'works' table using DataFrame creation
    # directly rather than Row objects
    data = [
        {
            "id": "test_id_1",
            "items": [
                {
                    "id": "item_1",
                    "locations": [
                        {"url": "https://example.com/presentation/manifest1"}
                    ]
                }
            ]
        },
        {
            "id": "test_id_2",
            "items": [
                {
                    "id": "item_2",
                    "locations": [
                        {"url": "https://example.com/presentation/manifest2"}
                    ]
                }
            ]
        },
        {
            "id": "test_id_3",
            "items": [
                {
                    "id": "item_3",
                    "locations": [
                        {"url": "https://othersite.com/not_a_manifest"}
                    ]
                }
            ]
        }
    ]

    # Create a Spark DataFrame from the Python dictionaries
    import json
    from pyspark.sql import functions as F

    # Convert the data to JSON strings, then parse them with Spark
    # to correctly preserve the nested structure
    json_strings = [json.dumps(item) for item in data]
    df = spark.read.json(spark.sparkContext.parallelize(json_strings))

    # Register the DataFrame as a table
    df.createOrReplaceTempView("works")

    yield spark

    # Clean up
    spark.catalog.dropTempView("works")
    spark.stop()


def test_download_iiif_manifests_creates_files(
        mock_spark_session, temp_download_dir):
    """Test that files are created with correct names when downloading manifests."""
    with patch('wc_simd.download_iiif_manifests.SparkSession') as mock_spark:
        # Configure the mock to use our fixture
        mock_builder = MagicMock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark_session
        mock_spark.builder = mock_builder

        # Mock requests.get to return a successful response
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = '{"test": "manifest content"}'
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Call the function with a small subset fraction
            iiif_manifests.download_iiif_manifests(
                subset_fraction=1.0, download_dir=temp_download_dir)

            # Check that requests.get was called with the expected URLs
            assert mock_get.call_count == 2
            mock_get.assert_any_call(
                'https://example.com/presentation/manifest1', timeout=10)
            mock_get.assert_any_call(
                'https://example.com/presentation/manifest2', timeout=10)

            # Verify that files were created with expected names
            expected_files = [
                os.path.join(temp_download_dir, "test_id_1_manifest1.json"),
                os.path.join(temp_download_dir, "test_id_2_manifest2.json")
            ]
            for file_path in expected_files:
                assert os.path.exists(
                    file_path), f"Expected file {file_path} to exist"
                with open(file_path, 'r') as f:
                    assert f.read() == '{"test": "manifest content"}'


def test_download_iiif_manifests_error_handling(
        mock_spark_session, temp_download_dir):
    """Test that the function handles errors gracefully when downloads fail."""
    with patch('wc_simd.download_iiif_manifests.SparkSession') as mock_spark:
        mock_builder = MagicMock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark_session
        mock_spark.builder = mock_builder

        with patch('requests.get') as mock_get:
            # Set up the first call to succeed
            first_response = MagicMock()
            first_response.text = '{"test": "manifest content"}'
            first_response.raise_for_status = MagicMock()

            # Set up the second call to fail with an exception
            def side_effect(url, timeout):
                if url == 'https://example.com/presentation/manifest1':
                    return first_response
                else:
                    raise Exception("Download failed")

            mock_get.side_effect = side_effect

            # Call the function - it should not raise an exception
            iiif_manifests.download_iiif_manifests(
                subset_fraction=1.0, download_dir=temp_download_dir)

            # Verify that one file was still created despite the error with the
            # second
            assert os.path.exists(
                os.path.join(
                    temp_download_dir,
                    "test_id_1_manifest1.json"))
            assert os.path.exists(
                os.path.join(
                    temp_download_dir,
                    "test_id_2_manifest2.json"))
            # Assert that the second file is an JSON file with an empty object
            assert os.path.getsize(
                os.path.join(
                    temp_download_dir,
                    "test_id_2_manifest2.json")) == 2


def test_create_manifest_dataframe():
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

    df = iiif_manifests.create_manifest_dataframe(
        spark, manifests_dir="data/iiif_manifests", limit=10)

    df.write.saveAsTable("test_iiif_manifests", mode="overwrite")
    spark.sql("DROP TABLE IF EXISTS test_iiif_manifests")
    spark.stop()
