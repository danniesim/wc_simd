 #!/bin/bash

# Run this from the root of the repo

FILE_PATH="data/works_with_images_no_text_partitioned_embedded.parquet"

# Two modes based on first argument (upload and download)

if [ "$1" == "upload" ]; then
    echo "Uploading $FILE_PATH to S3..."

     # Delete data/works_with_images_no_text_partitioned_embedded.parquet.tar.gz if it exists
    if [ -f "$FILE_PATH.tar.gz" ]; then
        rm "$FILE_PATH.tar.gz"

    # Tar and gzip the file
        tar -czvf "$FILE_PATH.tar.gz" "$FILE_PATH"
    else
        echo "$FILE_PATH does not exist."
        exit 1
    fi

    python aws/upload_to_s3.py "$FILE_PATH.tar.gz"
elif [ "$1" == "download" ]; then
    echo "Downloading $FILE_PATH from S3..."

    python aws/upload_to_s3.py --download "$FILE_PATH.tar.gz"

    # Unzip and untar the file
    if [ -f "$FILE_PATH.tar.gz" ]; then
        tar -xzvf "$FILE_PATH.tar.gz" -C data/
    else
        echo "$FILE_PATH.tar.gz does not exist."
        exit 1
    fi
fi
