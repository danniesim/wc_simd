# Data Index

- [Online Catalog Elasticsearch Index Mappings](https://github.com/wellcomecollection/catalogue-pipeline/blob/main/index_config/mappings.works_indexed.2024-11-14.json)
- `work_schema_*`: direct printout from Spark Schema of `works.json`
- `works.dbml`: ChatGPT created DBML from the Spark Schema of `works.json`

## S3 Bucket

Large files are placed here instead of the git repository.

ARN: `arn:aws:s3:::wellcomecollection-dsim`
Path: `s3://wellcomecollection-dsim/wc_simd`

Files:

- `data/iiif_manifests.tar.gz` Downloaded IIIF manifest files for works.
