# Elasticsearch Notes

## Run a Local Instance

```sh
docker run -p 127.0.0.1:9200:9200 -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.license.self_generated.type=basic" \
  -v "elasticsearch-data:/Users/simd/elasticsearch_data" \
  docker.elastic.co/elasticsearch/elasticsearch:9.0.2
```
