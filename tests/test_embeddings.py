from wc_simd.embed import Qwen3Embedding


def test_qwen3_embeddings():
    embeddings = Qwen3Embedding(
        endpoint="http://ec2-18-134-162-140.eu-west-2.compute.amazonaws.com:8080/embed")
    texts = [
        "This is a test sentence.",
        "Another example of text to embed.",
        "Yet another piece of text for embedding."
    ]
    embedding_results = embeddings.get_embeddings(
        texts, is_query=False)
    print(embedding_results)
    assert len(embedding_results) == len(texts)
