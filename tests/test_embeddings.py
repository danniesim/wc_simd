import os
from wc_simd.embed import Qwen3Embedding
import dotenv
from sentence_transformers import SentenceTransformer
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain_core.documents import Document
from uuid import uuid4


def test_qwen3_embeddings():
    embeddings = Qwen3Embedding(
        endpoint="http://ec2-3-231-68-18.compute-1.amazonaws.com:8080/embed")
    texts = [
        "This is a test sentence.",
        "Another example of text to embed.",
        "Yet another piece of text for embedding."
    ]
    embedding_results = embeddings.get_embeddings(
        texts, is_query=False)
    print(embedding_results)
    assert len(embedding_results) == len(texts)


def test_es_vector_search():
    dotenv.load_dotenv()

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    # Load environment variables from .env file

    # query = "What did the president say about sweet things for breakfast?"
    query = "What did the president say about meatfor breakfast?"

    query_embeddings = model.encode(query, prompt_name="query")

    # Initialize ElasticsearchStore
    db = ElasticsearchStore(
        # replace with your cloud ID
        es_cloud_id=os.environ.get("ES_CLOUD_ID"),
        index_name="vectorsearch",
        embedding=None,
        es_user=os.environ.get("ES_USERNAME"),
        es_password=os.environ.get("ES_PASSWORD"),
        # replace with your password
        strategy=DenseVectorStrategy(),  # strategy for dense vector search
    )

    # Prepare documents
    documents = [
        Document(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            metadata={"source": "tweet"}),
        Document(
            page_content="I had bacon for breakfast this morning.",
            metadata={"source": "tweet"}),
        Document(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news"}),
        Document(
            page_content="Building an exciting new project with LangChain - come check it out!",
            metadata={"source": "tweet"}),]

    doc_embeddings = model.encode(
        [doc.page_content for doc in documents])  # , prompt_name="document")

    # Generate unique IDs for documents
    ids = [str(uuid4()) for _ in documents]

    # Add documents to the vector store
    db.add_embeddings(
        text_embeddings=zip([x.page_content for x in documents],
                            doc_embeddings),
        metadatas=[x.metadata for x in documents],
        ids=ids)

    # Alternatively, if you have precomputed embeddings as a list of vectors corresponding to documents:
    # embeddings_list = [...]  # your precomputed vectors
    # texts = [doc.page_content for doc in documents]
    # db.add_texts(texts=texts, embeddings=doc_embeddings)

    # Now you can perform similarity search
    results = db.similarity_search_by_vector_with_relevance_scores(
        query_embeddings,
        k=4)

    for doc in results:
        print(doc)

    response = db.client.get(index="vectorsearch", id=ids[0])
    if response and response.get("found", False):
        source = response["_source"]
        print(source)
    else:
        print("Document not found")

    db.delete(ids=ids)
