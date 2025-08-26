import pandas as pd
from wc_simd.dedupe_service import dedup_data_file
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

    dedup_data = pd.read_csv(dedup_data_file, index_col=0)

    # force all columns to string type
    dedup_data = dedup_data.astype(str)

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    labels = dedup_data["label"]

    # embed_prompt = "Instruct: Given a search query with a person's name, retrieve relevant passages that has the person mentioned.\nQuery: "
    # embeddings = model.encode(labels, prompt=embed_prompt)

    embeddings = model.encode(labels)

    # Save embeddings on disk
    import numpy as np
    np.save("data/name_rec_embeddings_no_prompt.npy", embeddings)
