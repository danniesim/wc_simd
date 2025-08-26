import os
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import dotenv
from wc_simd.dedupe_service import dedup_data_file
from wc_simd.llm import ChatSession
import tqdm


index_path = "data/name_rec_faiss.index"
embed_prompt = "Instruct: Given a search query that contains a person's name, retrieve relevant passages that mention the same person.\nQuery: "


def get_reconciled(model, index, to_index, label, k):
    """
    Search for the k most similar embeddings to a given query text
    """
    # Find the query embedding by matching the text
    query_idx = to_index[to_index['label'] == label].index

    if len(query_idx) == 0:
        print(f"Label'{label}' not found in dataset")
        return None

    query_embedding = model.encode(
        [label],
        prompt=embed_prompt,
        show_progress_bar=False)

    # Search for k most similar embeddings
    similarities, indices = index.search(query_embedding, k)

    MIN_MAX_SIMILARITY = 0.8

    max_similarity = max(similarities[0][0] if len(
        similarities) > 0 else MIN_MAX_SIMILARITY, MIN_MAX_SIMILARITY)
    cutoff_similarity = max_similarity * 0.7  # Set a cutoff similarity threshold

    records_for_context = []
    records_for_candidates = []
    for i, (sim, vec_idx) in enumerate(zip(similarities[0], indices[0])):
        if sim >= cutoff_similarity:
            records_for_context.append(dict(
                label=str(to_index.iloc[vec_idx]['label']),
                idx=int(to_index.iloc[vec_idx]['idx'])
            ))
        records_for_candidates.append(dict(
            label=str(to_index.iloc[vec_idx]['label']),
            idx=int(to_index.iloc[vec_idx]['idx']),
            similarity=float(sim)
        ))

    # To csv in string buffer
    csv_string = pd.DataFrame(records_for_context).to_csv(
        index=False, header=False, sep=",", encoding="utf-8")
    # print(csv_string)

    chat = ChatSession(
        system_instruction="""# INSTRUCTIONS
You will be given the name of a person and a CSV string with the following format:

name,idx

## Goal
Identify all mentions of the person in the context and return their indices.

## Rules
    1. If the person is mentioned in the context, return the idx.
    2. If the person is not mentioned, return an empty list.
    3. If there are multiple mentions, return all matching idxs.
    4. If the person is mentioned with different names, return all variations as a json list.
    5. If there is not enough information contained in either name, do not return the idx.

## Output Format
Return a JSON list with the following format:

[idx1, idx2, ...]
}
""")

    result = chat.send(f"Person: {label}\nCSV:\n{csv_string}", stdout=False)
    return dict(reconciled=json.loads(
        result[0]), candidates=records_for_candidates)


def name_rec_run():
    dotenv.load_dotenv(dotenv_path=".env.name_rec")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    index = faiss.read_index(index_path)

    dedup_data = pd.read_csv(dedup_data_file, index_col=0)

    # force all columns to string type
    dedup_data = dedup_data.astype(str)
    to_index = dedup_data[dedup_data['type'].isin(
        ["Person", "Agent"])].reset_index(drop=False)

    # Save sample to csv
    to_index_sample = to_index.sample(n=400, random_state=420)
    to_index_sample.to_csv("data/name_rec_sample.csv", index=False)

    for _, row in tqdm.tqdm(
            to_index_sample.iterrows(),
            total=to_index_sample.shape[0]):
        label = row['label']
        idx = row['idx']
        out_file = f"data/name_rec/name_rec_{idx}.json"
        # Skip if file exists
        if os.path.exists(out_file):
            continue
        similar_embeddings = get_reconciled(
            model, index, to_index, label, k=100)
        reconciled_labels = [
            dict(
                label=to_index[to_index['idx'] == x]
                ['label'].values[0],
                idx=x) for x in similar_embeddings["reconciled"]]
        out_dict = dict(
            label=str(label),
            idx=int(idx),
            reconciled_labels=reconciled_labels,
            candidates=similar_embeddings["candidates"]
        )

        # Make sure dir exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(out_dict, f)


if __name__ == "__main__":
    name_rec_run()
