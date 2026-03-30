import json
from collections import defaultdict
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.config import *

INPUT_FILE = "all_merged_articles.json"
OUTPUT_FILE = "outliers.json"
SIMILARITY_THRESHOLD = 0.40
NEIGHBORS_TO_CHECK = 40

def load_documents(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_embeddings(documents: List[Dict], model: SentenceTransformer):
    texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in documents]
    return model.encode(texts, show_progress_bar=True, batch_size=64)

def detect_outliers(documents: List[Dict], embeddings) -> List[Dict]:
    k_value = min(NEIGHBORS_TO_CHECK + 1, len(embeddings))
    knn_model = NearestNeighbors(n_neighbors=k_value, metric="cosine")
    knn_model.fit(embeddings)

    distances, _ = knn_model.kneighbors(embeddings)
    mean_similarities = 1 - distances[:, 1:].mean(axis=1)

    outliers = []
    for index, score in enumerate(mean_similarities):
        if score < SIMILARITY_THRESHOLD:
            outliers.append({
                "document_id": documents[index].get("document_id"),
                "title": documents[index].get("title", "Unknown"),
                "source": documents[index].get("source", "Unknown"),
                "mean_similarity": round(float(score), 4),
            })

    return sorted(outliers, key=lambda x: x["mean_similarity"])

def print_statistics(outliers: List[Dict], all_documents: List[Dict]):
    total_by_source = defaultdict(int)
    for doc in all_documents:
        total_by_source[doc.get("source", "Unknown")] += 1

    outliers_by_source = defaultdict(list)
    for outlier in outliers:
        outliers_by_source[outlier["source"]].append(outlier)

    for source, docs in sorted(outliers_by_source.items(), key=lambda x: -len(x[1])):
        print(f"{source}: {len(docs)} / {total_by_source[source]}")

def main():
    raw_documents = load_documents(PROCESSED_DATA_DIR / INPUT_FILE)
    valid_documents = [doc for doc in raw_documents if doc.get("text", "").strip()]

    embedding_model = SentenceTransformer(MULTILINGUAL_EMBEDDING)
    embeddings = generate_embeddings(valid_documents, embedding_model)

    outliers = detect_outliers(valid_documents, embeddings)

    output_path = PROCESSED_DATA_DIR / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(outliers, file, ensure_ascii=False, indent=4)

    print(f"Outliers found: {len(outliers)} / {len(valid_documents)}")
    print_statistics(outliers, valid_documents)

if __name__ == "__main__":
    main()