import json

from src.config import *

INPUT_FILE = "all_merged_articles.json"


def deduplicate(docs):
    seen = set()
    unique = []
    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc)
    return unique


def main():
    input_path = PROCESSED_DATA_DIR / INPUT_FILE
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Documents before deduplication: {len(docs)}")
    unique_docs = deduplicate(docs)
    print(f"Documents after deduplication: {len(unique_docs)}")
    print(f"Removed: {len(docs) - len(unique_docs)} duplicates")

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(unique_docs, f, ensure_ascii=False, indent=4)

    print(f"Saved to {input_path}")


if __name__ == "__main__":
    main()