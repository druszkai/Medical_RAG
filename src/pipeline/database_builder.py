import json
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from src.config import *

DATA_FILES = [
    "all_merged_articles.json",
]


def build_db(processed_data):
    with open(PROCESSED_DATA_DIR / processed_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    embedding_function = SentenceTransformerEmbeddingFunction(MULTILINGUAL_EMBEDDING)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    medical_collection = chroma_client.get_or_create_collection(
        name="medical_collection",
        embedding_function=embedding_function,
    )

    print(f"Building database from {processed_data}...\n")
    print(f"Processing {len(data)} documents...\n")

    for doc in data:
        doc_id = doc.get("document_id", "unknown_id")
        source = doc.get("source", "unknown_source")
        language = doc.get("language", "unknown_language")
        title = doc.get("title", "No Title")
        text = doc.get("text", "")

        if not text.strip():
            continue

        chunks = chunker.split_text(text)

        batch_ids = []
        batch_documents = []
        batch_metadata = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:6]}"
            batch_ids.append(chunk_id)
            batch_documents.append(f"Title: {title}\n\n{chunk_text}")
            batch_metadata.append({
                "document_id": doc_id,
                "source": source,
                "language": language,
                "title": title,
                "chunk_index": i,
            })

        if batch_ids:
            medical_collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadata,
            )


if __name__ == "__main__":
    for data_file in DATA_FILES:
        build_db(data_file)