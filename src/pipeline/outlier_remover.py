from src.config import *

import json

OUTLIERS = "outliers.json"
ALL_ARTICLES = "all_merged_articles.json"

def load_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_outliers():
    outliers = load_docs(PROCESSED_DATA_DIR / OUTLIERS)
    articles = load_docs(PROCESSED_DATA_DIR / ALL_ARTICLES)

    outlier_ids = {outlier["document_id"] for outlier in outliers}
    saved_articles = [article for article in articles if article.get("document_id") not in outlier_ids]
    with open(PROCESSED_DATA_DIR / ALL_ARTICLES, "w", encoding="utf-8") as f:
        json.dump(saved_articles, f, indent=4, ensure_ascii=False)
        print(len(saved_articles))

if __name__ == "__main__":
    remove_outliers()