import json
import textwrap

from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

from src.config import *


def clean_html(raw_html):
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(strip=True, separator=" ")


def translate_text(text):
    if not text:
        return ""
    translator = GoogleTranslator(source="hu", target="en")
    chunks = textwrap.wrap(text, width=4900, replace_whitespace=False)
    return " ".join(translator.translate(chunk) for chunk in chunks)


def clean_data(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for doc in data:
        text = clean_html(doc.get("text", ""))
        title = doc.get("title", "")
        lang = doc.get("language", "en").lower()

        if lang == "hu":
            text = translate_text(text)
            title = translate_text(title)
            lang = "en"

        cleaned.append({
            "document_id": doc.get("document_id"),
            "source": doc.get("source"),
            "language": lang,
            "title": title,
            "text": text,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    docs = [
        "pubmed_1000_en.json",
        "webmd_formatted_articles.json",
        "cleaned_data_en_hu.json",
    ]
    for doc in docs:
        clean_data(RAW_DATA_DIR / doc, PROCESSED_DATA_DIR / doc)