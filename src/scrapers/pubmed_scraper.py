from src.config import *

import json
import time
from Bio import Entrez

Entrez.email = "email_asd@dom_domain.com"


def fetch_pubmed_data_formatted(max_results=1200):
    query = '("Cardiovascular Diseases"[Mesh] OR "Lipids"[Mesh] OR "Hypertension"[Mesh] OR "Myocardial Infarction"[Mesh]) OR ("Phytotherapy"[Mesh] OR "Hypertension"[Mesh])'

    print(f"--- PubMed lekérés indítása (Cél: maximum {max_results} cikk) ---")

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]
    print(f"Talált azonosítók száma: {len(id_list)}")

    articles = []
    batch_size = 100

    print("Adatok letöltése és formázása...")

    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        try:
            fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, retmode="xml")
            records = Entrez.read(fetch_handle)
            fetch_handle.close()

            for pubmed_article in records.get("PubmedArticle", []):
                medline_citation = pubmed_article.get("MedlineCitation", {})
                article = medline_citation.get("Article", {})

                pmid = str(medline_citation.get("PMID", ""))
                title = article.get("ArticleTitle", "Nincs cím")

                # Absztrakt kinyerése és összefűzése
                abstract_list = article.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join([str(item) for item in abstract_list])

                # A kért formátum összeállítása
                if abstract:
                    articles.append({
                        "document_id": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source": "PubMed",
                        "language": "en",
                        "title": title,
                        "categories": [
                            "medical",
                            "cardiovascular",
                            "natural_remedy"
                        ],
                        "text": abstract if abstract else "Nincs absztrakt."
                    })

        except Exception as e:
            print(f"Hiba a kötegelt letöltésnél ({i}-{i + batch_size}): {e}")

        print(f"Feldolgozva: {len(articles)} / {len(id_list)}")
        time.sleep(1.5)  # Biztonsági szünet

    return articles


# Futás és mentés
formatted_results = fetch_pubmed_data_formatted(1200)

output_file = RAW_DATA_DIR / 'pubmed_1000_en.json'
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_results, f, ensure_ascii=False, indent=4)

print(f"\nKész! Az adatok mentve ide: {output_file}")