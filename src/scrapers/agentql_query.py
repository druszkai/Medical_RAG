import json
import logging

import agentql
from agentql.ext.playwright.sync_api import Page
from playwright.sync_api import sync_playwright

from src.config import *
from src.scrapers.urls import AHA, DRAXE, EVERYDAYHEALTH, HEALTHLINE, MAYO_CLINIC, MNT, NIH, VERYWELL, WEBMD

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_FILE = RAW_DATA_DIR / "scraped_articles.json"

ALL_URLS = {
    "Healthline": HEALTHLINE,
    "WebMD": WEBMD,
    "MayoClinic": MAYO_CLINIC,
    "AHA": AHA,
    "MedicalNewsToday": MNT,
    "VerywellHealth": VERYWELL,
    "DrAxe": DRAXE,
    "NIH": NIH,
    "EverydayHealth": EVERYDAYHEALTH,
}

QUERY = """
{
  article {
    title
    text
    description
  }
}
"""


def is_valid(response):
    article = response.get("article", {})
    if not article:
        return False
    title = article.get("title", "").strip()
    text = article.get("text", "").strip()
    return bool(title) and len(text) > 200


def scrape_page(page: Page, url: str, source: str):
    try:
        page.goto(url, timeout=30000)
        response = page.query_data(QUERY)
        if is_valid(response):
            article = response["article"]
            return {
                "document_id": url,
                "source": source,
                "language": "en",
                "title": article["title"].strip(),
                "text": article["text"].strip(),
            }
        else:
            log.info(f"Skipped (empty/short): {url}")
            return None
    except Exception as e:
        log.warning(f"Failed: {url} — {e}")
        return None


def main():
    results = []

    with sync_playwright() as p, p.chromium.launch(headless=False) as browser:
        page = agentql.wrap(browser.new_page())

        for source, urls in ALL_URLS.items():
            log.info(f"\nScraping {source} ({len(urls)} URLs)...")
            for url in urls:
                result = scrape_page(page, url, source)
                if result:
                    results.append(result)
                    log.info(f"  Saved: {result['title']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nDone. Saved {len(results)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()