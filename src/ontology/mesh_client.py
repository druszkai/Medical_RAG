import requests
from functools import lru_cache

class MeSHClient:
    BASE = "https://id.nlm.nih.gov/mesh"

    @lru_cache(maxsize=512)
    def find_descriptor(self, term: str) -> dict | None:
        """Terminus to descriptor"""
        r = requests.get(f"{self.BASE}/lookup/descriptor",
                         params={"label": term, "match": "contains",
                                 "limit": 1, "lang": "eng"})
        results = r.json()
        return results[0] if results else None

    @lru_cache(maxsize=512)
    def get_entry_terms(self, ui: str) -> list[str]:
        """Szinonimák (Entry Terms)"""
        r = requests.get(f"{self.BASE}/{ui}.json")
        data = r.json()
        terms = []
        for concept in data.get("concepts", []):
            for term in concept.get("terms", []):
                terms.append(term["label"])
        return list(set(terms))

    @lru_cache(maxsize=512)
    def get_parents(self, ui: str) -> list[dict]:
        """Szülő node-ok"""
        r = requests.get(f"{self.BASE}/{ui}/parents.json")
        return r.json() if r.status_code == 200 else []