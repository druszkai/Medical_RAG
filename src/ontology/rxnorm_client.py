import requests
from functools import lru_cache

class RxNormClient:
    BASE = "https://rxnav.nlm.nih.gov/REST"

    @lru_cache(maxsize=512)
    def find_rxcui(self, term: str) -> str | None:
        r = requests.get(f"{self.BASE}/rxcui.json",
                         params={"name": term, "search": 2})
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    @lru_cache(maxsize=512)
    def get_synonyms(self, rxcui: str) -> list[str]:
        r = requests.get(f"{self.BASE}/rxcui/{rxcui}/allrelated.json")
        groups = r.json().get("allRelatedGroup", {}).get("conceptGroup", [])
        names = []
        for group in groups:
            for prop in group.get("conceptProperties", []):
                if name := prop.get("name"):
                    names.append(name)
        return list(set(names))

    @lru_cache(maxsize=512)
    def get_related_drugs(self, rxcui: str) -> list[str]:
        r = requests.get(f"{self.BASE}/rxcui/{rxcui}/related.json",
                         params={"tty": "IN BN"})
        try:
            groups = r.json().get("relatedGroup", {}).get("conceptGroup", [])
            return [p["name"] for g in groups for p in g.get("conceptProperties", [])]
        except Exception:
            return []


if __name__ == "__main__":
    client = RxNormClient()

    for term in ["ginkgo biloba", "lisinopril", "St. John's Wort"]:
        print(f"\n=== {term} ===")
        rxcui = client.find_rxcui(term)
        if rxcui:
            print(f"RxCUI: {rxcui}")
            synonyms = client.get_synonyms(rxcui)
            print(f"Synonyms ({len(synonyms)}): {synonyms[:5]}")
            related = client.get_related_drugs(rxcui)
            print(f"Related ({len(related)}): {related[:5]}")
        else:
            print("Not found.")