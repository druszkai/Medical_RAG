import requests
from functools import lru_cache

class RxNormClient:
    BASE = "https://rxnav.nlm.nih.gov/REST"

    @lru_cache(maxsize=512)
    def find_rxcui(self, term: str) -> str | None:
        """Gyógyszer- és kiegészítő név keresése"""
        r = requests.get(f"{self.BASE}/rxcui.json",
                         params={"name": term, "search": 2})
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    @lru_cache(maxsize=512)
    def get_synonyms(self, rxcui: str) -> list[str]:
        """RxCUI szinoníma nevek"""
        r = requests.get(f"{self.BASE}/rxcui/{rxcui}/allrelated.json",)
        groups = r.json().get("allRelatedGroup", {}).get("conceptGroup", [])
        names = []
        for group in groups:
            for prop in group.get("conceptProperties", []):
                names.append(prop.get("name"))
        return list(set(names))

