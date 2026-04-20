import requests
from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON

"""
Used sites for future code-revisions:
    https://id.nlm.nih.gov/mesh/swagger/ui#/
    
    https://hhs.github.io/meshrdf/terms
    https://hhs.github.io/meshrdf/concepts
    https://hhs.github.io/meshrdf/descriptors
    
    https://id.nlm.nih.gov/mesh/query
"""

class MeSHClient:
    BASE_REST = "https://id.nlm.nih.gov/mesh"
    SPARQL_ENDPOINT = "https://id.nlm.nih.gov/mesh/sparql"

    def __init__(self):
        self.sparql = SPARQLWrapper(self.SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)

    @lru_cache(maxsize=512)
    def find_descriptor(self, term: str) -> dict | None:
        """Kikeresi a kifejezéshez tartozó MeSH Descriptort (UI) a REST API-val."""
        for match_type in ("exact", "contains"):
            r = requests.get(f"{self.BASE_REST}/lookup/descriptor",
                             params={"label": term, "match": match_type,
                                     "limit": 5, "lang": "eng"})
            results = r.json()
            if results:
                for res in results:
                    if res.get("label", "").lower() == term.lower():
                        return res
                if match_type == "contains":
                    return results[0]
        return None

    def _run_sparql(self, query: str) -> dict:
        """Segédfüggvény a SPARQL lekérdezések futtatására."""
        self.sparql.setQuery(query)
        try:
            return self.sparql.queryAndConvert()
        except Exception as e:
            print(f"SPARQL hiba: {e}")
            return {"results": {"bindings": []}}

    @lru_cache(maxsize=512)
    def get_entry_terms(self, ui: str) -> list[str]:
        """Lekéri egy Descriptor összes releváns Concept-jét"""

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
        PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
        PREFIX mesh2026: <http://id.nlm.nih.gov/mesh/2026/>
        PREFIX mesh2025: <http://id.nlm.nih.gov/mesh/2025/>
        PREFIX mesh2024: <http://id.nlm.nih.gov/mesh/2024/>

        SELECT DISTINCT ?termLabel
        WHERE {{
            mesh:{ui} ?o1 ?c .
            ?c rdf:type meshv:Concept .
            ?c ?o2 ?term .
            ?term rdf:type meshv:Term .
            ?term meshv:prefLabel ?termLabel .
        }}
        """

        results = self._run_sparql(query)

        terms = [
            res["termLabel"]["value"]
            for res in results["results"]["bindings"]
            if "termLabel" in res
        ]
        return list(set(terms))

    def get_parents(self, ui: str) -> list[dict]:
        """Lekéri a szülő fogalmakat (Broader Descriptors) egyetlen SPARQL hívással."""
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
        PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
        PREFIX mesh2026: <http://id.nlm.nih.gov/mesh/2026/>
        PREFIX mesh2025: <http://id.nlm.nih.gov/mesh/2025/>
        PREFIX mesh2024: <http://id.nlm.nih.gov/mesh/2024/>

        SELECT DISTINCT ?parent ?parentLabel
        WHERE {{
          mesh:{ui} meshv:broaderDescriptor ?parent .
          ?parent rdfs:label ?parentLabel .
        }}
        """
        results = self._run_sparql(query)

        parents = []
        for binding in results["results"]["bindings"]:
            if "parent" in binding and "parentLabel" in binding:
                parent_uri = binding["parent"]["value"]
                parent_ui = parent_uri.split("/")[-1]
                parent_label = binding["parentLabel"]["value"]
                parents.append({"ui": parent_ui, "label": parent_label})

        return parents

if __name__ == "__main__":
    client = MeSHClient()

    for term in ["hypertension", "insulin resistance", "headache"]:
        print(f"Term: {term}")
        descriptor = client.find_descriptor(term)

        if descriptor:
            ui = descriptor["resource"].split("/")[-1]
            print(f"UI: {ui} | Label: {descriptor['label']}")

            synonyms = client.get_entry_terms(ui)
            print(f"Entry terms ({len(synonyms)}): {synonyms[:5]}...")

            parents = client.get_parents(ui)
            print(f"Parents: {[(p['ui'], p['label']) for p in parents]}")
        else:
            print("Not found.")