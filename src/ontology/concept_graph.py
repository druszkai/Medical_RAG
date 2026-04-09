from dataclasses import dataclass, field
from src.ontology.mesh_client import MeSHClient
from src.ontology.rxnorm_client import RxNormClient

@dataclass
class ConceptNode:
    term: str
    mesh_ui: str | None = None
    rxcui: str | None = None
    synonyms: list[str] = field(default_factory=list)
    parent_terms: list[str] = field(default_factory=list)
    level: int = 0 # 0 - eredeti, 1 - szinonima, 2 - szülő, 3 - nagyszülő

class ConceptGraph:
    def __init__(self):
        self.mesh = MeSHClient()
        self.rxnorm = RxNormClient()

    def build_concept_graph(self, entities: list[str]) -> list[ConceptNode]:
        """Build concept graph"""
        nodes: list[ConceptNode] = []
        for entity in entities:
            node = ConceptNode(term=entity, level=0)

            descriptor = self.mesh.find_descriptor(entity)
            if descriptor:
                node.mesh_ui = descriptor["resource"].split("/")[-1]
                node.synonyms = self.mesh.get_entry_terms(node.mesh_ui)
                node.parent_terms = [p["label"] for p in self.mesh.get_parents(node.mesh_ui)]

            rxcui = self.rxnorm.find_rxcui(entity)
            if rxcui:
                node.rxcui = rxcui
                rx_synonyms = self.rxnorm.get_synonyms(rxcui)
                node.synonyms = list(set(node.synonyms + rx_synonyms))

            nodes.append(node)
        return nodes

    def get_levels(self, nodes: list[ConceptNode]) -> dict[int, list[str]]:
        levels: dict[int, list[str]] = {0: [], 1: [], 2: []}
        for node in nodes:
            levels[0].append(node.term)
            levels[1].extend(node.synonyms)
            levels[2].extend(node.parent_terms)
        return {k: list(set(v)) for k, v in levels.items() if v}
