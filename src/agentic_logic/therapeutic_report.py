from concurrent.futures import ThreadPoolExecutor

from src.agentic_logic.retrieval_logic import Retriever
from src.agentic_logic.term_annotator import TermAnnotator

SECTIONS = [
    ("Clinical background", "mechanism pathomechanism risk factors", "weighted"),
    ("Treatment options", "treatment protocol first-line drug", "weighted"),
    ("Lifestyle and Natural remedies", "diet supplements herbs lifestyle", "layperson"),
    ("Warning signs and Side effects", "side effects contraindications warning", "weighted"),
]

class TherapeuticReport:
    def __init__(self, retriever: Retriever, qa_chain, critic_chain, annotator: TermAnnotator):
        self.retriever = retriever
        self.qa_chain = qa_chain
        self.critic_chain = critic_chain
        self.annotator = annotator

    def generate(self, text: str) -> str:
        # UC1: extract medical entities from input text
        entities = self.annotator.extract_entities(text)
        if not entities:
            return "Nem található orvosi kifejezés a szövegben."

        query = " ".join(entities)

        # 4 sections in parallel -> ThreadPool because Chroma + OpenAI are I/O-bound
        with ThreadPoolExecutor(max_workers=4) as pool:
            sections = list(pool.map(lambda s: self._section(query, *s), SECTIONS))

        return "\n\n".join(sections)

    def _section(self, query: str, name: str, suffix: str, mode: str) -> str:
        # L0 retrieval: weighted (PubMed-leaning) or layperson (lifestyle-leaning)
        docs = self._retrieve(query, suffix, mode)
        answer = self._qa(query, suffix, name, docs)

        # Critic gate. If inadequate -> hierarchical fallback walks L0 -> L1
        # (MeSH/RxNorm synonyms) -> L2 (parent concepts) via the existing
        # retrieve_hierarchical method.
        verdict = self.critic_chain.invoke({
            "question": f"{query} — {name}",
            "answer": answer,
        }).content.strip()

        if "ADEQUATE" not in verdict:
            fallback_docs, _ = self.retriever.retrieve_hierarchical(query, suffix, max_level=2)
            if fallback_docs:
                answer = self._qa(query, suffix, name, fallback_docs)

        return f"## {name}\n\n{answer}"

    def _retrieve(self, query: str, suffix: str, mode: str):
        if mode == "layperson":
            return self.retriever.retrieve_layperson(f"{query} {suffix}")
        return self.retriever.retrieve_weighted(query, suffix)

    def _qa(self, query: str, suffix: str, name: str, docs) -> str:
        return self.qa_chain.invoke({
            "context": docs,
            "input": f"{query} — {name}",
            "keywords": suffix,
        })
