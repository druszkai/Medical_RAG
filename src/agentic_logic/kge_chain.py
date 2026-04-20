import json
from typing import Sequence

import networkx as nx
from rapidfuzz import fuzz, process as fuzz_process
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import *

RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a biomedical relation extractor.
Given a medical text, identify pairs of entities that have an EXPLICIT connection stated in the text.
Do NOT infer relationships — only extract what is clearly written.
Output ONLY a JSON array of ["source", "target"] string pairs. If none found, output [].

Example: [["beta-blocker", "heart rate"], ["hypertension", "left ventricular hypertrophy"]]"""),
    ("human", "{chunk}")
])

ANCHOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract the two primary medical entities from the question whose relationship the answer should explain.
Output ONLY valid JSON: {{"source": "<entity1>", "target": "<entity2>"}}
Use English clinical terms."""),
    ("human", "{query}")
])

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional medical assistant.
Answer the question using the retrieved documents and the graph context below.
The graph context maps conceptual connections found within the retrieved evidence — use it to structure your explanation.
Base your answer STRICTLY on the provided information. Do not use outside knowledge.
Respond in the same language as the question."""),
    ("human", """Question: {query}

Graph Context:
{graph_context}

Retrieved Documents:
{context}""")
])


class KGEChain:
    def __init__(self):
        self.fast_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            model=LLM_OPENAI_GPT_4O_MINI,
            max_tokens=800,
        )
        self.strong_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            model=LLM_OPENAI_GPT_4O,
            max_tokens=1200,
        )
        self.relation_extraction_chain = RELATION_EXTRACTION_PROMPT | self.fast_llm
        self.anchor_chain = ANCHOR_PROMPT | self.fast_llm
        self.synthesis_chain = SYNTHESIS_PROMPT | self.strong_llm

    def run(self, query: str, docs: Sequence[Document], retriever=None) -> dict:
        """
        Returns:
            answer      – final answer string
            path        – detected conceptual chain (list[str])
            graph_size  – number of nodes in the mini graph (int)
            gap_fills   – always 0; gap-filling is intentionally excluded
            relations   – all extracted entity pairs (list[dict])
        """
        source, target = self._extract_anchors(query)
        relations, doc_map = self._extract_relations(docs)
        graph = self._build_graph(relations, doc_map)

        src_node = self._fuzzy_match(graph, source)
        tgt_node = self._fuzzy_match(graph, target)

        path = []
        if src_node and tgt_node and src_node != tgt_node:
            try:
                path = nx.shortest_path(graph, src_node, tgt_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        answer = self._synthesize(query, path, graph, docs)

        return {
            "answer": answer,
            "path": path,
            "graph_size": graph.number_of_nodes(),
            "gap_fills": 0,
            "relations": relations,
        }

    def _extract_anchors(self, query: str) -> tuple[str, str]:
        raw = self.anchor_chain.invoke({"query": query}).content.strip()
        try:
            parsed = json.loads(raw)
            return parsed.get("source", ""), parsed.get("target", "")
        except json.JSONDecodeError:
            print(f"[KGE] Anchor extraction JSON error: {raw}")
            return "", ""

    def _extract_relations(self, docs: Sequence[Document]) -> tuple[list[dict], dict[tuple, str]]:
        relations = []
        doc_map: dict[tuple, str] = {}

        for doc in docs:
            raw = self.relation_extraction_chain.invoke({"chunk": doc.page_content}).content.strip()
            try:
                pairs = json.loads(raw)
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        src = str(pair[0]).lower().strip()
                        tgt = str(pair[1]).lower().strip()
                        if src and tgt:
                            relations.append({"source": src, "target": tgt})
                            doc_map.setdefault((src, tgt), doc.page_content)
            except (json.JSONDecodeError, ValueError):
                continue

        return relations, doc_map

    def _build_graph(self, relations: list[dict], doc_map: dict[tuple, str]) -> nx.Graph:
        graph = nx.Graph()
        for rel in relations:
            src, tgt = rel["source"], rel["target"]
            graph.add_edge(src, tgt, source_chunk=doc_map.get((src, tgt), "")[:200])
        return graph

    def _fuzzy_match(self, graph: nx.Graph, entity: str) -> str | None:
        if not entity or graph.number_of_nodes() == 0:
            return None
        result = fuzz_process.extractOne(
            entity.lower(), list(graph.nodes()), scorer=fuzz.WRatio, score_cutoff=70
        )
        return result[0] if result else None

    def _synthesize(self, query: str, path: list[str], graph: nx.Graph, docs: Sequence[Document]) -> str:
        if path:
            graph_context = f"Conceptual path: {' → '.join(path)}\nNeighbors:\n"
            graph_context += "\n".join(
                f"  {node} → [{', '.join(list(graph.neighbors(node))[:6])}]"
                for node in path
                if list(graph.neighbors(node))
            )
        else:
            top_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:8]
            lines = [
                f"  {node} → [{', '.join(list(graph.neighbors(node))[:5])}]"
                for node, deg in top_nodes if deg > 0
            ]
            graph_context = ("Key concept connections:\n" + "\n".join(lines)) if lines else "No graph structure found."

        context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)

        return self.synthesis_chain.invoke({
            "query": query,
            "graph_context": graph_context,
            "context": context_text,
        }).content
