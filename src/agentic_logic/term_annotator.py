"""
UC1: TermAnnotator
---
Takes a medical text (Hungarian or English) and returns a plain-language
explanation for each medical term found in it.

Flow:
  input text
    1.) NER chain          - extract medical terms (translated to English)
    2.) synonym enrichment - MeSH / RxNorm L1 synonyms for better retrieval
    3.) per-term retrieval - layperson-weighted search for each term
    4.) annotation chain   - 1-2 sentence plain-language explanation per term
    5.) dict {term: explanation}

Optional second output: rewrite() rewrites the full original text in plain
language using the annotations as a glossary (one extra API call).
"""

import json
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agentic_logic.retrieval_logic import Retriever
from src.ontology.concept_graph import ConceptGraph
from src.config import OPENAI_API_KEY, LLM_OPENAI_GPT_4O_MINI, LLM_OPENAI_GPT_4O

# Prompts

NER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert medical entity extractor.
Extract the core medical concepts from the text below. Focus ONLY on:
  1. Diseases and conditions (e.g. hypertension, myocardial infarction)
  2. Medications and active ingredients (e.g. aspirin, lisinopril)
  3. Herbs, supplements and nutrients (e.g. omega-3, vitamin D)
  4. Symptoms and clinical findings (e.g. bradycardia, oedema)

Return a clean JSON list of strings in English clinical terms.
If the input is Hungarian, translate the terms to English.
If no medical terms are found, return [].
Output ONLY the JSON array — no explanation, no preamble.

Example: ["hypertension", "lisinopril", "oedema"]"""),
    ("human", "{text}")
])

ANNOTATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical communicator who explains clinical terms to patients
with no medical background.
Given a medical term and relevant passages from health articles, write a
1-2 sentence plain-language explanation of what the term means.

Rules:
- Use everyday language (aim for an average reading level)
- Do NOT introduce other medical jargon in the explanation
- Base your answer STRICTLY on the provided context. No outside knowledge
- If the context does not contain enough information to explain the term,
  respond with exactly: "No clear explanation found in available sources."

Respond with ONLY the explanation, nothing else."""),
    ("human", """Term: {term}

Context:
{context}""")
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical communicator. Rewrite the medical text below in plain
language for a patient with no medical background.
Use the provided glossary to replace or clarify technical terms.
Preserve the original meaning exactly. Do not add or remove information.
Aim for a grade 6-8 reading level."""),
    ("human", """Original text:
{text}

Glossary (plain explanation from term):
{glossary}""")
])

class TermAnnotator:
    def __init__(self, retriever: Retriever):
        # Injected retriever to avoid multiple connections
        self.retriever = retriever
        self.concept_graph = ConceptGraph()

        fast_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O_MINI,
            temperature=0.0,
            max_tokens=300,
        )
        strong_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O,
            temperature=0.0,
            max_tokens=300,
        )

        self.ner_chain = NER_PROMPT | fast_llm
        self.annotation_chain = ANNOTATION_PROMPT | strong_llm
        self.rewrite_chain = REWRITE_PROMPT | strong_llm

    def annotate(self, text: str) -> dict[str, str]:
        """Returns {medical_term: layperson_explanation}.

        Each term is retrieved and explained independently so explanations
        stay focused. Synonym enrichment improves recall for variant spellings
        and abbreviations (e.g. "HTN" -> finds "hypertension" docs).
        """
        entities = self._extract_entities(text)
        if not entities:
            return {}

        synonym_map = self._get_synonyms(entities)

        annotations = {}
        for entity in entities:
            explanation, _ = self._explain_term(entity, synonym_map.get(entity, []))
            annotations[entity] = explanation

        return annotations

    def annotate_single(self, term: str) -> tuple[str, Sequence[Document]]:
        """Explain one pre-identified term. Skips the NER step.

        Used by the eval pipeline where we already know the term from the testset.
        Returns (explanation, retrieved_docs) so the eval can capture contexts.
        """
        synonyms = self._get_synonyms([term]).get(term, [])
        return self._explain_term(term, synonyms)

    def rewrite(self, text: str, annotations: dict[str, str]) -> str:
        """Rewrite the full text in plain language using annotations as a glossary.

        Call annotate() first and pass the result here.
        This is a single LLM call, so no additional retrieval is needed.
        """
        glossary = "\n".join(
            f"- {term}: {explanation}"
            for term, explanation in annotations.items()
        )
        return self.rewrite_chain.invoke({
            "text": text,
            "glossary": glossary,
        }).content.strip()

    def _extract_entities(self, text: str) -> list[str]:
        """Calls NER chain and parses the JSON result."""
        raw = self.ner_chain.invoke({"text": text}).content.strip()
        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except json.JSONDecodeError:
            print(f"[TermAnnotator] NER JSON parse error: {raw}")
        return []

    def _get_synonyms(self, entities: list[str]) -> dict[str, list[str]]:
        """Returns MeSH/RxNorm synonyms (L1) for each entity.

        Synonyms are appended to the retrieval query so that variant spellings
        and abbreviations still match relevant documents.
        Capped at 4 synonyms per term to keep the query focused.
        """
        nodes = self.concept_graph.build_concept_graph(entities)
        levels = self.concept_graph.get_levels(nodes)

        # L0 = original terms, L1 = MeSH entry terms + RxNorm synonyms
        l1_terms = set(levels.get(1, []))

        synonym_map: dict[str, list[str]] = {}
        for node in nodes:
            # Keep only synonyms that actually came from this node's lookups
            node_synonyms = [s for s in node.synonyms if s in l1_terms and s.lower() != node.term.lower()]
            synonym_map[node.term] = node_synonyms[:4]

        return synonym_map

    def _explain_term(self, term: str, synonyms: list[str]) -> tuple[str, Sequence[Document]]:
        """Retrieves layperson docs for a term and generates a plain explanation.

        The search query is the term + its top synonyms so variant spellings
        don't silently miss relevant documents.
        Returns (explanation_text, retrieved_docs) so callers can capture contexts.
        """
        # Build enriched search query from term + synonyms
        search_query = term + (" " + " ".join(synonyms) if synonyms else "")
        docs = self.retriever.retrieve_layperson(search_query)

        if not docs:
            return "No explanation found in available sources.", []

        # Use only the top 4 docs for the annotation context, just enough signal
        # without bloating the prompt
        context = "\n\n---\n\n".join(doc.page_content for doc in docs[:4])
        explanation = self.annotation_chain.invoke({
            "term": term,
            "context": context,
        }).content.strip()

        return explanation, docs
