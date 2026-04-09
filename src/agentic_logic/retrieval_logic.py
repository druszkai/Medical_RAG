import numpy as np
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from src.config import OPENAI_API_KEY, LLM_OPENAI_GPT_4O_MINI
from src.ontology.concept_graph import ConceptGraph

RRF_K = 60
QUERY_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

CONCEPT_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are an expert medical librarian. Extract structured concepts from the question below.
Include BOTH clinical terminology AND natural/lifestyle equivalents where relevant.
Output only a space-separated list. No explanations, no punctuation.

Cover: conditions, symptoms, treatments, drugs, body parts, 
natural remedies, herbs, supplements, dietary approaches, lifestyle factors.

Question: {input}
""")])

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are an expert medical researcher. Generate 3 distinct variations of the user's question to optimize database retrieval.
Focus the variations on:
1. Underlying clinical mechanisms or anatomy.
2. Common symptoms or diagnostic criteria.
3. Standard treatments, drugs, or interventions.

Output ONLY the 3 questions, separated by newlines. Do not number them. No conversational text.

Question: {input}
""")])

PRF_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are an expert medical librarian. I will provide a user question and a few initial medical documents.
Read these documents and extract 5-7 highly specific clinical keywords or phrases that actually appear in the text and are highly relevant to answering the question.
Output ONLY a space-separated list of keywords.

Question: {question}

Initial Documents:
{documents}
""")])

HYDE_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are a medical knowledge base article writer.
Write a short, factual passage (3-5 sentences) that directly answers the question below.
The passage should read like an excerpt from a health or home-remedy article —
use plain language, mention both clinical terms and common natural/lifestyle approaches where relevant.
Do NOT address the user. Do NOT say "this article". Just write the passage.

Question: {input}
""")])

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", top_k: int = 8):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, keywords: str, docs: list[Document]) -> Sequence[Document]:
        if keywords:
            query = f"{query}\n{keywords}"
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key = lambda x: x[0], reverse = True)
        return [doc for _, doc in ranked][:self.top_k]

    def rerank_weighted(self, query: str, keywords: str, docs: list[Document]) -> Sequence[Document]:
        query_pairs = [[query, doc.page_content] for doc in docs]
        keyword_pairs = [[keywords, doc.page_content] for doc in docs]

        query_scores = np.array(self.model.predict(query_pairs))
        keyword_scores = np.array(self.model.predict(keyword_pairs))

        combined = QUERY_WEIGHT * query_scores + KEYWORD_WEIGHT * keyword_scores
        ranked = sorted(zip(combined, docs), key = lambda x: x[0], reverse = True)
        return [doc for _, doc in ranked][:self.top_k]

class Retriever:
    def __init__(self, vector_store: Chroma, reranker: Reranker, top_k: int = 25):
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k

        fast_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O_MINI,
            temperature=0.0,
            max_tokens=100,
        )
        self.concept_chain = CONCEPT_PROMPT | fast_llm

        self.query_expansion_chain = MULTI_QUERY_PROMPT | fast_llm  # NEW
        self.relevance_chain = PRF_PROMPT | fast_llm

        self.hyde_chain = HYDE_PROMPT | fast_llm

        self.concept_graph = ConceptGraph()


    def retrieve_standard(self, query: str, keywords: str) -> Sequence[Document]:
        if keywords:
            search_query = f"{query}\n{keywords}"
        else:
            search_query = query

        docs = self.vector_store.similarity_search(search_query, k=self.top_k)
        return self.reranker.rerank(query, keywords, docs)

    def retrieve_weighted(self, query: str, keywords: str) -> Sequence[Document]:
        search_query = f"{query}\n{keywords}"
        docs = self.vector_store.similarity_search(search_query, k=self.top_k)
        return self.reranker.rerank_weighted(query, keywords, docs)

    def retrieve_rrf(self, query: str, keywords: str) -> Sequence[Document]:
        query_docs = self.vector_store.similarity_search(query, k=self.top_k)
        keyword_docs = self.vector_store.similarity_search(keywords, k=self.top_k) if keywords else []

        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}

        for rank, doc, in enumerate(query_docs):
            doc_id = doc.metadata.get("document_id", doc.page_content[:50])
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + RRF_K)
            docs[doc_id] = doc

        for rank, doc in enumerate(keyword_docs):
            doc_id = doc.metadata.get("document_id", doc.page_content[:50])
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + RRF_K)
            docs[doc_id] = doc

        merged_docs = [docs[doc_id] for doc_id in sorted(scores, key=scores.__getitem__, reverse=True)][:self.top_k]
        return self.reranker.rerank(query, keywords, merged_docs)

    def retrieve_with_concept(self, query: str, keywords: str) -> Sequence[Document]:
        concepts = self.concept_chain.invoke({"input": query}).content.strip()
        enriched_query = f"{query}\n{keywords}\n{concepts}" if keywords else f"{query}\n{concepts}"
        docs = self.vector_store.similarity_search(enriched_query, k=self.top_k)
        return self.reranker.rerank(query, keywords, docs)

    def retrieve_concepts_weighted(self, query: str, keywords: str) -> Sequence[Document]:
        concepts = self.concept_chain.invoke({"input": query}).content.strip()
        search_query = f"{query}\n{keywords}" if keywords else query
        docs = self.vector_store.similarity_search(search_query, k=self.top_k)
        return self.reranker.rerank(query, concepts, docs)

    def retrieve_with_query_expansion(self, query: str, keywords: str) -> Sequence[Document]:
        variations_text = self.query_expansion_chain.invoke({"input": query}).content.strip()
        queries = variations_text.split('\n')
        queries.extend([query, keywords])

        unique_docs = {}
        for q in queries:
            if not q.strip(): continue
            docs = self.vector_store.similarity_search(q, k=10)
            for doc in docs:
                doc_id = doc.page_content[:50]
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc

        merged_docs = list(unique_docs.values())

        return self.reranker.rerank_weighted(query, keywords, merged_docs)

    def retrieve_with_relevance_feedback(self, query: str, keywords: str) -> Sequence[Document]:
        initial_search = f"{query}\n{keywords}" if keywords else query
        initial_docs = self.vector_store.similarity_search(initial_search, k=3)

        doc_text = "\n\n".join([d.page_content for d in initial_docs])
        prf_keywords = self.relevance_chain.invoke({"question": query, "documents": doc_text}).content.strip()

        final_query = f"{query}\n{keywords}\n{prf_keywords}"
        final_docs = self.vector_store.similarity_search(final_query, k=self.top_k)

        return self.reranker.rerank_weighted(query, f"{keywords} {prf_keywords}", final_docs)

    def retrieve_with_hyde(self, query: str, keywords: str) -> Sequence[Document]:
        """Hypothetical Document Embeddings (HyDE) retrieval.

        Instead of embedding the question itself, generates a hypothetical ideal
        answer first and embeds that for similarity search. Bridges the
        question–document embedding gap, especially useful for mixed corpora
        (clinical + home-remedy / lifestyle content).
        """
        hypothetical_doc = self.hyde_chain.invoke({"input": query}).content.strip()
        docs = self.vector_store.similarity_search(hypothetical_doc, k=self.top_k)
        return self.reranker.rerank_weighted(query, keywords, docs)

    def retrieve_hierarchical(self, query: str, keywords: str, max_level: int = 2) -> tuple[Sequence[Document], int]:
        raw_concepts = self.concept_chain.invoke({"input": query}).content.strip()
        entities = raw_concepts.split()

        nodes = self.concept_graph.build_concept_graph(entities)
        levels = self.concept_graph.get_levels(nodes)

        for level_idx in range(max_level + 1):
            if level_idx not in levels:
                continue

            level_terms = " ".join(levels[level_idx])
            search_query = f"{query}\n{keywords}\n{level_terms}"

            docs = self.vector_store.similarity_search(search_query, k=self.top_k)
            reranked = self.reranker.rerank(query, keywords, docs)

            if reranked:
                return reranked, level_idx

        return [], -1

