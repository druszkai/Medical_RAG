from typing import Sequence, Any

from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
#from langchain_community.document_transformers import LongContextReorder

from src.agentic_logic.retrieval_logic import Retriever, Reranker
from src.agentic_logic.kge_chain import KGEChain
from src.config import *

REFUSAL_MSG = "A rendelkezésemre álló információk alapján nem tudok válaszolni"

HUNGARIAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Te egy professzionális orvosi asszisztens vagy. KIZÁRÓLAG az alábbi kontextus alapján válaszolj a kérdésre!

Kövesd pontosan az alábbi lépéseket:
1. BELSŐ ELEMZÉS: Keresd meg a kontextusban azokat a konkrét mondatokat, amelyek tartalmazzák a választ.
2. VÁLASZADÁS: Ha a kontextus tartalmazza a választ (akár csak részben is), fogalmazz meg egy egyértelmű, tényszerű és részletes feleletet. Közvetlenül a kérdésre válaszolj.
3. HIÁNYZÓ INFORMÁCIÓ: Ha a kontextus csak részben válaszolja meg a kérdést, add meg az elérhető információkat, majd egyértelműen jelezd, hogy mi hiányzik a kontextusból.
4. MEGTAGADÁS: Ha a kontextus teljesen irreleváns és semmilyen hasznos információt nem tartalmaz a kérdéshez, KIZÁRÓLAG ezt a mondatot írd le: "A rendelkezésemre álló információk alapján nem tudok válaszolni."

Szigorúan tilos külső tudást felhasználni vagy hallucinálni! Nyelvezete egyezzen meg a kérdés nyelvével!

Kontextus:
{context}"""),
    ("human", "Kérdés:\n{input}\n\nKulcsszavak:\n{keywords}")
])

KEYWORD_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are an expert medical librarian. Your task is to optimize the user's question for a vector database search.
Extract the core medical concepts, and add highly relevant clinical synonyms, alternate names, or broader medical terms.
Format your output STRICTLY as a simple, space-separated list of keywords. No commas, no bullet points, no conversational text.

User Question:
{input}
""")])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are a strict medical QA critic. Evaluate whether the answer adequately responds to the question based only on what was retrieved.

Respond with ONLY one of the following:
- ADEQUATE: the answer contains a clear, relevant response to the question
- RETRY: the answer is a refusal, too vague, or clearly incomplete

Question: {question}
Answer: {answer}
""")])

AGENT_SYSTEM_PROMPT = """You are a medical assistant with access to a medical knowledge base search tool.
Your job is to answer the user's medical question accurately.
Use the search tool to find relevant information. If the first result is inadequate, try again with broader or different terms.
Always base your final answer strictly on what the tool returns. Never use outside knowledge.
Respond in the same language as the question."""

NER_PROMPT = """
You are an expert medical entity extractor. Extract the core medical concepts from the user's text.
Focus ONLY on:
1. Diseases & Symptoms (e.g., headache, hypertension)
2. Medications & Active Ingredients (e.g., aspirin, lisinopril)
3. Herbs & Supplements (e.g., ginkgo biloba, vitamin D)

Return the output as a clean JSON list of strings, translated to English clinical terms if possible.
Example input: "Fáj a fejem és ginkgot szedek"
Example output: ["headache", "ginkgo biloba"]
"""

class MedicalRAG:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)

        vector_store = Chroma(
            persist_directory=CHROMA_DB,
            collection_name="medical_collection",
            embedding_function=embeddings,
        )

        self.retriever = Retriever(
            vector_store=vector_store,
            reranker=Reranker(
                top_k=8
            ),
            top_k=40
        )

        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O,
            temperature=0.0,
        )

        self.qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=HUNGARIAN_PROMPT,
        )

        keyword_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O_MINI,
            temperature=0.1,
            max_tokens=500,
        )

        self.keyword_chain = KEYWORD_PROMPT | keyword_llm

        critic_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_OPENAI_GPT_4O_MINI,
            temperature=0.0,
            max_tokens=10,
        )

        self.critic_chain = CRITIC_PROMPT | critic_llm

        # KGEChain builds a concept graph from retrieved chunks and uses it
        # to guide synthesis only instantiated once, reused across queries
        self.kge_chain = KGEChain()

        self._agent = self._build_agent()

    def _build_agent(self):
        @tool
        def search_medical_database(query: str) -> str:
            """Search the medical knowledge base for information relevant to the query.
            Use this for questions about cardiovascular health, symptoms, treatments,
            natural remedies, and lifestyle advice."""
            keywords = self.keyword_chain.invoke({"input": query}).content
            docs = self.retriever.retrieve_concepts_weighted(query, keywords)
            self._print_sources(docs)
            return self.qa_chain.invoke({
                "context": docs,
                "input": query,
                "keywords": keywords,
            })

        return create_agent(
            model=self.llm,
            tools=[search_medical_database],
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def _print_sources(self, docs: Sequence[Document]):
        pass
        #for i, doc in enumerate(docs, start=1):
        #    print(f"{i}.: {doc.metadata.get('title', 'Ismeretlen')} \n{doc.metadata.get('document_id')}\n")

    def query(self, query_str: str) -> str:
        """Single-pass RAG query."""
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_concepts_weighted(query_str, keywords)

            result = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })
            self._print_sources(docs)
            return result
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}"

    def query_iterative(self, query_str: str, max_iterations: int = 2) -> tuple[Any, Sequence[Document]] | tuple[
        str, list[Any]]:
        """Iterative RAG query with critic.
        Retries if the critic deems the answer inadequate"""
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_weighted(query_str, keywords)
            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })

            for _ in range(max_iterations - 1):
                verdict = self.critic_chain.invoke({
                    "question": query_str,
                    "answer": answer,
                }).content.strip()

                if "ADEQUATE" in verdict:
                    break

                refined_query = f"Provide different, broader search terms for this question: {query_str}"
                refined_keywords = self.keyword_chain.invoke({"input": refined_query}).content
                refined_docs = self.retriever.retrieve_weighted(query_str, refined_keywords)
                answer = self.qa_chain.invoke({
                    "context": refined_docs,
                    "input": query_str,
                    "keywords": refined_keywords,
                })

            self._print_sources(docs)
            return answer, docs

        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []

    def query_agent(self, query: str, max_iterations: int = 2) -> tuple[str, list[Any]] | Any:
        """Agent-based RAG query.
        The agent decides when and how to search."""
        try:
            result = self._agent.invoke({"messages": [("human", query)]})
            return result["messages"][-1].content
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []

    def query_with_expansion(self, query_str: str) -> tuple[Any, Sequence[Document]] | tuple[str, list[Any]]:
        """Multi-Query Expansion (Generates perspectives to capture vocabulary mismatch)"""
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_with_query_expansion(query_str, keywords)

            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })
            self._print_sources(docs)
            return answer, docs
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []

    def query_with_relevance_feedback(self, query_str: str) -> tuple[Any, Sequence[Document]] | tuple[str, list[Any]]:
        """Pseudo-Relevance Feedback.
        (Learns terminology from initial search before final search)"""
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_with_relevance_feedback(query_str, keywords)

            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })
            self._print_sources(docs)
            return answer, docs
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []

    def query_with_hyde(self, query_str: str) -> tuple[Any, Sequence[Document]] | tuple[str, list[Any]]:
        """HyDE (Hypothetical Document Embeddings) alapú lekérdezés.
        Hipotetikus választ generál először, azt embedeli a kereséshez."""
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_with_hyde(query_str, keywords)

            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })
            self._print_sources(docs)
            return answer, docs
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []

    def query_combined(self, query_str: str, max_iterations: int = 2) -> tuple[Any, Sequence[Document]] | tuple[
        str, list[Any]]:
        """
        HyDE alapú retrieval -> Súlyozott Reranking -> Iteratív QA.
        Ha az első kör elbukik, Query Expansion-re vált.
        """
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_with_hyde(query_str, keywords)
            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })

            for _ in range(max_iterations - 1):
                verdict = self.critic_chain.invoke({
                    "question": query_str,
                    "answer": answer,
                }).content.strip()

                if "ADEQUATE" in verdict:
                    break

                refined_query = f"Provide different, broader search terms and synonyms for this question: {query_str}"
                refined_keywords = self.keyword_chain.invoke({"input": refined_query}).content
                refined_docs = self.retriever.retrieve_with_query_expansion(
                    query_str, refined_keywords
                )

                answer = self.qa_chain.invoke({
                    "context": refined_docs,
                    "input": query_str,
                    "keywords": refined_keywords,
                })

            self._print_sources(docs)
            return answer, docs

        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", []


    def query_kge(self, query_str: str) -> tuple[str, Sequence[Document], dict]:
        """KGE-enriched RAG query.

        Flow:
          1. Retrieve docs with HyDE (same as query_combined)
          2. KGEChain extracts entity pairs from each chunk and builds a graph
          3. Finds the shortest conceptual path between the two anchor entities
          4. Synthesizes a final answer using both the graph context and raw docs

        Returns the answer, the retrieved docs, and KGE metadata
        (path, graph_size, relations) for logging during eval.
        """
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self.retriever.retrieve_with_hyde(query_str, keywords)
            result = self.kge_chain.run(query_str, docs)

            kge_meta = {
                "path": result["path"],             # conceptual chain found in the graph
                "graph_size": result["graph_size"], # how many unique entities were extracted
                "relations": result["relations"],   # all entity pairs found across chunks
            }

            self._print_sources(docs)
            return result["answer"], docs, kge_meta

        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}", [], {}


if __name__ == "__main__":
    agent = MedicalRAG()
    test_query = "Mit tudsz az akutt agyi infarktusról?"
    print(agent.query_iterative(test_query))
    print(agent.query_kge(test_query))
    #print(agent.query(test_query))