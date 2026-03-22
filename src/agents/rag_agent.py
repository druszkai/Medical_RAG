from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import *

REFUSAL_MSG = "A rendelkezésemre álló információk alapján nem tudok válaszolni"

HUNGARIAN_PROMPT = ChatPromptTemplate.from_messages([("system", """
KIZÁRÓLAG az alábbi kontextus alapján válaszolj a kérdésre!
Ha a kontextusban nem található meg a válasz a kérdésre, KÖTELEZŐ ezt mondanod: "A rendelkezésemre álló információk alapján nem tudok válaszolni a kérdésre."
Ne találj ki semmilyen egyéb információt, és ne használj fel külső tudást.
Válaszolj részletesen, minden releváns információt foglalj bele a kontextusból! Nyelve egyezzen meg a kérdés nyelvével!

Kontextus:
{context}

Kérdés:
{input}

Kulcsszavak:
{keywords}
""")])

KEYWORD_PROMPT = ChatPromptTemplate.from_messages([("system", """
You are a medical assistant. Based on the user's question, provide 3-5 English search keywords or short phrases related to the topic.
Do not provide any other conversational text. Format your output strictly as follows:

English: [word1], [word2], [word3], ...

User Question:
{input}
""")])


class MedicalRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)

        self.vector_store = Chroma(
            persist_directory=CHROMA_DB,
            collection_name="medical_collection",
            embedding_function=self.embeddings,
        )

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_LLAMA_3_3_70B_VERSATILE,
            temperature=0.0,
        )

        self.qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=HUNGARIAN_PROMPT,
        )

        self.keyword_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_OPENAI_GTP_OSS_20B,
            temperature=0.1,
            max_tokens=500,
        )

        self.keyword_chain = KEYWORD_PROMPT | self.keyword_llm

    def _retrieve(self, query: str, keywords: str) -> list:
        search_query = f"{query}\n{keywords}"
        docs = self.vector_store.similarity_search(search_query, k=20)
        pairs = [[keywords, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        return [doc for _, doc in sorted(zip(scores, docs), reverse=True)][:5]

    def _print_sources(self, docs):
        for i, doc in enumerate(docs, start=1):
            print(f"{i}.: {doc.metadata.get('title', 'Ismeretlen')} \n{doc.metadata.get('document_id')}\n")

    def query(self, query_str: str) -> str:
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self._retrieve(query_str, keywords)
            result = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })
            self._print_sources(docs)
            return result
        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}"

    def query_iterative(self, query_str: str, max_iterations: int = 2) -> str:
        # Uses the first answer to refine retrieval in a second pass.
        # Falls back to the first answer if the second pass returns a refusal.
        try:
            keywords = self.keyword_chain.invoke({"input": query_str}).content
            docs = self._retrieve(query_str, keywords)
            answer = self.qa_chain.invoke({
                "context": docs,
                "input": query_str,
                "keywords": keywords,
            })

            for _ in range(max_iterations - 1):
                if REFUSAL_MSG in answer:
                    break

                refined_query = f"{query_str}\n{answer}"
                refined_keywords = self.keyword_chain.invoke({"input": refined_query}).content
                refined_docs = self._retrieve(refined_query, refined_keywords)
                refined_answer = self.qa_chain.invoke({
                    "context": refined_docs,
                    "input": query_str,
                    "keywords": refined_keywords,
                })

                if REFUSAL_MSG not in refined_answer:
                    answer = refined_answer
                    docs = refined_docs

            self._print_sources(docs)
            return answer

        except Exception as e:
            return f"Hiba történt a lekérdezés során: {str(e)}"


if __name__ == "__main__":
    agent = MedicalRAG()
    test_query = "Mit tudsz az akutt agyi infarktusról?"
    print(agent.query_iterative(test_query))