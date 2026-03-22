import ast
import re

import pandas as pd
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.config import *
from src.agents.rag_agent import MedicalRAG

TESTSET_CSV = EVAL_DATA_DIR / "ragas_testset.csv"
RESULTS_CSV = EVAL_DATA_DIR / "ragas_eval_results.csv"

SAMPLE_N = 25  # increase have budget

METRICS = [
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    Faithfulness(),
    ResponseRelevancy(),
    FactualCorrectness(),
]

ENGLISH_PROMPT = ChatPromptTemplate.from_messages([("system", """
Answer the question strictly based on the context below. In English.
If the answer is not in the context, say: "I cannot answer based on the available information."
Do not use any outside knowledge.

Context: {context}
Question: {input}
Keywords: {keywords}
""")])

_UUID_RE = re.compile(r"^[0-9a-f\-]{36}\s*\n\n", re.IGNORECASE)


def patch_english(agent: MedicalRAG):
    # Swap the Hungarian prompt for English during eval
    agent.qa_chain = create_stuff_documents_chain(llm=agent.llm, prompt=ENGLISH_PROMPT)


def parse_contexts(raw) -> list[str]:
    if isinstance(raw, list):
        return [_UUID_RE.sub("", str(c)).strip() for c in raw]
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [_UUID_RE.sub("", str(c)).strip() for c in parsed]
    except (ValueError, SyntaxError):
        pass
    return [_UUID_RE.sub("", str(raw)).strip()]


def run_rag(agent: MedicalRAG, question: str) -> tuple[str, list[str]]:
    keywords = agent.keyword_chain.invoke({"input": question}).content
    search_query = f"{question}\n{keywords}"
    docs = agent.vector_store.similarity_search(search_query, k=10)
    answer = agent.qa_chain.invoke({"context": docs, "input": question, "keywords": keywords})
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


def main():
    df = pd.read_csv(TESTSET_CSV, encoding="utf-8").sample(n=SAMPLE_N, random_state=42)

    agent = MedicalRAG()
    patch_english(agent)

    samples = []
    for _, row in df.iterrows():
        answer, contexts = run_rag(agent, row["user_input"])
        samples.append(SingleTurnSample(
            user_input=row["user_input"],
            response=answer,
            retrieved_contexts=contexts,
            reference=row["reference"],
            reference_contexts=parse_contexts(row["reference_contexts"]),
        ))

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=LangchainLLMWrapper(ChatGroq(
            model=LLM_OPENAI_GTP_OSS_20B,
            api_key=GROQ_API_KEY,
            temperature=0.0,
            max_tokens=16384,
        )),
        embeddings=LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)
        ),
        run_config=RunConfig(max_workers=1, max_retries=5, max_wait=60),
    )

    results_df = results.to_pandas()
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    non_metric_cols = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    metric_cols = [c for c in results_df.columns if c not in non_metric_cols]
    print(results_df[metric_cols].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()