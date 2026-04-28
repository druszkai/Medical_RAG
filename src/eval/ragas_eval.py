import sys
import asyncio
from typing import Any

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import ast
import re
import httpx
import pandas as pd

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    #FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    #LLMContextRecall,
    ResponseRelevancy,
)

from src.agentic_logic.rag_agent import MedicalRAG
from src.config import *

TESTSET_CSV = EVAL_DATA_DIR / "ragas_testset.csv"
MANUAL_CSV = EVAL_DATA_DIR / "manual_testset.csv"
RESULTS_CSV = EVAL_DATA_DIR / "ragas_eval_results.csv"
RESULTS_MANUAL_CSV = EVAL_DATA_DIR / "ragas_eval_results_manual.csv"
RESULTS_COMBINED_CSV = EVAL_DATA_DIR / "ragas_eval_results_combined.csv"

# KGE eval: mechanistic questions vs. plain baseline on the same testset
KGE_TESTSET_CSV = EVAL_DATA_DIR / "kge_testset.csv"
KGE_RESULTS_BASELINE_CSV = EVAL_DATA_DIR / "ragas_kge_baseline_results.csv"
KGE_RESULTS_KGE_CSV = EVAL_DATA_DIR / "ragas_kge_results.csv"

METRICS = [
    #LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    Faithfulness(),
    ResponseRelevancy(),
    #FactualCorrectness(),
]

ENGLISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert medical assistant. Answer the question based ONLY on the provided context.

Instructions:
- Analyze the context carefully before answering.
- Provide a comprehensive response that addresses every part of the user's question using ONLY the provided text.
- Structure your response clearly. Use bullet points if you are listing multiple symptoms, treatments, or facts.
- If the context only partially answers the question, provide the available information directly and confidently. 
- If the context is completely unrelated and contains NO helpful information, reply ONLY with: "I cannot answer based on the available information."
- Do not use outside knowledge.

Context:
{context}"""),
    ("human", "Question: {input}\nExpanded Search Terms: {keywords}")
])

_UUID_RE = re.compile(r"^[0-9a-f\-]{36}\s*\n\n", re.IGNORECASE)


def patch_english(agent: MedicalRAG):
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
    #keywords = agent.keyword_chain.invoke({"input": question}).content
    #docs = agent.retriever.retrieve_concepts_weighted(question, keywords)
    #answer = agent.qa_chain.invoke({"context": docs, "input": question, "keywords": keywords})
    answer, docs = agent.query_combined(question)
    contexts = [doc.page_content for doc in docs]

    return answer, contexts

def run_rag_hierarchical(agent: MedicalRAG, question: str) -> tuple[Any, list[Any], Any]:
    keywords = agent.keyword_chain.invoke({"input": question,}).content
    docs, level_used = agent.retriever.retrieve_hierarchical(question, keywords)
    answer = agent.qa_chain.invoke({
        "context": docs,
        "input": question,
        "keywords": keywords,
    })
    return answer, [doc.page_content for doc in docs], level_used

def df_to_samples(df: pd.DataFrame, agent: MedicalRAG) -> list[SingleTurnSample]:
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
    return samples


def run_eval(samples, llm, embeddings) -> pd.DataFrame:
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=8, max_retries=5, max_wait=120),
    )
    return results.to_pandas()


def print_metrics(label: str, df: pd.DataFrame):
    non_metric_cols = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    metric_cols = [c for c in df.columns if c not in non_metric_cols]
    print(f"\n--- {label} ---")
    print(df[metric_cols].agg(["mean", "std"]).round(4))


def run_rag_kge(agent: MedicalRAG, question: str) -> tuple[str, list[str]]:
    """KGE-enriched pipeline runner for eval.

    Same interface as run_rag so it can be dropped into df_to_samples as-is.
    Prints graph stats per question so you can see whether the graph was useful.
    """
    answer, docs, kge_meta = agent.query_kge(question)
    # Log the graph path so you can manually inspect whether it matched the question
    print(f"  [KGE] nodes={kge_meta.get('graph_size', 0)}, path={kge_meta.get('path', [])}")
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


def df_to_samples_with(df: pd.DataFrame, agent: MedicalRAG, runner) -> list[SingleTurnSample]:
    """Like df_to_samples but accepts any runner function (run_rag or run_rag_kge).

    This avoids duplicating the loop, just pass a different runner to switch pipelines.
    """
    samples = []
    for _, row in df.iterrows():
        answer, contexts = runner(agent, row["user_input"])
        samples.append(SingleTurnSample(
            user_input=row["user_input"],
            response=answer,
            retrieved_contexts=contexts,
            reference=row["reference"],
            reference_contexts=parse_contexts(row["reference_contexts"]),
        ))
    return samples


def main_kge():
    """Compares plain RAG vs. KGE-enriched RAG on mechanistic questions.

    Runs the same testset through both pipelines and saves results side by side.
    The interesting metrics are Faithfulness (does the answer stay grounded?)
    and ResponseRelevancy (does it actually answer the mechanism question?).
    """
    kge_df = pd.read_csv(KGE_TESTSET_CSV, encoding="utf-8")
    print(f"KGE testset: {len(kge_df)} mechanistic questions")

    agent = MedicalRAG()
    patch_english(agent)

    custom_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=300.0, connect=30.0),
        http2=False
    )

    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_OPENAI_GPT_4O_MINI,
        api_key=OPENAI_API_KEY,
        temperature=0.0,
        max_tokens=4096,
        max_retries=5,
        http_async_client=custom_http_client,
    ))

    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)
    )

    # Run both pipelines on the exact same questions so the comparison is fair
    print("\nRunning baseline (query_combined) on KGE testset...")
    baseline_samples = df_to_samples_with(kge_df, agent, run_rag)

    print("\nRunning KGE-enriched pipeline on KGE testset...")
    kge_samples = df_to_samples_with(kge_df, agent, run_rag_kge)

    print("\nEvaluating baseline...")
    baseline_results = run_eval(baseline_samples, eval_llm, eval_embeddings)
    baseline_results.to_csv(KGE_RESULTS_BASELINE_CSV, index=False, encoding="utf-8")

    print("Evaluating KGE-enriched...")
    kge_results = run_eval(kge_samples, eval_llm, eval_embeddings)
    kge_results.to_csv(KGE_RESULTS_KGE_CSV, index=False, encoding="utf-8")

    print_metrics("Baseline (query_combined)", baseline_results)
    print_metrics("KGE-enriched", kge_results)


def main_baseline():
    auto_df = pd.read_csv(TESTSET_CSV, encoding="utf-8") #.sample(n=125, random_state=42)
    manual_df = pd.read_csv(MANUAL_CSV, encoding="utf-8") #.sample(n=25, random_state=42)

    print(f"Auto testset: {len(auto_df)} samples")
    print(f"Manual testset: {len(manual_df)} samples")

    agent = MedicalRAG()
    patch_english(agent)

    custom_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=300.0, connect=30.0),
        http2=False
    )

    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_OPENAI_GPT_4O_MINI,
        api_key=OPENAI_API_KEY,
        temperature=0.0,
        max_tokens=4096,
        max_retries=5,
        http_async_client=custom_http_client,
    ))

    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)
    )

    print("\nRunning RAG on auto-generated tests...")
    auto_samples = df_to_samples(auto_df, agent)

    print("Running RAG on manual tests...")
    manual_samples = df_to_samples(manual_df, agent)

    print("\nEvaluating auto-generated testset...")
    auto_results = run_eval(auto_samples, eval_llm, eval_embeddings)
    auto_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    print("Evaluating manual testset...")
    manual_results = run_eval(manual_samples, eval_llm, eval_embeddings)
    manual_results.to_csv(RESULTS_MANUAL_CSV, index=False, encoding="utf-8")

    #print("Evaluating combined testset...")
    #combined_results = run_eval(auto_samples + manual_samples, eval_llm, eval_embeddings)
    #combined_results.to_csv(RESULTS_COMBINED_CSV, index=False, encoding="utf-8")

    try:
        print("\nCombining results (no-rerun)...")
        combined_results = pd.concat([auto_results, manual_results], ignore_index=True)
        combined_results.to_csv(RESULTS_COMBINED_CSV, index=False, encoding="utf-8")
        print_metrics("Combined", combined_results)
    except Exception as e:
        print(e)

    print_metrics("Auto-generated", auto_results)
    print_metrics("Manual", manual_results)

if __name__ == "__main__":
    main_baseline()