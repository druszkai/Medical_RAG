import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import random

import httpx
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

from src.config import *

# Two personas — kept after previous model caused issues with more
PERSONAS = [
    Persona(name="student", role_description="A student looking to understand medical topics"),
    Persona(name="medical professional", role_description="A professional seeking detailed clinical information"),
]


def generate_tests(input_json_file, output_csv_file, sample_size, test_size):
    with open(PROCESSED_DATA_DIR / input_json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sample_data = random.sample(raw_data, sample_size)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = []
    for item in sample_data:
        text = item.get("text", "").strip()
        if text:
            for chunk in splitter.split_text(text):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": item.get("source", "unknown")},
                ))

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=300.0, connect=30.0),
        http2=False,
    )

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_OPENAI_GPT_4O_MINI,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=4096,
        max_retries=5,
        http_async_client=http_client,
    ))

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)
    )

    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=120)

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, persona_list=PERSONAS)
    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        run_config=run_config,
    )

    df = dataset.to_pandas()
    df.to_csv(EVAL_DATA_DIR / output_csv_file, index=False, encoding="utf-8")
    print(f"Saved {len(df)} samples -> {output_csv_file}")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        sys.exit(1)

    generate_tests(
        input_json_file="all_merged_articles.json",
        output_csv_file="ragas_testset.csv",
        sample_size=1000,
        test_size=250,
    )