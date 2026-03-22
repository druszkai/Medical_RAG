import json
import random

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

from src.config import *

# Two personas - kept after previous model caused issues with more
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

    llm = LangchainLLMWrapper(ChatGroq(
        model=LLM_OPENAI_GTP_OSS_20B,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=4092,
        model_kwargs={"response_format": {"type": "json_object"}},
    ))

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MULTILINGUAL_EMBEDDING)
    )

    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=60)

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, persona_list=PERSONAS)
    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        run_config=run_config,
    )

    df = dataset.to_pandas()
    df.to_csv(PROCESSED_DATA_DIR / output_csv_file, index=False, encoding="utf-8")
    print(df.head())
    print(f"Saved {len(df)} samples → {output_csv_file}")


if __name__ == "__main__":
    generate_tests(
        input_json_file="all_merged_articles.json",
        output_csv_file="ragas_testset.csv",
        sample_size=150,
        test_size=50,
    )