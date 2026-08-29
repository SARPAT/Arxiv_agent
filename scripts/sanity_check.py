"""Throwaway script to verify the embedding and generation model dependencies
work before any RAG logic is written. Not part of the application."""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

TEST_QUERY = "What is retrieval-augmented generation?"


def check_embedding_model():
    print("Loading BAAI/bge-base-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embedding = model.encode(TEST_QUERY)
    print(f"Embedding dimension: {embedding.shape[0]}")


def check_generation_model():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found. Set it in a .env file.")

    print("Calling meta/llama-3.1-8b-instruct ...")
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=api_key)
    response = llm.invoke(TEST_QUERY)
    print(f"Response: {response.content}")


if __name__ == "__main__":
    check_embedding_model()
    check_generation_model()
