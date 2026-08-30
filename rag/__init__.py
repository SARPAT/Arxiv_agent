"""Retrieve/generate RAG pipeline for the Arxiv Agent.

This package loads the FAISS index built by ``ingestion/build_index.py``
and exposes a single-turn question-answering pipeline on top of it. See
``pipeline.py`` for the orchestration entry point.
"""
