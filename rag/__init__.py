"""Retrieve/generate RAG pipeline for the Arxiv Agent.

This package queries the Qdrant Cloud collection populated by
``ingestion/build_index.py`` and exposes a single-turn question-answering
pipeline on top of it. See ``pipeline.py`` for the orchestration entry
point and ``vectorstore.py`` for all vector-store access.
"""
