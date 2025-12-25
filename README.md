# Real Estate AI Agent: Multimodal Property Matching

**A RAG-based Generative AI system for personalized property recommendations using LLMs and CLIP**

This project implements a cutting-edge AI agent designed to transform the real estate search experience. [cite_start]By combining a **Retrieval-Augmented Generation (RAG)** architecture with **CLIP-based multimodal search**, the system bridges the gap between complex buyer preferences and property listings, delivering tailored descriptions and advanced visual-semantic matching[cite: 16, 22].

## Key Features

- [cite_start]**Multimodal Search Engine**: Integrated CLIP (ViT-B/32) model for searching properties based on both text descriptions and visual similarity[cite: 16, 22].
- [cite_start]**RAG-Powered Personalization**: Dynamic generation of property descriptions tailored to specific user "Personas" (e.g., Young Professional, Luxury Seeker)[cite: 17, 21].
- [cite_start]**Semantic Vector Search**: Property retrieval using high-dimensional embeddings (text-embedding-ada-002) for accuracy superior to traditional keyword filters[cite: 16, 19].
- [cite_start]**Intelligent Persona Detection**: Automatic classification of user preferences into actionable buyer profiles through natural language analysis.
- [cite_start]**Vector Database Integration**: Utilizes ChromaDB for efficient storage and querying of structured and unstructured property data[cite: 16, 19].

## Project Overview

This system demonstrates the practical application of Gen-AI in the real estate sector, focusing on the synergy between unstructured data and user intent.

- [cite_start]**LLM Model**: GPT-3.5-Turbo for preference extraction and personalized response generation[cite: 16, 21].
- [cite_start]**Embeddings**: OpenAI models (text-embedding-ada-002) for semantic property mapping[cite: 16, 19].
- [cite_start]**Multimodal Alignment**: CLIP to unify image and text search spaces, enabling searches based on "visual style"[cite: 16, 22].
- [cite_start]**Dataset**: 20 detailed property listings in Rome with structured metadata and narrative descriptions[cite: 18, 23].

## Repository Architecture

```text
real-estate-ai-agent/
├── HomeMatch.py                # Core application logic and RAG engine
├── Personalized Real Estate Agent.ipynb  # Full development and analysis pipeline
├── listings.json               # Structured property dataset (JSON format)
├── listings.txt                # Textual real estate listings
├── requirements.txt            # Project dependencies (LangChain, OpenAI, ChromaDB)
└── LICENSE                     # GNU GPLv3 License
