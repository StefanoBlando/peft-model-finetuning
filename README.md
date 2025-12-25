# Real Estate AI Agent: Multimodal Property Matching

**A RAG-based Generative AI system for personalized property recommendations using LLMs and CLIP**

This project implements a cutting-edge AI agent designed to transform the real estate search experience by combining **Retrieval-Augmented Generation (RAG)** with **CLIP-based multimodal search**.  
The system bridges the gap between complex buyer preferences and property listings, delivering tailored descriptions and advanced visual–semantic matching.[^1][^2]

---

## Key Features

- **Multimodal Search Engine**  
  Integrated CLIP (ViT-B/32) model for searching properties based on both text descriptions and visual similarity.[^2]

- **RAG-Powered Personalization**  
  Dynamic generation of property descriptions tailored to specific user personas (e.g., Young Professional, Luxury Seeker).[^3]

- **Semantic Vector Search**  
  Property retrieval using high-dimensional embeddings (`text-embedding-ada-002`) for accuracy superior to traditional keyword filters.[^4]

- **Intelligent Persona Detection**  
  Automatic classification of user preferences into actionable buyer profiles through natural language analysis.[^5]

- **Vector Database Integration**  
  Uses ChromaDB for efficient storage and querying of structured and unstructured property data.[^6]

---

## Project Overview

This system demonstrates a practical application of Generative AI in the real estate sector, focusing on the synergy between unstructured data and user intent.

- **LLM Model**: GPT-3.5-Turbo for preference extraction and personalized response generation.[^3]
- **Embeddings**: OpenAI embedding models for semantic property mapping.[^4]
- **Multimodal Alignment**: CLIP to unify image and text search spaces, enabling searches based on visual style.[^2]
- **Dataset**: 20 detailed property listings in Rome with structured metadata and narrative descriptions.[^7]

---

## Repository Architecture

```text
real-estate-ai-agent/
├── HomeMatch.py                        # Core application logic and RAG engine
├── Personalized Real Estate Agent.ipynb # Full development and analysis pipeline
├── listings.json                      # Structured property dataset (JSON format)
├── listings.txt                       # Textual real estate listings
├── requirements.txt                   # Project dependencies
└── LICENSE                             # GNU GPLv3 License
