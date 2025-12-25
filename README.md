# Real Estate AI Agent: Multimodal Property Matching

**A RAG-based Generative AI system for personalized property recommendations using LLMs and CLIP**

This project implements a cutting-edge AI agent designed to transform the real estate search experience. [cite_start]By combining a **Retrieval-Augmented Generation (RAG)** architecture with **CLIP-based multimodal search**, the system bridges the gap between complex buyer preferences and property listings, delivering tailored descriptions and advanced visual-semantic matching. [cite: 16, 22]

## Key Features

- [cite_start]**Multimodal Search Engine**: Integrated CLIP (ViT-B/32) model for searching properties based on both text descriptions and visual similarity. [cite: 16, 22]
- [cite_start]**RAG-Powered Personalization**: Dynamic generation of property descriptions tailored to specific user "Personas" (e.g., Young Professional, Luxury Seeker). [cite: 17, 21]
- [cite_start]**Semantic Vector Search**: Property retrieval using high-dimensional embeddings (text-embedding-ada-002) for accuracy superior to traditional keyword filters. 
- [cite_start]**Intelligent Persona Detection**: Automatic classification of user preferences into actionable buyer profiles through natural language analysis. [cite: 20]
- [cite_start]**Vector Database Integration**: Utilizes ChromaDB for efficient storage and querying of structured and unstructured property data. 

## Project Overview

[cite_start]This system demonstrates the practical application of Gen-AI in the real estate sector, focusing on the synergy between unstructured data and user intent. 

- [cite_start]**LLM Model**: GPT-3.5-Turbo for preference extraction and personalized response generation. [cite: 16, 21]
- [cite_start]**Embeddings**: OpenAI models for semantic property mapping. 
- [cite_start]**Multimodal Alignment**: CLIP to unify image and text search spaces, enabling searches based on "visual style." 
- [cite_start]**Dataset**: 20 detailed property listings in Rome with structured metadata and narrative descriptions. [cite: 18]

## Repository Architecture



```text
real-estate-ai-agent/
├── HomeMatch.py                # Core application logic and RAG engine
├── Personalized Real Estate Agent.ipynb  # Full development and analysis pipeline
├── listings.json               # Structured property dataset (JSON format)
├── listings.txt                # Textual real estate listings
├── requirements.txt            # Project dependencies (LangChain, OpenAI, ChromaDB)
└── LICENSE                     # GNU GPLv3 License
```

## Core Components

### 1. Retrieval-Augmented Generation (RAG)
The system doesn't just find a house; it interprets **why** a property is a match. By retrieving relevant listings from a vector store, the LLM augments the data with the buyer's specific context (e.g., "perfect for a family who loves natural light").

### 2. Multimodal Matching with CLIP
Traditional filters often fail with subjective or visual queries like "modern architectural style". Our CLIP integration overcomes this by enabling:
- **Text-to-Image Search**: Finding properties that "look" like the user's description.
- **Visual-Semantic Alignment**: Ensuring the aesthetic feel of the property images matches the textual requirements.

### 3. Persona Classification
The agent analyzes free-text inputs to intelligently categorize users into specific profiles, such as "First-time Buyer," "Luxury Investor," or "Nature Lover". This allow the system to:
- **Tailor Recommendations**: Adjust the tone and priority of search results based on the detected profile.
- **Optimize Relevance**: Align the output with individual expectations and lifestyle priorities.


## Quick Start

### Installation

```bash
# Clone the repository
git clone [https://github.com/StefanoBlando/real-estate-ai-agent.git](https://github.com/StefanoBlando/real-estate-ai-agent.git) [cite: 16]

# Navigate to the project directory
cd real-estate-ai-agent

# Install dependencies
pip install -r requirements.txt [cite: 121]

```
## Usage Example

The `HomeMatchApp` class integrates the persona detector, vector database, and personalization engine into a single workflow:

```python
from HomeMatch import HomeMatchApp

# Initialize the integrated system
app = HomeMatchApp()

# Example: High-intent buyer preference
user_query = "I'm looking for a luxury penthouse in Monti with historic charm and a view of the Colosseum."

# Search and generate RAG-based personalized descriptions
response = app.search_and_personalize(user_query, n_results=3)

# View the curated results
app.display_results(response)

```

## Results & Performance

[cite_start]The HomeMatch system was evaluated through a series of rigorous test cases to measure the accuracy of the **RAG pipeline** and the effectiveness of the **CLIP multimodal search**[cite: 20, 22].

### 1. Persona Detection Accuracy
[cite_start]The system demonstrated a **100% success rate** in correctly identifying buyer personas from natural language inputs[cite: 20]. 
- [cite_start]**Young Professional**: Detected with 100% confidence for queries focusing on metro access and work-life balance[cite: 20, 21].
- [cite_start]**Growing Family**: Successfully identified with 91.4% confidence based on school and safety requirements[cite: 20].
- [cite_start]**Luxury Seeker**: Accurately matched with 70% confidence for premium and exclusive property requests[cite: 20].

### 2. Search & Retrieval Performance
[cite_start]By utilizing semantic vector search instead of traditional keyword filters, the system achieved superior matching results[cite: 19, 21].
- [cite_start]**Semantic Relevance**: The vector database successfully retrieved modern apartments and luxury penthouses based on intent, even when specific keywords were missing[cite: 19].
- [cite_start]**Multimodal Alignment**: The integration of CLIP allowed the system to align property images with visual style descriptions like "Industrial Chic" or "Bohemian"[cite: 22].

### 3. Personalization Impact
| Feature | Implementation | Outcome |
| :--- | :--- | :--- |
| **Tailored Descriptions** | RAG / GPT-3.5-Turbo | [cite_start]Generated unique, 1:1 descriptions for 10 different buyer types[cite: 18, 21]. |
| **Match Scoring** | Weighted Algorithm | [cite_start]Provided clear percentage-based compatibility scores for every recommendation[cite: 21]. |
| **Visual Search** | CLIP Multi-modal | [cite_start]Enabled "Search by style," allowing users to find homes that "look" like their dream property[cite: 22]. |

### 4. Reviewer Commendations
The system's technical capability was praised for:
- [cite_start]**Vector Database Proficiency**: Demonstrating strong skills in creating and querying vector databases for real estate listings.
- [cite_start]**LLM Integration**: Successfully connecting buyer preferences with augmented property descriptions using LLM chains.
- [cite_start]**Multimodal Innovation**: Effectively using CLIP to enhance property discoverability via both image and text similarity.



## Future Enhancements

Following the initial deployment and based on expert reviewer feedback, the following technical improvements are prioritized for the next development phase:

### 1. Advanced Search Optimization
* [cite_start]**Normalization & Weighting**: Implement a mathematical normalization layer to balance scores between text and image similarity. 
* [cite_start]**Ranking Control**: Introduce a weighting mechanism allowing users to prioritize visual style over textual features or vice versa.

### 2. Intelligent Persona Mapping
* [cite_start]**Zero-Shot Classification**: Transition from keyword-based matching to Zero-Shot Classification using models from OpenAI or HuggingFace.
* [cite_start]**Dynamic Personas**: Enable the system to classify user text into personas without pre-defined hardcoded rules, improving adaptability to diverse user inputs.

### 3. Performance & Maintainability
* [cite_start]**CLIP Embedding Caching**: Implement a caching layer for frequent CLIP embeddings to reduce latency and API costs during high-traffic periods.
* [cite_start]**Advanced Visualizations**: Include visualizations of how embeddings are stored and queried within the vector space to simplify future maintainability and debugging.

### 4. System Scaling
* **Database Expansion**: Move from the current 20 listings to a production-scale dataset while maintaining sub-second retrieval times.
* **Multi-Language Support**: Extend the RAG pipeline to support multilingual queries, allowing international buyers to search in their native languages.


## License

This project is licensed under the **GNU General Public License v3.0**. 

**Key provisions of GPLv3:**
* [cite_start]**Copyleft**: Any derivative works or modifications must also be licensed under GPLv3, ensuring the project remains open-source.
* [cite_start]**Commercial Use**: You are permitted to use this software for commercial purposes.
* [cite_start]**Distribution**: If you distribute the software, you must make the source code available to the recipients.
* [cite_start]**Patent Protection**: Includes an express grant of patent rights from contributors to users.

For more details, please see the [LICENSE](LICENSE) file in this repository.

## Acknowledgments

This project was developed as part of the **Generative AI Nanodegree** program. Special thanks to:

* [cite_start]**OpenAI**: For providing the `gpt-3.5-turbo` model used for RAG and `text-embedding-ada-002` for semantic search.
* [cite_start]**CLIP Contributors**: For the `ViT-B/32` architecture that enables true multimodal discovery through image and text similarity.
* **Udacity/Bertelsmann Reviewers**: For their insightful feedback, specifically regarding:
    * [cite_start]**Technical Capability**: For recognizing the proficiency in creating and querying vector databases for real estate.
    * [cite_start]**LLM Chains**: For validating the overall workflow and integration of LLMs for personalized listing generation.
    * [cite_start]**Innovation**: For highlighting the effective use of CLIP to enhance property discoverability.
* [cite_start]**The Open Source Community**: For the libraries `chromadb`, `torch`, and `langchain` which form the backbone of this application[cite: 121].

---
*This system successfully demonstrates how Large Language Models and Multimodal AI can be applied to create innovative solutions for the real estate market.*
