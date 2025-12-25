# Real Estate AI Agent: Multimodal Property Matching

**A Retrieval-Augmented Generation (RAG) system for personalized real estate recommendations using LLMs and CLIP**

This project implements an advanced AI-driven real estate recommendation system that combines **semantic search**, **Large Language Models (LLMs)**, and **multimodal retrieval** to match buyer preferences with property listings.

By leveraging a **vector database**, **conversational retrieval chains**, and **LLM-based personalization**, the system generates tailored property descriptions that align with individual buyer intents, lifestyles, and priorities.
## Key Capabilities

- **Semantic Search with Vector Embeddings**  
  Uses OpenAI embeddings and a vector database (ChromaDB) to retrieve property listings that align with buyer intent beyond keyword matching.

- **LLM-Based Personalization**  
  Generates personalized property descriptions using a Large Language Model, adapting tone and emphasis based on detected buyer preferences.

- **Conversational Retrieval Chain**  
  Integrates document loading, embeddings, chat models, and retrieval logic using LangChain to enable end-to-end semantic reasoning.

- **Multimodal Search with CLIP**  
  Enhances discoverability by aligning text and image embeddings, allowing users to search properties by visual style and semantic meaning.

- **Persona-Aware Recommendation Logic**  
  Matches free-text user inputs to buyer personas and uses this information to guide retrieval and description augmentation.
## System Architecture Overview

The system integrates multiple AI components into a single workflow:

1. **User Input**  
   Buyer preferences are provided as free-text queries.

2. **Semantic Retrieval**  
   The query is embedded and compared against stored property embeddings in a vector database.

3. **Persona Detection**  
   User intent is analyzed to infer a buyer persona (e.g., Luxury Seeker, Growing Family).

4. **Retrieval-Augmented Generation (RAG)**  
   Relevant listings are retrieved and passed to the LLM along with buyer context.

5. **Personalized Output Generation**  
   The LLM generates tailored property descriptions while preserving factual accuracy.
## Repository Structure

```text
real-estate-ai-agent/
├── HomeMatch.py                         # Core application logic and RAG workflow
├── Personalized Real Estate Agent.ipynb # Development, testing, and analysis notebook
├── listings.json                       # Structured property listings
├── listings.txt                        # Textual property descriptions
├── requirements.txt                    # Project dependencies
└── LICENSE                             # GNU GPLv3 License

```

---

## Semantic Search & Vector Database


## Semantic Search and Vector Database

The project uses a vector database to store embeddings of real estate listings, enabling semantic retrieval based on buyer intent rather than exact keyword matches.

Each property listing is embedded using an OpenAI embedding model and stored in ChromaDB.  
At query time, the buyer’s input is embedded and compared against stored vectors to retrieve the most semantically relevant listings.

This approach allows the system to:
- Match abstract preferences such as lifestyle or design
- Handle varied phrasing and synonyms
- Retrieve relevant listings even when keywords are absent

## Semantic Search Effectiveness

The effectiveness of semantic search was evaluated using multiple buyer queries with varying intents, such as lifestyle-focused, luxury-oriented, and family-oriented preferences.

### Example: Lifestyle-Oriented Buyer

**User Query**  
"I'm a young professional looking for a modern apartment near public transport in a vibrant neighborhood."

**Top Retrieved Listings**
- Renovated apartment near metro access
- Modern loft close to tram lines
- City-center apartment with open-plan design

These results demonstrate the system’s ability to infer user intent and retrieve relevant listings beyond traditional keyword-based filtering.

## LLM-Based Personalization

A Large Language Model (LLM) is used to generate personalized property descriptions based on both retrieved listing data and inferred buyer preferences.

The LLM augments listing descriptions by emphasizing aspects that are most relevant to the buyer, such as:
- Location and prestige for luxury-oriented users
- Space, safety, and amenities for families
- Connectivity and lifestyle for young professionals

All factual property attributes (e.g., size, location) are preserved to ensure accuracy.

## Personalization Workflow

1. Buyer preferences are extracted from free-text input.
2. Relevant listings are retrieved using semantic vector search.
3. Buyer persona information is inferred from the query.
4. Retrieved listings and persona context are passed to the LLM.
5. A structured prompt guides the LLM to generate a personalized description
   without altering factual listing details.
   
## Personalization Examples

### Same Listing, Different Buyers

**Property**  
Two-bedroom apartment in Monti, 85 sqm, historic building, city view.

**Luxury Seeker Output**  
This refined Monti residence combines historic elegance with breathtaking city views, offering an exclusive living experience in one of Rome’s most prestigious districts.

**Growing Family Output**  
Located in the heart of Monti, this spacious two-bedroom apartment provides a comfortable and secure environment with ample room for family living and easy access to essential services.

## Listing Augmentation: Before and After

**Original Description**  
Two-bedroom apartment in Monti, 85 sqm, historic building, city view.

**Augmented Description (Luxury Seeker)**  
This elegant two-bedroom apartment in Monti blends historic charm with stunning city views, making it ideal for buyers seeking prestige and timeless architectural appeal.

## Multimodal Search with CLIP

The system integrates CLIP (ViT-B/32) to align image and text embeddings, enabling multimodal property search.

This allows users to:
- Search properties based on visual style descriptions
- Discover listings through both text and image similarity
- Enhance retrieval quality for subjective or aesthetic queries

## Future Improvements

- **Score Normalization and Weighting**  
  Introduce normalization and adjustable weighting between text and image similarity scores.

- **Embedding Visualization**  
  Add visual diagrams illustrating how embeddings are stored and queried within the vector database.

- **CLIP Embedding Caching**  
  Cache frequent CLIP embeddings to reduce latency and improve performance.

- **Zero-Shot Persona Classification**  
  Replace keyword-based persona matching with zero-shot classification models.

## License

This project is licensed under the GNU General Public License v3.0.

---

## Acknowledgments

This project was developed as part of the **Generative AI Nanodegree** program.

Special thanks to:
- OpenAI for LLM and embedding models
- LangChain for conversational retrieval chains
- CLIP contributors for multimodal alignment research
- Udacity and Bertelsmann reviewers for valuable feedback



