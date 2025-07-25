# Methodology for Persona-Driven Document Intelligence

Our solution is engineered as a multi-stage pipeline that moves beyond simple keyword matching to understand the true semantic intent behind a user's request. It excels at finding contextually relevant information within a large collection of documents by focusing on meaning, relevance, and targeted extraction, while strictly adhering to all offline and performance constraints.

### Stage 1: Dynamic Query Formulation

Instead of using a literal and often ambiguous concatenation of the persona and job description, our system first generates a richer, more descriptive query. By framing the request into a prompt like, *"Based on the needs of a [Persona Role], find information to accomplish: [Job Task],"* we guide the language model to focus on the underlying goal, leading to a more nuanced and accurate understanding of the user's intent.

### Stage 2: Semantic Chunking and Relevance Scoring

The core of our intelligence lies in a sophisticated chunking and ranking process. Recognizing that analyzing entire multi-page sections can dilute meaning, our system follows these steps:

1.  **Outline and Section Extraction:** We first use the robust outline extractor from Round 1A to identify all major sections in each document.
2.  **Semantic Chunking:** Each section's text is then intelligently broken down into smaller, more focused semantic chunks (i.e., paragraphs). This ensures that each unit of text being analyzed has a concentrated topic.
3.  **High-Precision Scoring:** Using a lightweight `all-MiniLM-L6-v2` Sentence Transformer model, we encode our smart query and every single chunk from all documents into numerical vectors. Cosine similarity is then calculated between the query vector and every chunk vector.
4.  **Section Ranking:** A section's final relevance score is determined by the score of its *single most relevant chunk*. This powerful technique allows us to pinpoint documents and sections that contain even a small, highly relevant piece of information that would have otherwise been lost.

### Stage 3: Granular Sub-Section Analysis

This chunking strategy provides an elegant and accurate solution to the Sub-Section Analysis requirement, which is worth 40% of the total score. The "refined text" we provide for our top-ranked sections is the *exact, highest-scoring paragraph* that caused that section to be ranked highly in the first place. This offers a precise, context-rich snippet that is directly correlated with the user's task.

### System Architecture and Compliance

The entire solution is designed for offline, resource-constrained environments. We chose the `all-MiniLM-L6-v2` model (~86MB) as it provides the optimal balance between high semantic accuracy and low computational overhead, ensuring we meet the strict CPU-only time constraints. Our Dockerfile copies the pre-downloaded model directly, guaranteeing a fully self-contained and network-independent execution.