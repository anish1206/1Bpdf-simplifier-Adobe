# Persona-Driven Document Intelligence Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/anish1206/1Bpdf-simplifier-Adobe)

A powerful, offline-first solution that moves beyond keyword search to find hyper-relevant information for any user persona across a vast collection of documents.

---

### Table of Contents
- [The Challenge: Beyond Keyword Search](#the-challenge-beyond-keyword-search)
- [Our Solution: A Multi-Stage Intelligence Pipeline](#our-solution-a-multi-stage-intelligence-pipeline)
  - [Stage 1: Foundational Parsing & Structuring](#stage-1-foundational-parsing--structuring)
  - [Stage 2: The Intelligence Core - Multi-Factor Semantic Ranking](#stage-2-the-intelligence-core---multi-factor-semantic-ranking)
  - [Stage 3: Granular Sub-section Analysis](#stage-3-granular-sub-section-analysis)
  - [Stage 4: Diverse & Ranked Output Generation](#stage-4-diverse--ranked-output-generation)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Execution Instructions](#execution-instructions)

### The Challenge: Beyond Keyword Search

Traditional search fails when a user's goal is nuanced or described using different terms than the source documents. For example, a "Compliance Officer" looking for "risk exposure" might miss a critical section titled "Internal Control Deficiencies." The challenge is to bridge this semantic gap—to deliver information that matches a user's *intent*, not just their words.

### Our Solution: A Multi-Stage Intelligence Pipeline

We engineered a sophisticated, multi-stage pipeline that understands the meaning behind both the user's request and the documents' content. Our system ingests a collection of PDFs and, given a specific **Persona** and **Job-to-be-Done**, pinpoints the exact sections across all documents most relevant to that user's task.

### How It Works: The Pipeline

#### Stage 1: Foundational Parsing & Structuring
The pipeline begins with a highly robust document ingestion process. We use a three-stage fallback system (ToC extraction, then font-based analysis, then pattern matching) to create a structured outline for every PDF, ensuring stability even with poorly formatted documents. This converts unstructured text into machine-readable sections mapped to headings and pages.

#### Stage 2: The Intelligence Core - Multi-Factor Semantic Ranking
This is where the magic happens. We go beyond standard similarity by using an enhanced, multi-factor relevance score. This proprietary algorithm provides a holistic measure of relevance:

1.  **Semantic Similarity (40% weight):** The user's Persona and Job-to-be-Done are encoded into a query vector using the `all-MiniLM-L6-v2` Sentence Transformer. We calculate the cosine similarity between this query and the text of every section, understanding contextual alignment.
2.  **Persona Alignment (25% weight):** We analyze the section text for keywords specific to the user's professional role (e.g., 'methodology' for a Researcher, 'trend' for an Analyst), creating a persona-relevance score.
3.  **Task Relevance (25% weight):** Key terms are extracted from the job description and matched against the section content, directly measuring how well it addresses the task at hand.
4.  **Structural Importance (10% weight):** Sections appearing earlier in a document or with titles like "Introduction" or "Conclusion" are given a slight boost, recognizing their contextual importance.

#### Stage 3: Granular Sub-section Analysis
To address the need for deep insights (worth 40 points), our system performs a targeted analysis on the top-ranked sections. Each section is broken down into its constituent paragraphs (sub-sections), which are individually scored for relevance. The most pertinent paragraphs are then summarized using an extractive technique to generate a "Refined Text" output, providing key sentences and insights directly.

#### Stage 4: Diverse & Ranked Output Generation
Finally, all relevant sections are aggregated and sorted by their final weighted score. To ensure a broad and useful overview, we apply a **diversity filter**, limiting the number of top-ranked sections from any single document. This prevents one large, relevant document from drowning out key insights from other sources. The final, ranked, and diversified list is then structured into the required JSON format.

### Key Features

-   **Multi-Factor Relevance:** Proprietary scoring model delivers hyper-accurate, context-aware results.
-   **Granular Sub-section Analysis:** Provides deep, paragraph-level insights and extractive summaries.
-   **Result Diversification:** Ensures a balanced overview from all documents in the collection.
-   **Robust & Resilient:** Three-stage PDF parsing and graceful error handling ensure high uptime.
-   **Lightweight & Fast:** Uses a highly optimized 86MB NLP model for speed.
-   **Fully Offline & Self-Contained:** Built with a multi-stage Dockerfile to run without any internet connection, meeting all security and resource constraints.

### Technology Stack

-   **Language:** Python
-   **NLP / ML:** `sentence-transformers`, `torch`
-   **PDF Processing:** `PyMuPDF`
-   **Containerization:** Docker

### Execution Instructions

The project is containerized with Docker and designed to run exactly as specified in the hackathon requirements.

1.  **Input Structure:**
    Before running, ensure your input directory is structured as follows:
    ```
    input/
    ├── docs/
    │   ├── doc1.pdf
    │   └── doc2.pdf
    └── request.json
    ```

2.  **Build the Docker Image:**
    This command builds the image and downloads the model. An internet connection is required for this step only.
    ```bash
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    ```

3.  **Run the Analysis:**
    This command runs the container entirely offline. It mounts the `input` and `output` directories.
    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
    ```
    The final ranked `challenge1b_output.json` file will be generated in your local `output` folder.