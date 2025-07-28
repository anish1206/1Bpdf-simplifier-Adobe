# Round 1B: Persona-Driven Document Intelligence

This project is a sophisticated system that acts as an intelligent document analyst. It processes a collection of PDF documents and, based on a specific user persona and task, extracts and ranks the most relevant sections from across the entire collection.

A detailed explanation of the methodology (including semantic chunking and relevance scoring) can be found in the `approach_explanation.md` file.

## Project Structure

The solution expects the following input structure:

```
input/
├── docs/
│   ├── document_1.pdf
│   └── document_2.pdf
│
└── request.json
```

-   The `/app/input/docs` directory should contain all the PDF documents (3-10 files).
-   The `/app/input/request.json` file should contain the user `persona` and `job_to_be_done`.

## How to Build and Run

The entire solution is containerized using Docker and is designed to run completely offline, meeting all hackathon constraints.

**1. Build the Docker Image**

Use the following command to build the image. This command will also download and package the necessary NLP model within the container.

```bash
docker build --platform linux/amd64 -t my-r1b-solution .
```

**2. Run the Solution**

After the image is built, use the following command to run the analysis. The container will automatically process the files in the `input` directory and generate a single `challenge1b_output.json` in the `output` directory.

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none my-r1b-solution
```
