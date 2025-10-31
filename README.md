# Production-Ready RAG System with Docker

This project provides a complete, containerized Retrieval-Augmented Generation (RAG) system. It uses local models for embeddings and language generation, ensuring privacy and cost-effectiveness. The system is served via a Flask REST API and orchestrated with Docker Compose.



## Features 

- **Document Ingestion**: Supports PDF and TXT files.
- **Local AI**: Uses `sentence-transformers` for embeddings and `Ollama` for the LLM, keeping all data local.
- **Vector Storage**: Persists document embeddings in a local `ChromaDB` instance.
- **REST API**: Simple and clean API endpoints for document indexing and querying.
- **Containerized**: Fully configured with Docker for easy, one-command deployment.

## Project Structure

```
.
├── app.py                  # Flask API endpoints
├── data_ingestion.py       # Document loading and chunking
├── embedding_service.py    # Embedding generation
├── llm_service.py          # LLM interaction (Ollama)
├── rag_pipeline.py         # Main RAG orchestration logic
├── vector_store.py         # Vector database management (Chroma)
├── documents/              # Place your PDF/TXT files here
├── chroma_db/              # Persistent vector store (created automatically)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition for the API
└── docker-compose.yml      # Docker multi-container setup
```

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Deployment Steps

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd production-rag-system
```

**2. Add Your Documents**

Create a `documents` folder in the project root and place the `.pdf` or `.txt` files you want the system to learn from inside it.

```bash
mkdir documents
# Copy your files into the 'documents' folder
```

**3. Pull the Ollama LLM Model**

Before starting the main services, you need to pull the LLM model that Ollama will use. This command starts the Ollama service temporarily to download the model.

```bash
docker-compose run --rm ollama ollama pull llama2
```
*You can replace `llama2` with another model like `mistral` or `gemma:2b`, but remember to update it in `docker-compose.yml` as well.*

**4. Build and Run the Services**

Use Docker Compose to build the API image and start both the RAG API and Ollama services in the background.

```bash
docker-compose up --build -d
```

**5. Verify Services are Running**

Check the status of the containers. Both should be `Up` or `running`.
```bash
docker-compose ps
```
You can view the logs for troubleshooting:
```bash
docker-compose logs -f rag-api
```

## API Usage

### 1. Index Your Documents

Send a `POST` request to the `/index` endpoint. The `document_path` must be the path *inside the container*, which is `/app/documents`. The `collection_name` is how you'll identify this set of documents later.

```bash
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{
        "document_path": "/app/documents",
        "collection_name": "my_project_docs"
      }'
```
Wait for the success response. This can take time depending on the size of your documents.

### 2. Query Your Documents

Once indexing is complete, send a `POST` request to the `/query` endpoint with your question.

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the main purpose of this project?",
        "collection_name": "my_project_docs",
        "n_results": 3
      }'
```

The API will return a JSON response containing the generated answer and the source text chunks it used as context.

### 3. Stop the System

To stop and remove all containers, networks, and volumes, run:
```bash
docker-compose down -v
```

## Customization

- **Embedding Model**: Change the model in `embedding_service.py` and `docker-compose.yml`. See the `sentence-transformers` library for options.
- **LLM Model**: Change the `LLM_MODEL` environment variable in `docker-compose.yml` and make sure to pull the model with the `ollama pull` command.
- **Chunking Strategy**: Modify the `chunk_size` and `chunk_overlap` parameters in `data_ingestion.py`.