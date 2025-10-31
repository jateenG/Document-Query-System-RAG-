# File: rag_pipeline.py

import os
import logging
from typing import Dict, Any
from data_ingestion import DocumentIngestionService
from embedding_service import EmbeddingService
from vector_store import VectorStoreManager
from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_db_path: str = "./chroma_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", llm_model: str = "llama2"):
        """Initialize complete RAG pipeline."""
        self.ingestion = DocumentIngestionService()
        self.embeddings = EmbeddingService(embedding_model)
        self.vector_store = VectorStoreManager(vector_db_path)
        self.llm = LLMService(model_type="ollama", model_name=llm_model)

    def index_documents(self, document_path: str, collection_name: str):
        """Load, chunk, embed, and store documents."""
        # Load documents
        logger.info("Loading documents...")
        if os.path.isdir(document_path):
            documents = self.ingestion.load_documents_from_directory(document_path)
        else:
            if document_path.endswith(".pdf"):
                documents = self.ingestion.load_pdf(document_path)
            else:
                documents = self.ingestion.load_text(document_path)

        # Chunk documents
        logger.info("Chunking documents...")
        [cite_start]chunks = self.ingestion.chunk_documents(documents) [cite: 27]

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)

        # Store in vector database
        logger.info("Storing embeddings in vector database...")
        self.vector_store.add_documents(chunks, embeddings, collection_name)
        logger.info("Indexing complete.")

    def query(self, query_text: str, collection_name: str, n_results: int = 5) -> Dict[str, Any]:
        """Execute RAG query: retrieve relevant documents and generate response."""
        # Ensure collection is loaded
        self.vector_store.create_collection(collection_name)
        
        # Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = self.embeddings.embed_text(query_text)

        # Search for similar documents
        logger.info("Searching for relevant documents...")
        retrieval_results = self.vector_store.search(query_embedding, n_results)

        # Prepare context from retrieved documents
        retrieved_docs = retrieval_results['documents'][0]
        context = "\n\n".join(retrieved_docs)

        # Generate response using LLM
        logger.info("Generating response from LLM...")
        response = self.llm.generate_response(query_text, context)

        return {
            "query": query_text,
            "response": response,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }