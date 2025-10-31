# File: vector_store.py

import logging
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        [cite_start]Initialize Chroma vector database. [cite: 11]
        Args:
            [cite_start]persist_directory: Path where vector embeddings are stored [cite: 12]
        """
        self.persist_directory = persist_directory
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = None

    def create_collection(self, collection_name: str):
        """Create or get a collection in the vector database."""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created/accessed collection: {collection_name}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray, collection_name: str):
        """Add documents and their embeddings to the vector store."""
        if self.collection is None or self.collection.name != collection_name:
            self.create_collection(collection_name)

        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(documents)} documents to vector store")
        self.client.persist()

    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents."""
        if self.collection is None:
            raise ValueError("Collection not initialized. [cite_start]Call create_collection first.") [cite: 16, 17]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results