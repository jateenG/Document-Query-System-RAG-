# File: app.py

import os
import logging
from flask import Flask, request, jsonify
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize RAG pipeline
rag = RAGPipeline(
    vector_db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    llm_model=os.getenv("LLM_MODEL", "llama2")
)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/index", methods=["POST"])
def index_documents():
    """Index documents from provided path."""
    try:
        data = request.json
        document_path = data.get("document_path")
        collection_name = data.get("collection_name", "documents")

        if not document_path:
            return jsonify({"error": "document_path is required"}), 400

        if not os.path.exists(document_path):
            return jsonify({"error": f"Path not found: {document_path}"}), 404

        rag.index_documents(document_path, collection_name)
        return jsonify({
            "status": "success",
            "message": f"Documents indexed successfully from {document_path} into collection '{collection_name}'"
        }), 200
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    """Query the RAG system."""
    try:
        data = request.json
        query_text = data.get("query")
        collection_name = data.get("collection_name", "documents")
        n_results = data.get("n_results", 5)

        if not query_text:
            return jsonify({"error": "query is required"}), 400

        result = rag.query(query_text, collection_name, n_results)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)