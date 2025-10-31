import os
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngestionService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        [cite_start]Initialize document ingestion service. [cite: 2]
        Args:
            [cite_start]chunk_size: Size of each text chunk in tokens [cite: 3]
            [cite_start]chunk_overlap: Number of overlapping tokens between chunks [cite: 3]
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and extract text from PDF file."""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Loaded PDF with {len(documents)} pages from {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []

    def load_text(self, text_path: str) -> List[Document]:
        """Load text file."""
        try:
            loader = TextLoader(text_path)
            [cite_start]documents = loader.load() [cite: 6]
            logger.info(f"Loaded text file from {text_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {text_path}: {e}")
            return []

    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Recursively load all PDF and TXT files from directory."""
        documents = []
        path = Path(directory)

        for pdf_file in path.glob("**/*.pdf"):
            documents.extend(self.load_pdf(str(pdf_file)))

        for txt_file in path.glob("**/*.txt"):
            documents.extend(self.load_text(str(txt_file)))

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with overlap."""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks