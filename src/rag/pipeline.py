"""RAG (Retrieval-Augmented Generation) pipeline."""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from loguru import logger
from config.settings import settings


class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete documents from the vector store."""
        pass


class ChromaDBProvider(VectorStoreProvider):
    """ChromaDB vector store provider."""
    
    def __init__(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            chroma_settings = ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=settings.rag.chromadb_path,
                anonymized_telemetry=False,
            )
            self.client = chromadb.Client(chroma_settings)
            self.collection = self.client.get_or_create_collection(
                name="agentic_framework",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized")
        except ImportError:
            logger.error("chromadb not installed")
            raise
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        for i, doc in enumerate(documents):
            self.collection.add(
                ids=[f"doc_{i}"],
                documents=[doc.page_content],
                metadatas=[doc.metadata or {}]
            )
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    async def search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Search ChromaDB for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        docs_with_scores = []
        if results and results["documents"]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # Convert distance to similarity score
                similarity = 1 - (distance / 2)
                document = Document(page_content=doc, metadata=metadata)
                docs_with_scores.append((document, similarity))
        
        return docs_with_scores
    
    async def delete(self, ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        self.collection.delete(ids=ids)


class PineconeProvider(VectorStoreProvider):
    """Pinecone vector store provider."""
    
    def __init__(self):
        try:
            from pinecone import Pinecone
            
            if not settings.rag.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not set")
            
            self.pc = Pinecone(api_key=settings.rag.pinecone_api_key)
            self.index = self.pc.Index(settings.rag.pinecone_index)
            logger.info("Pinecone initialized")
        except ImportError:
            logger.error("pinecone-client not installed")
            raise
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Pinecone."""
        from langchain.embeddings.openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model=settings.rag.embedding_model)
        
        vectors = []
        for i, doc in enumerate(documents):
            embedding = await embeddings.aembed_query(doc.page_content)
            vectors.append({
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {
                    "text": doc.page_content[:500],  # Store truncated text
                    **(doc.metadata or {})
                }
            })
        
        self.index.upsert(vectors=vectors)
        logger.info(f"Added {len(documents)} documents to Pinecone")
    
    async def search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Search Pinecone for similar documents."""
        from langchain.embeddings.openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(model=settings.rag.embedding_model)
        query_embedding = await embeddings.aembed_query(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        docs_with_scores = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            document = Document(page_content=text, metadata=metadata)
            docs_with_scores.append((document, match["score"]))
        
        return docs_with_scores
    
    async def delete(self, ids: List[str]) -> None:
        """Delete documents from Pinecone."""
        self.index.delete(ids=ids)


class RAGPipeline:
    """Main RAG pipeline for document ingestion and retrieval."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        if settings.rag.vector_store == "chromadb":
            self.vector_store = ChromaDBProvider()
        elif settings.rag.vector_store == "pinecone":
            self.vector_store = PineconeProvider()
        else:
            raise ValueError(f"Unsupported vector store: {settings.rag.vector_store}")
        
        self.documents: Dict[str, Document] = {}
        logger.info(f"RAG Pipeline initialized with {settings.rag.vector_store}")
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add text to the RAG pipeline."""
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
        
        for i, doc in enumerate(documents):
            doc_id = f"text_{len(self.documents)}_{i}"
            self.documents[doc_id] = doc
        
        # Add to vector store
        import asyncio
        asyncio.run(self.vector_store.add_documents(documents))
        logger.info(f"Added {len(documents)} chunks from text")
    
    def add_documents_from_directory(self, directory: str, pattern: str = "**/*.pdf") -> None:
        """Add documents from a directory."""
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory not found: {directory}")
            return
        
        documents = []
        for file_path in path.glob(pattern):
            try:
                if file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                self.documents[f"file_{i}"] = chunk
            
            import asyncio
            asyncio.run(self.vector_store.add_documents(chunks))
            logger.info(f"Added {len(chunks)} chunks from directory")
    
    async def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve documents relevant to a query."""
        if k is None:
            k = settings.rag.max_retrieved_docs
        
        results = await self.vector_store.search(query, k=k)
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc, score in results
            if score >= settings.rag.similarity_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query: {query[:50]}")
        return filtered_results
    
    def get_context_from_documents(self, documents: List[Document]) -> str:
        """Generate context string from documents."""
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in documents
        ])
        return context


# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
