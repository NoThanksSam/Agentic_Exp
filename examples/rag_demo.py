"""Example: RAG pipeline with document retrieval."""

import asyncio
from src.rag import get_rag_pipeline


async def main():
    """Example: Using RAG pipeline."""
    
    print("Initializing RAG Pipeline...")
    rag_pipeline = get_rag_pipeline()
    
    # Add documents
    documents = {
        "doc1": "Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data.",
        "doc2": "Deep Learning uses neural networks with multiple layers to process complex patterns.",
        "doc3": "Natural Language Processing helps computers understand human language.",
        "doc4": "Computer Vision enables machines to interpret and understand visual information.",
    }
    
    print("\nAdding documents to RAG...")
    for name, text in documents.items():
        rag_pipeline.add_text(text, metadata={"source": name})
    
    # Retrieve documents
    queries = [
        "Tell me about machine learning",
        "What is deep learning?",
        "How do computers understand images?",
    ]
    
    print("\n\nRetrieving documents for queries:")
    print("-" * 50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        retrieved = await rag_pipeline.retrieve(query, k=2)
        
        for i, doc in enumerate(retrieved, 1):
            print(f"\n  Result {i}:")
            print(f"    Content: {doc.page_content[:100]}...")
            print(f"    Source: {doc.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
