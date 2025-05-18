"""vector_tools.py – FAISS-based vector storage and retrieval utilities

Features:
- Graceful file handling: Appends to existing index or creates new one
- Reset functionality: Clear the index when needed
- Informative logging: Detailed status messages at INFO level
- Configurable similarity thresholds: Control recall precision
"""

from __future__ import annotations
import os
import shutil
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ---------- Global Configuration ---------- #
EMBEDDINGS = HuggingFaceEmbeddings()
INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")


def _ensure_dir() -> str:
    """Ensure the index directory exists and return its path."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    return INDEX_DIR


def _load_or_init(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> FAISS:
    """Load existing index or initialize a new one if none exists.
    
    Args:
        texts: List of text documents to index
        metadatas: Optional list of metadata dictionaries
        
    Returns:
        FAISS vector store instance
    """
    _ensure_dir()
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    
    if os.path.exists(index_path):
        logger.info("🔄 Loading existing FAISS index...")
        db = FAISS.load_local(INDEX_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
        if texts:
            db.add_texts(texts, metadatas or [{} for _ in texts])
    else:
        logger.info("✨ Creating new FAISS index...")
        db = FAISS.from_texts(
            texts=texts,
            embedding=EMBEDDINGS,
            metadatas=metadatas or [{} for _ in texts]
        )
    return db


# ---------- Public API ---------- #

def reset_index() -> str:
    """Delete the on-disk FAISS index.
    
    Returns:
        Status message indicating success or failure
    """
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        return "🗑️  Index reset – directory removed."
    return "ℹ️  No index to reset."


def store_to_vectordb(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Store documents in the FAISS index, creating it if necessary.
    
    Args:
        texts: List of text documents to store
        metadatas: Optional list of metadata dictionaries
        
    Returns:
        Status message
    """
    if not texts:
        return "⚠️  No texts supplied."
    if metadatas and len(metadatas) != len(texts):
        return "❌ Texts and metadatas length mismatch."
        
    try:
        db = _load_or_init(texts, metadatas)
        db.save_local(INDEX_DIR)
        return f"✅ Stored {len(texts)} docs to '{INDEX_DIR}'."
    except Exception as e:
        logger.exception(f"Failed to store documents: {e}")
        return f"❌ Error storing documents: {str(e)}"


def vector_query(query: str, k: int = 4) -> str:
    """Search the index and return top-k matching documents.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        Formatted string of matching documents or error message
    """
    try:
        if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            return "❌ Index not found. Run store_to_vectordb first."
            
        db = FAISS.load_local(INDEX_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in docs) or "(no hits)"
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        return f"❌ Search error: {str(e)}"


def retrieve_answer(query: str, threshold: float = 0.25, k: int = 4) -> str:
    """Retrieve the best matching answer from the index.
    
    Args:
        query: Search query
        threshold: Maximum distance threshold for a match (lower is more similar)
        k: Number of results to consider
        
    Returns:
        The best matching answer or an error message
    """
    try:
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        if not os.path.exists(index_path):
            return "❌ Index not found. Run store_to_vectordb first."
            
        db = FAISS.load_local(INDEX_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
        results: List[Tuple[Any, float]] = db.similarity_search_with_score(query, k=k)

        best = None
        for doc, dist in results:
            if dist <= threshold and (ans := doc.metadata.get("answer")):
                best = (ans, dist)
                break  # Return first good match
                
        if best:
            ans, dist = best
            logger.info(f"🧠 Memory hit (distance={dist:.2f}) → {ans}")
            return ans
            
        return "❌ No confident answer found."
        
    except Exception as e:
        logger.exception(f"Failed to retrieve answer: {e}")
        return f"❌ Error retrieving answer: {str(e)}"
