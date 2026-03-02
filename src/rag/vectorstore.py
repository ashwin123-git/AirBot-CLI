"""Simple in-memory vector store using TF-IDF for RAG"""
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class VectorStore:
    """Simple in-memory vector store using TF-IDF"""

    def __init__(self, persist_directory: str = "./data"):
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.matrix = None

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        """Add documents to the vector store"""
        if not documents:
            return

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.documents)
            ids = [f"doc_{i}" for i in range(start_id, start_id + len(documents))]

        # Add documents
        self.documents.extend(documents)
        self.ids.extend(ids)

        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))

        # Rebuild vector matrix
        self._rebuild_matrix()

    def _rebuild_matrix(self):
        """Rebuild the TF-IDF matrix"""
        if self.documents:
            self.matrix = self.vectorizer.fit_transform(self.documents)

    def similarity_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents"""
        if not self.documents or self.matrix is None:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        # Format results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                    'distance': float(similarities[idx])
                })

        return results

    def delete_all(self):
        """Delete all documents"""
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.matrix = None

    def count(self) -> int:
        """Get the number of documents"""
        return len(self.documents)
