"""RAG system combining local documents and web search"""
from typing import List, Optional
import os
from .web_search import WebSearch
from .document_loader import DocumentLoader


class RAGSystem:
    """Retrieval Augmented Generation system"""

    def __init__(self, data_dir: str = "data"):
        self.web_search = WebSearch()
        self.document_loader = DocumentLoader()
        self.data_dir = data_dir

        # Load local documents if available
        self.local_chunks = []
        docs_dir = os.path.join(data_dir, "documents")
        if os.path.exists(docs_dir):
            self.local_chunks = self.document_loader.load_directory(docs_dir)

    def retrieve(self, query: str, use_web: bool = True, use_local: bool = True) -> str:
        """Retrieve relevant information for a query"""
        context_parts = []

        # Retrieve from local documents
        if use_local and self.local_chunks:
            local_context = self._retrieve_local(query)
            if local_context:
                context_parts.append(local_context)

        # Retrieve from web search
        if use_web:
            web_context = self._retrieve_web(query)
            if web_context:
                context_parts.append(web_context)

        if not context_parts:
            return ""

        return "\n\n---\n\n".join(context_parts)

    def _retrieve_local(self, query: str) -> Optional[str]:
        """Simple keyword-based retrieval from local documents"""
        if not self.local_chunks:
            return None

        query_words = set(query.lower().split())
        relevant_chunks = []

        for chunk in self.local_chunks:
            content_words = set(chunk['content'].lower().split())
            # Simple overlap scoring
            overlap = len(query_words & content_words)
            if overlap > 2:
                relevant_chunks.append((overlap, chunk['content']))

        if not relevant_chunks:
            return None

        # Sort by relevance and take top 3
        relevant_chunks.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk for _, chunk in relevant_chunks[:3]]

        return "Local documents:\n\n" + "\n\n".join(top_chunks)

    def _retrieve_web(self, query: str) -> Optional[str]:
        """Search the web and fetch content"""
        print(f"  Searching the web for: {query}")
        results = self.web_search.search_and_fetch(query, num_results=2)

        if not results:
            return None

        return "Web search results:\n\n" + "\n\n---\n\n".join(results)

    def build_prompt(self, query: str, context: str) -> str:
        """Build a prompt with context"""
        if context:
            return f"""Based on the following context, answer the user's question. If the context doesn't contain relevant information, you can use your general knowledge but prefer information from the context.

Context:
{context}

User question: {query}

Answer:"""

        return f"""User question: {query}

Answer:"""
