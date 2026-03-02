"""Document loader for local files"""
import os
from typing import List, Optional
from pathlib import Path


class DocumentLoader:
    """Load and chunk documents from local files"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> List[dict]:
        """Load a single file and return chunks"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext == '.txt':
            return self._load_txt(path)
        elif ext == '.md':
            return self._load_md(path)
        elif ext == '.pdf':
            return self._load_pdf(path)
        elif ext == '.docx':
            return self._load_docx(path)
        else:
            # Try as text
            return self._load_txt(path)

    def load_directory(self, directory: str, extensions: Optional[List[str]] = None) -> List[dict]:
        """Load all files from a directory"""
        if extensions is None:
            extensions = ['.txt', '.md', '.pdf', '.docx']

        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        chunks = []
        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    file_chunks = self.load_file(str(file_path))
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return chunks

    def _load_txt(self, path: Path) -> List[dict]:
        """Load plain text file"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        return self._chunk_text(text, {'source': str(path), 'type': 'txt'})

    def _load_md(self, path: Path) -> List[dict]:
        """Load markdown file"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        return self._chunk_text(text, {'source': str(path), 'type': 'markdown'})

    def _load_pdf(self, path: Path) -> List[dict]:
        """Load PDF file"""
        try:
            import PyPDF2

            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

            return self._chunk_text(text, {'source': str(path), 'type': 'pdf'})
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []

    def _load_docx(self, path: Path) -> List[dict]:
        """Load DOCX file"""
        try:
            from docx import Document

            doc = Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])

            return self._chunk_text(text, {'source': str(path), 'type': 'docx'})
        except ImportError:
            print("python-docx not installed. Install with: pip install python-docx")
            return []

    def _chunk_text(self, text: str, metadata: dict) -> List[dict]:
        """Split text into chunks"""
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct != -1:
                        end = start + last_punct + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'metadata': metadata.copy()
                })

            start = end - self.chunk_overlap

        return chunks
