"""Chunking-Strategien
- Fixed Character (1000/150): Character-basiertes Slicing mit Overlap
- Sentence (8/1): Satz-basiert via NLTK (deutsch)
- Recursive Character (1000/150): LangChain RecursiveCharacterTextSplitter
- Semantic (Threshold 0.75): LangChain SemanticChunker auf Basis OpenAI-Embeddings
"""
import os
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# NLTK-Daten beim ersten Import sicherstellen (deutscher Satz-Tokenizer)
for resource in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def chunk_fixed(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Character-basiertes Fixed-Size-Chunking mit Overlap."""
    if not text:
        return []
    if len(text) <= size:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_sentence(text: str, max_sentences: int = 8, overlap_sentences: int = 1) -> list[str]:
    """Satz-basiertes Chunking via NLTK (deutscher Tokenizer)."""
    if not text:
        return []
    sentences = nltk.sent_tokenize(text, language="german")
    if not sentences:
        return []
    chunks = []
    step = max(1, max_sentences - overlap_sentences)
    i = 0
    while i < len(sentences):
        c = " ".join(sentences[i:i + max_sentences]).strip()
        if c:
            chunks.append(c)
        i += step
    return chunks


def chunk_recursive(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Recursive Character Chunking via LangChain mit hierarchischen Trennzeichen."""
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


_semantic_splitter = None


def _get_semantic_splitter(threshold: float = 0.75):
    """Lazy-loaded Semantic Chunker (nutzt text-embedding-3-small als Hilfsmodell)."""
    global _semantic_splitter
    if _semantic_splitter is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        _semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold * 100,
        )
    return _semantic_splitter


def chunk_semantic(text: str, threshold: float = 0.75) -> list[str]:
    """Semantic Chunking via LangChain SemanticChunker."""
    if not text:
        return []
    splitter = _get_semantic_splitter(threshold)
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk(text: str, strategy: dict) -> list[str]:
    """Dispatcher: wählt die richtige Chunking-Funktion gemäß Strategy-Dict."""
    t = strategy.get("type")
    if t == "fixed":
        return chunk_fixed(text, strategy.get("size", 1000), strategy.get("overlap", 150))
    if t == "sentence":
        return chunk_sentence(text, strategy.get("max_sentences", 8), strategy.get("overlap_sentences", 1))
    if t == "recursive":
        return chunk_recursive(text, strategy.get("size", 1000), strategy.get("overlap", 150))
    if t == "semantic":
        return chunk_semantic(text, strategy.get("threshold", 0.75))
    raise ValueError(f"Unbekannter Strategy-Typ: {t}")
