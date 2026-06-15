"""Embedding-Adapter

- 3Small: OpenAI text-embedding-3-small (1536 Dimensionen)
- 3Large: OpenAI text-embedding-3-large (3072 Dimensionen)
- SBERT: paraphrase-multilingual-mpnet-base-v2 (768 Dimensionen, lokal)
- E5Large: intfloat/multilingual-e5-large (1024 Dimensionen, lokal, mit Prefix)

"""
import os
from openai import OpenAI

OPENAI_MODELS = {"3Small", "3Large"}
LOCAL_MODELS = {"SBERT", "E5Large"}

_openai_client = None
_st_models = {}


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _sentence_transformer(model_name: str):
    """Lazy-loaded Sentence-Transformer-Modell."""
    global _st_models
    if model_name not in _st_models:
        from sentence_transformers import SentenceTransformer
        _st_models[model_name] = SentenceTransformer(model_name)
    return _st_models[model_name]


def embed(model_key: str, texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Erzeugt Embeddings für eine Liste von Texten.

    Args:
        model_key: einer aus {"3Small", "3Large", "SBERT", "E5Large"}
        texts: Liste der zu vektorisierenden Texte
        is_query: True bei Suchanfrage (nur für E5-Modelle relevant)
    """
    if not texts:
        return []

    if model_key == "3Small":
        resp = _openai().embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]

    if model_key == "3Large":
        resp = _openai().embeddings.create(model="text-embedding-3-large", input=texts)
        return [d.embedding for d in resp.data]

    if model_key == "SBERT":
        model = _sentence_transformer("paraphrase-multilingual-mpnet-base-v2")
        return model.encode(texts, normalize_embeddings=True).tolist()

    if model_key == "E5Large":
        model = _sentence_transformer("intfloat/multilingual-e5-large")
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        return model.encode(prefixed, normalize_embeddings=True).tolist()

    raise ValueError(f"Unbekanntes Embedding-Modell: {model_key}")
