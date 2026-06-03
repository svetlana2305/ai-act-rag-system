import logging
import os
import uuid
from urllib.parse import urlparse

import tiktoken
import weaviate
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty

logger = logging.getLogger(__name__)

# Drei Embedding-Modelle im Vergleich
MODELS = {
    "3Small": "text-embedding-3-small",
    "3Large": "text-embedding-3-large",
    "Ada":    "text-embedding-ada-002",
}

# Drei feste Chunking-Strategien — bewusst nicht frei konfigurierbar,
# damit die Collections für die Seminararbeit vergleichbar bleiben.
# max_tokens=None bedeutet: ganzes Dokument als ein Chunk.
STRATEGIES = {
    "Artikel": {
        "label":      "Artikel-weise (ganzes Dokument als ein Chunk)",
        "max_tokens": None,
        "overlap":    0,
    },
    "500": {
        "label":      "500 Tokens / 50 Overlap — feinkörnig",
        "max_tokens": 500,
        "overlap":    50,
    },
    "2000": {
        "label":      "2000 Tokens / 200 Overlap — grob",
        "max_tokens": 2000,
        "overlap":    200,
    },
}

# cl100k_base gilt für alle drei Modelle
_enc = tiktoken.get_encoding("cl100k_base")


def collection_prefix(model_key: str, strategy_key: str) -> str:
    return f"AIAct_{model_key}_{strategy_key}"


def get_client():
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    secure = url.startswith("https")

    headers = {}
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        headers["X-OpenAI-Api-Key"] = key

    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        headers=headers,
    )


def setup_collection(client, model_key: str, strategy_key: str):
    model = MODELS[model_key]
    prefix = collection_prefix(model_key, strategy_key)
    art = f"{prefix}_Article"
    chunk = f"{prefix}_Chunk"

    if not client.collections.exists(art):
        client.collections.create(
            name=art,
            properties=[
                Property(name="article_number", data_type=DataType.INT,  skip_vectorization=True),
                Property(name="source_type",    data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="title",          data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="full_text",      data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_openai(model=model),
            generative_config=Configure.Generative.openai(),
        )
        logger.info("Collection %s erstellt.", art)

    if not client.collections.exists(chunk):
        client.collections.create(
            name=chunk,
            properties=[
                Property(name="content",        data_type=DataType.TEXT),
                Property(name="chunk_index",    data_type=DataType.INT,  skip_vectorization=True),
                Property(name="chunk_total",    data_type=DataType.INT,  skip_vectorization=True),
                # Artikel-Metadaten direkt im Chunk speichern — erleichtert spätere Abfragen
                Property(name="article_number", data_type=DataType.INT,  skip_vectorization=True),
                Property(name="title",          data_type=DataType.TEXT, skip_vectorization=True),
            ],
            references=[
                ReferenceProperty(name="ofArticle", target_collection=art)
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_openai(model=model),
            generative_config=Configure.Generative.openai(),
        )
        logger.info("Collection %s erstellt.", chunk)


def delete_collection(client, model_key: str, strategy_key: str):
    prefix = collection_prefix(model_key, strategy_key)
    for suffix in ("_Chunk", "_Article"):
        name = f"{prefix}{suffix}"
        if client.collections.exists(name):
            client.collections.delete(name)
            logger.info("Collection %s gelöscht.", name)


def import_articles(client, documents: list, model_key: str, strategy_key: str) -> dict:
    strategy = STRATEGIES[strategy_key]
    max_tokens = strategy["max_tokens"]
    overlap = strategy["overlap"]

    prefix = collection_prefix(model_key, strategy_key)
    art_col = client.collections.get(f"{prefix}_Article")
    chunk_col = client.collections.get(f"{prefix}_Chunk")

    doc_count = 0
    chunk_count = 0
    pairs = []

    with art_col.batch.dynamic() as batch:
        for doc in documents:
            art_uuid = str(uuid.uuid4())
            batch.add_object(
                properties={
                    "article_number": int(doc.get("article_number", 0)),
                    "source_type":    doc.get("source_type", "artikel"),
                    "title":          doc.get("title", ""),
                    "full_text":      doc.get("full_text", ""),
                },
                uuid=art_uuid,
            )
            doc_count += 1

            text = doc.get("full_text", "")
            if max_tokens is None:
                # Artikel-weise: ganzes Dokument ist ein einziger Chunk
                chunks = [text] if text else []
            else:
                chunks = split_tokens(text, max_tokens, overlap)
            pairs.append(((art_uuid, int(doc.get("article_number", 0)), doc.get("title", "")), chunks))

    with chunk_col.batch.dynamic() as batch:
        for (art_uuid, art_number, art_title), chunks in pairs:
            total = len(chunks)
            for i, text in enumerate(chunks):
                batch.add_object(
                    properties={
                        "content":        text,
                        "chunk_index":    i,
                        "chunk_total":    total,
                        "article_number": art_number,
                        "title":          art_title,
                    },
                    references={"ofArticle": art_uuid},
                )
                chunk_count += 1

    return {"documents": doc_count, "chunks": chunk_count}


def get_status(client) -> dict:
    """Gibt Status für alle 9 Collections (3 Modelle × 3 Strategien) zurück."""
    result = {}
    for model_key in MODELS:
        for strategy_key in STRATEGIES:
            key = (model_key, strategy_key)
            prefix = collection_prefix(model_key, strategy_key)
            art = f"{prefix}_Article"
            chunk_c = f"{prefix}_Chunk"
            docs = 0
            chunks = 0

            if client.collections.exists(art):
                agg = client.collections.get(art).aggregate.over_all(total_count=True)
                docs = agg.total_count or 0

            if client.collections.exists(chunk_c):
                agg = client.collections.get(chunk_c).aggregate.over_all(total_count=True)
                chunks = agg.total_count or 0

            result[key] = {"documents": docs, "chunks": chunks}

    return result


def browse_collection(client, model_key: str, strategy_key: str, limit: int = 20) -> list:
    """
    Gibt die ersten N Artikel einer Collection zurück (ohne Vektoren).
    Für jeden Artikel wird zusätzlich die Chunk-Anzahl aus der Chunk-Collection ermittelt.
    """
    from weaviate.classes.query import Filter

    prefix = collection_prefix(model_key, strategy_key)
    art = f"{prefix}_Article"
    chunk_c = f"{prefix}_Chunk"

    if not client.collections.exists(art):
        return []

    col = client.collections.get(art)
    results = col.query.fetch_objects(limit=limit, include_vector=False)

    has_chunks = client.collections.exists(chunk_c)
    chunk_col = client.collections.get(chunk_c) if has_chunks else None

    rows = []
    for obj in results.objects:
        article_uuid = str(obj.uuid)
        chunk_count = 0
        if chunk_col:
            try:
                agg = chunk_col.aggregate.over_all(
                    filters=Filter.by_ref("ofArticle").by_id().equal(article_uuid),
                    total_count=True,
                )
                chunk_count = agg.total_count or 0
            except Exception:
                chunk_count = 0
        rows.append({
            "uuid":           article_uuid,
            "article_number": obj.properties.get("article_number", 0),
            "title":          obj.properties.get("title", ""),
            "source_type":    obj.properties.get("source_type", ""),
            "full_text":      obj.properties.get("full_text", ""),
            "chunk_count":    chunk_count,
        })

    return rows


def retrieve_chunks(
    client,
    model_key: str,
    strategy_key: str,
    question: str,
    top_k: int = 3,
    method: str = "semantic",
    hybrid_alpha: float = 0.5,
) -> list:
    """
    Suche in der Chunk-Collection einer bestimmten Collection.

    method:
      "semantic"  — Vektor-Nearest-Neighbour (near_text), Score = 1 − Distanz
      "bm25"      — Keyword-Suche nach Zipf/TF-IDF-Prinzip, Score = BM25-Rohwert
      "hybrid"    — BM25 + Semantic kombiniert, alpha steuert Gewichtung
                    (0 = reines BM25, 1 = rein semantisch), Score = Hybrid-Score 0–1

    article_number und title sind direkt im Chunk als Properties gespeichert
    (gleiche Struktur wie im alten Hausarbeit-Schema).
    """
    from weaviate.classes.query import MetadataQuery

    prefix = collection_prefix(model_key, strategy_key)
    chunk_c = f"{prefix}_Chunk"

    if not client.collections.exists(chunk_c):
        return []

    col = client.collections.get(chunk_c)

    if method == "bm25":
        results = col.query.bm25(
            query=question,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
            include_vector=False,
        )
    elif method == "hybrid":
        results = col.query.hybrid(
            query=question,
            alpha=hybrid_alpha,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
            include_vector=False,
        )
    else:
        # Standard: semantische Suche
        results = col.query.near_text(
            query=question,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
            include_vector=False,
        )

    hits = []
    for obj in results.objects:
        meta = obj.metadata

        if method == "semantic":
            dist = meta.distance if meta else None
            score = round(max(0.0, 1.0 - (dist or 0.0)), 3)
        else:
            raw = (meta.score if meta else None) or 0.0
            score = round(raw, 4)

        hits.append({
            "content":        obj.properties.get("content", ""),
            "article_number": obj.properties.get("article_number", 0),
            "title":          obj.properties.get("title", ""),
            "chunk_index":    obj.properties.get("chunk_index", 0),
            "score":          score,
        })

    return hits


def get_chunks_for_article(client, model_key: str, strategy_key: str, article_uuid: str) -> list:
    """Gibt alle Chunks zurück die zu einem Artikel gehören."""
    from weaviate.classes.query import Filter
    prefix = collection_prefix(model_key, strategy_key)
    chunk_c = f"{prefix}_Chunk"
    if not client.collections.exists(chunk_c):
        return []
    col = client.collections.get(chunk_c)
    results = col.query.fetch_objects(
        limit=100,
        filters=Filter.by_ref("ofArticle").by_id().equal(article_uuid),
        include_vector=False,
    )
    chunks = [
        {
            "index":   obj.properties.get("chunk_index", 0),
            "total":   obj.properties.get("chunk_total", 0),
            "content": obj.properties.get("content", ""),
        }
        for obj in results.objects
    ]
    return sorted(chunks, key=lambda c: c["index"])


def split_tokens(text: str, max_tokens: int = 500, overlap: int = 50) -> list:
    if not text:
        return []

    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = _enc.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = max(0, end - overlap)

    return chunks
