"""Weaviate-Anbindung für die Hausarbeit.
Steuert Schema, Import (mit Chunking + Embedding), Retrieval und Status.
"""
import logging
import os
import uuid
import weaviate
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty

from chunking import chunk
from embeddings import embed, OPENAI_MODELS, LOCAL_MODELS

logger = logging.getLogger(__name__)

# Vier Embedding-Modelle: zwei OpenAI (Cloud) und zwei Open-Source (lokal)
MODELS = {
    "3Small": "text-embedding-3-small",
    "3Large": "text-embedding-3-large",
    "SBERT": "paraphrase-multilingual-mpnet-base-v2",
    "E5Large": "intfloat/multilingual-e5-large",
}

# Vier Chunking-Strategien gemäß NLP-Konzept
STRATEGIES = {
    "Fixed": {"label": "Fixed Character (1000/150)", "type": "fixed", "size": 1000, "overlap": 150},
    "Sentence": {"label": "Sentence (8/1)", "type": "sentence", "max_sentences": 8, "overlap_sentences": 1},
    "Recursive": {"label": "Recursive Character (1000/150)", "type": "recursive", "size": 1000, "overlap": 150},
    "Semantic": {"label": "Semantic (Threshold 0.75)", "type": "semantic", "threshold": 0.75},
}


def collection_prefix(model_key: str, strategy_key: str) -> str:
    return f"AIAct_{model_key}_{strategy_key}"


def get_client():
    url = os.getenv("WEAVIATE_URL", "http://weaviate.internal:8080")
    api_key = os.getenv("OPENAI_API_KEY")

    clean_host = url.replace("http://", "").replace("https://", "")
    if ":" in clean_host:
        host, port_str = clean_host.split(":")
        port = int(port_str)
    else:
        host = clean_host
        port = 8080

    headers = {"X-OpenAI-Api-Key": api_key} if api_key else {}

    return weaviate.connect_to_custom(
        http_host=host, http_port=port, http_secure=False,
        grpc_host=host, grpc_port=50051, grpc_secure=False,
        headers=headers,
    )


def get_status(client) -> dict:
    """Liefert pro Modell × Strategie die Anzahl Dokumente und Chunks."""
    results = {}
    for m_key in MODELS:
        for s_key in STRATEGIES:
            prefix = collection_prefix(m_key, s_key)
            name = f"{prefix}_Chunk"
            exists = client.collections.exists(name)
            docs, chunks = 0, 0
            if exists:
                try:
                    chunks = client.collections.get(name).aggregate.over_all(total_count=True).total_count or 0
                    art_name = f"{prefix}_Article"
                    if client.collections.exists(art_name):
                        docs = client.collections.get(art_name).aggregate.over_all(total_count=True).total_count or 0
                except Exception:
                    pass
            results[(m_key, s_key)] = {"documents": docs, "chunks": chunks, "exists": exists}
    return results


def setup_collection(client, model_key, strategy_key):
    """Erstellt Article- und Chunk-Collection. OpenAI: server-vectorizer, lokal: none."""
    model_name = MODELS[model_key]
    prefix = collection_prefix(model_key, strategy_key)
    art, chunk_name = f"{prefix}_Article", f"{prefix}_Chunk"

    if model_key in OPENAI_MODELS:
        vec_cfg = Configure.Vectorizer.text2vec_openai(model=model_name)
        gen_cfg = Configure.Generative.openai()
    else:
        vec_cfg = Configure.Vectorizer.none()
        gen_cfg = None

    if not client.collections.exists(art):
        kwargs = dict(
            name=art,
            properties=[
                Property(name="article_number", data_type=DataType.INT, skip_vectorization=True),
                Property(name="source_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="title", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="full_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_cfg,
        )
        if gen_cfg:
            kwargs["generative_config"] = gen_cfg
        client.collections.create(**kwargs)

    if not client.collections.exists(chunk_name):
        kwargs = dict(
            name=chunk_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT, skip_vectorization=True),
                Property(name="chunk_total", data_type=DataType.INT, skip_vectorization=True),
                Property(name="article_number", data_type=DataType.INT, skip_vectorization=True),
                Property(name="source_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="title", data_type=DataType.TEXT, skip_vectorization=True),
            ],
            references=[ReferenceProperty(name="ofArticle", target_collection=art)],
            vectorizer_config=vec_cfg,
        )
        if gen_cfg:
            kwargs["generative_config"] = gen_cfg
        client.collections.create(**kwargs)


def delete_collection(client, m, s):
    prefix = collection_prefix(m, s)
    for n in [f"{prefix}_Chunk", f"{prefix}_Article"]:
        if client.collections.exists(n):
            client.collections.delete(n)


def import_articles(client, documents, model_key, strategy_key):
    """Importiert Dokumente mit gewählter Chunking-Strategie und Embedding-Modell."""
    delete_collection(client, model_key, strategy_key)
    setup_collection(client, model_key, strategy_key)
    strat = STRATEGIES[strategy_key]
    prefix = collection_prefix(model_key, strategy_key)
    art_col = client.collections.get(f"{prefix}_Article")
    chunk_col = client.collections.get(f"{prefix}_Chunk")

    is_local = model_key in LOCAL_MODELS
    doc_count, chunk_count, pairs = 0, 0, []

    # 1. Artikel schreiben + chunken
    with art_col.batch.dynamic() as batch:
        for doc in documents:
            uid = str(uuid.uuid4())
            properties = {
                "article_number": int(doc.get("article_number", 0)),
                "source_type": doc.get("source_type", "artikel"),
                "title": doc.get("title", ""),
                "full_text": doc.get("full_text", ""),
            }
            if is_local:
                doc_vec = embed(model_key, [doc.get("full_text", "")])[0]
                batch.add_object(properties=properties, uuid=uid, vector=doc_vec)
            else:
                batch.add_object(properties=properties, uuid=uid)
            doc_count += 1
            chunks = chunk(doc.get("full_text", ""), strat)
            pairs.append(((uid, int(doc.get("article_number", 0)), doc.get("title", ""), doc.get("source_type", "artikel")), chunks))

    # 2. Bei lokalen Modellen: ALLE Chunks in einem Rutsch vektorisieren (schnelles Batching)
    local_vectors = {}
    if is_local:
        flat_texts, index_map = [], []
        for p_idx, (_, chunks) in enumerate(pairs):
            for c_idx, text in enumerate(chunks):
                flat_texts.append(text)
                index_map.append((p_idx, c_idx))
        all_vecs = embed(model_key, flat_texts) if flat_texts else []
        for (p_idx, c_idx), vec in zip(index_map, all_vecs):
            local_vectors[(p_idx, c_idx)] = vec

    # 3. Chunks schreiben
    with chunk_col.batch.dynamic() as batch:
        for p_idx, ((uid, nr, title, src), chunks) in enumerate(pairs):
            for c_idx, text in enumerate(chunks):
                properties = {
                    "content": text,
                    "chunk_index": c_idx,
                    "chunk_total": len(chunks),
                    "article_number": nr,
                    "source_type": src,
                    "title": title,
                }
                kwargs = dict(properties=properties, references={"ofArticle": uid})
                if is_local:
                    kwargs["vector"] = local_vectors[(p_idx, c_idx)]
                batch.add_object(**kwargs)
                chunk_count += 1

    return {"documents": doc_count, "chunks": chunk_count}

    is_local = model_key in LOCAL_MODELS

    doc_count, chunk_count, pairs = 0, 0, []

    with art_col.batch.dynamic() as batch:
        for doc in documents:
            uid = str(uuid.uuid4())
            properties = {
                "article_number": int(doc.get("article_number", 0)),
                "source_type": doc.get("source_type", "artikel"),
                "title": doc.get("title", ""),
                "full_text": doc.get("full_text", ""),
            }
            if is_local:
                vec = embed(model_key, [doc.get("full_text", "")])[0]
                batch.add_object(properties=properties, uuid=uid, vector=vec)
            else:
                batch.add_object(properties=properties, uuid=uid)
            doc_count += 1

            chunks = chunk(doc.get("full_text", ""), strat)
            pairs.append(((uid, int(doc.get("article_number", 0)), doc.get("title", ""), doc.get("source_type", "artikel")), chunks))

    with chunk_col.batch.dynamic() as batch:
        for (uid, nr, title, src), chunks in pairs:
            if not chunks:
                continue
            vectors = embed(model_key, chunks) if is_local else None
            for i, text in enumerate(chunks):
                properties = {
                    "content": text,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "article_number": nr,
                    "source_type": src,
                    "title": title,
                }
                kwargs = dict(properties=properties, references={"ofArticle": uid})
                if is_local:
                    kwargs["vector"] = vectors[i]
                batch.add_object(**kwargs)
                chunk_count += 1

    return {"documents": doc_count, "chunks": chunk_count}


def browse_collection(client, m, s, limit=20):
    from weaviate.classes.query import Filter

    art = f"{collection_prefix(m, s)}_Article"
    if not client.collections.exists(art):
        return []
    res = client.collections.get(art).query.fetch_objects(limit=limit)
    rows = []
    chunk_col = client.collections.get(f"{collection_prefix(m, s)}_Chunk")
    for obj in res.objects:
        try:
            cnt = chunk_col.aggregate.over_all(
                filters=Filter.by_ref("ofArticle").by_id().equal(str(obj.uuid)),
                total_count=True,
            ).total_count or 0
        except Exception:
            cnt = 0
        rows.append({
            "uuid": str(obj.uuid),
            "article_number": obj.properties.get("article_number", 0),
            "title": obj.properties.get("title", ""),
            "source_type": obj.properties.get("source_type", ""),
            "full_text": obj.properties.get("full_text", ""),
            "chunk_count": cnt,
        })
    return rows


def retrieve_chunks(client, m, s, question, top_k=3, method="semantic", hybrid_alpha=0.5):
    from weaviate.classes.query import MetadataQuery

    name = f"{collection_prefix(m, s)}_Chunk"
    if not client.collections.exists(name):
        return []
    col = client.collections.get(name)
    is_local = m in LOCAL_MODELS

    if method == "bm25":
        res = col.query.bm25(query=question, limit=top_k, return_metadata=MetadataQuery(score=True))
    elif method == "hybrid":
        if is_local:
            qvec = embed(m, [question], is_query=True)[0]
            res = col.query.hybrid(query=question, vector=qvec, alpha=hybrid_alpha, limit=top_k, return_metadata=MetadataQuery(score=True))
        else:
            res = col.query.hybrid(query=question, alpha=hybrid_alpha, limit=top_k, return_metadata=MetadataQuery(score=True))
    else:  # semantic
        if is_local:
            qvec = embed(m, [question], is_query=True)[0]
            res = col.query.near_vector(near_vector=qvec, limit=top_k, return_metadata=MetadataQuery(distance=True))
        else:
            res = col.query.near_text(query=question, limit=top_k, return_metadata=MetadataQuery(distance=True))

    hits = []
    for obj in res.objects:
        m_data = obj.metadata
        if method == "semantic":
            score = round(max(0.0, 1.0 - (m_data.distance or 0.0)), 3)
        else:
            score = round(m_data.score or 0.0, 4)
        hits.append({
            "content": obj.properties.get("content", ""),
            "article_number": obj.properties.get("article_number", 0),
            "source_type": obj.properties.get("source_type", "artikel"),
            "title": obj.properties.get("title", ""),
            "chunk_index": obj.properties.get("chunk_index", 0),
            "score": score,
        })
    return hits


def get_chunks_for_article(client, m, s, uid):
    from weaviate.classes.query import Filter

    name = f"{collection_prefix(m, s)}_Chunk"
    res = client.collections.get(name).query.fetch_objects(
        filters=Filter.by_ref("ofArticle").by_id().equal(uid)
    )
    return sorted([
        {"index": o.properties["chunk_index"], "total": o.properties["chunk_total"], "content": o.properties["content"]}
        for o in res.objects
    ], key=lambda x: x["index"])
