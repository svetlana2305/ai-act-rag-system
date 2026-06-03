import logging
import os
import uuid
import tiktoken
import weaviate
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty

logger = logging.getLogger(__name__)

MODELS = {
    "3Small": "text-embedding-3-small",
    "3Large": "text-embedding-3-large",
}

STRATEGIES = {
    "Artikel": {"label": "Dokument-weise", "max_tokens": None, "overlap": 0},
    "500": {"label": "500 Tokens / 50 Overlap", "max_tokens": 500, "overlap": 50},
    "2000": {"label": "2000 Tokens / 200 Overlap", "max_tokens": 2000, "overlap": 200},
}

_enc = tiktoken.get_encoding("cl100k_base")

def collection_prefix(model_key: str, strategy_key: str) -> str:
    return f"AIAct_{model_key}_{strategy_key}"

def get_client():
    url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    api_key = os.getenv("OPENAI_API_KEY")
        clean_host = url.replace("http://", "").replace("https://", "")
        if ":" in clean_host:
        host, port_str = clean_host.split(":")
        port = int(port_str)
    else:
        host = clean_host
        port = 8080
        
    return weaviate.connect_to_custom(
        host=host,
        port=port,
        grpc_port=port + 1, # Sliplane tunnelt gRPC meist über den Folgeport oder Standard
        headers={"X-OpenAI-Api-Key": api_key}
    )

def get_status(client) -> dict:
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
                except: pass
            results[(m_key, s_key)] = {"documents": docs, "chunks": chunks, "exists": exists}
    return results

def setup_collection(client, model_key, strategy_key):
    model = MODELS[model_key]
    prefix = collection_prefix(model_key, strategy_key)
    art, chunk = f"{prefix}_Article", f"{prefix}_Chunk"
    if not client.collections.exists(art):
        client.collections.create(
            name=art,
            properties=[
                Property(name="article_number", data_type=DataType.INT, skip_vectorization=True),
                Property(name="source_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="title", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="full_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_openai(model=model),
            generative_config=Configure.Generative.openai(),
        )
    if not client.collections.exists(chunk):
        client.collections.create(
            name=chunk,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT, skip_vectorization=True),
                Property(name="chunk_total", data_type=DataType.INT, skip_vectorization=True),
                Property(name="article_number", data_type=DataType.INT, skip_vectorization=True),
                Property(name="title", data_type=DataType.TEXT, skip_vectorization=True),
            ],
            references=[ReferenceProperty(name="ofArticle", target_collection=art)],
            vectorizer_config=Configure.Vectorizer.text2vec_openai(model=model),
            generative_config=Configure.Generative.openai(),
        )

def delete_collection(client, m, s):
    prefix = collection_prefix(m, s)
    for n in [f"{prefix}_Chunk", f"{prefix}_Article"]:
        if client.collections.exists(n): client.collections.delete(n)

def import_articles(client, documents, model_key, strategy_key):
    setup_collection(client, model_key, strategy_key)
    strat = STRATEGIES[strategy_key]
    prefix = collection_prefix(model_key, strategy_key)
    art_col, chunk_col = client.collections.get(f"{prefix}_Article"), client.collections.get(f"{prefix}_Chunk")
    doc_count, chunk_count, pairs = 0, 0, []
    with art_col.batch.dynamic() as batch:
        for doc in documents:
            uid = str(uuid.uuid4())
            batch.add_object(properties={
                "article_number": int(doc.get("article_number", 0)),
                "source_type": doc.get("source_type", "artikel"),
                "title": doc.get("title", ""),
                "full_text": doc.get("full_text", ""),
            }, uuid=uid)
            doc_count += 1
            text = doc.get("full_text", "")
            chunks = [text] if strat["max_tokens"] is None else split_tokens(text, strat["max_tokens"], strat["overlap"])
            pairs.append(((uid, int(doc.get("article_number", 0)), doc.get("title", "")), chunks))
    with chunk_col.batch.dynamic() as batch:
        for (uid, nr, title), chunks in pairs:
            for i, text in enumerate(chunks):
                batch.add_object(properties={
                    "content": text, "chunk_index": i, "chunk_total": len(chunks),
                    "article_number": nr, "title": title
                }, references={"ofArticle": uid})
                chunk_count += 1
    return {"documents": doc_count, "chunks": chunk_count}

def browse_collection(client, m, s, limit=20):
    from weaviate.classes.query import Filter
    art = f"{collection_prefix(m, s)}_Article"
    if not client.collections.exists(art): return []
    res = client.collections.get(art).query.fetch_objects(limit=limit)
    rows = []
    chunk_col = client.collections.get(f"{collection_prefix(m, s)}_Chunk")
    for obj in res.objects:
        try: cnt = chunk_col.aggregate.over_all(filters=Filter.by_ref("ofArticle").by_id().equal(str(obj.uuid)), total_count=True).total_count or 0
        except: cnt = 0
        rows.append({"uuid": str(obj.uuid), "article_number": obj.properties.get("article_number", 0), "title": obj.properties.get("title", ""), "source_type": obj.properties.get("source_type", ""), "full_text": obj.properties.get("full_text", ""), "chunk_count": cnt})
    return rows

def retrieve_chunks(client, m, s, question, top_k=3, method="semantic", hybrid_alpha=0.5):
    from weaviate.classes.query import MetadataQuery
    name = f"{collection_prefix(m, s)}_Chunk"
    if not client.collections.exists(name): return []
    col = client.collections.get(name)
    if method == "bm25": res = col.query.bm25(query=question, limit=top_k, return_metadata=MetadataQuery(score=True))
    elif method == "hybrid": res = col.query.hybrid(query=question, alpha=hybrid_alpha, limit=top_k, return_metadata=MetadataQuery(score=True))
    else: res = col.query.near_text(query=question, limit=top_k, return_metadata=MetadataQuery(distance=True))
    hits = []
    for obj in res.objects:
        m_data = obj.metadata
        score = round(max(0.0, 1.0 - (m_data.distance or 0.0)), 3) if method == "semantic" else round(m_data.score or 0.0, 4)
        hits.append({"content": obj.properties.get("content", ""), "article_number": obj.properties.get("article_number", 0), "title": obj.properties.get("title", ""), "chunk_index": obj.properties.get("chunk_index", 0), "score": score})
    return hits

def get_chunks_for_article(client, m, s, uid):
    from weaviate.classes.query import Filter
    name = f"{collection_prefix(m, s)}_Chunk"
    res = client.collections.get(name).query.fetch_objects(filters=Filter.by_ref("ofArticle").by_id().equal(uid))
    return sorted([{"index": o.properties["chunk_index"], "total": o.properties["chunk_total"], "content": o.properties["content"]} for o in res.objects], key=lambda x: x["index"])

def split_tokens(text, max_tokens=500, overlap=50):
    if not text: return []
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens: return [text]
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]).strip())
        if end == len(tokens): break
        start = max(0, end - overlap)
    return chunks
