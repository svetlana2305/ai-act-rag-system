# Systemdokumentation — EU AI Act RAG

Technische Dokumentation zum System. Ergänzt die README (Überblick) um Architektur, Datenfluss, Modul-Details und Fachbegriffe.

Stand: Juni 2026 · Team Svetlana + Ricarda · Prof. Dr. Christian Gawron

---

## 1. Architektur

Zwei Dienste auf Sliplane, verbunden über ein internes Docker-Netzwerk:

| Dienst | Erreichbarkeit | Aufgabe |
|---|---|---|
| `streamlit-app` | öffentliche URL, Login | Frontend, Chunking, Embedding, Retrieval, Eval |
| `weaviate` | intern (`weaviate.internal:8080`) | Vektordatenbank und BM25-Index |

Weaviate ist bewusst nicht öffentlich erreichbar. Alle Zugriffe laufen über die Streamlit-App. Server-Anforderung: mindestens 4 GB RAM, da die Open-Source-Modelle (SBERT, E5-Large) lokal im Container geladen und gerechnet werden.

---

## 2. Datenfluss

```
EU AI Act (artificialintelligenceact.eu/de)
   |  scraper.py  (requests + BeautifulSoup, Last-Modified-Cache)
   v
data/cache_*.json   (306 Dokumente: 113 Artikel, 13 Anhaenge, 180 Erwaegungsgruende)
   |
   |  Import (admin_app.py  ->  database_manager.import_articles)
   |
   +-- Phase 1: Chunking    chunking.py     Text  ->  Liste von Chunks
   +-- Phase 2: Embedding   embeddings.py   Chunk ->  Vektor
   |
   v
Weaviate Collection  (Article + Chunk, verknuepft ueber ofArticle)
   |
   |  Retrieval (database_manager.retrieve_chunks)
   |
   +-- BM25 / Dense / Hybrid  ->  Top-K Treffer
   |
   v
Eval (eval_ir.py)  ->  Precision@5, MRR@5, NDCG@5  gegen eval/goldstandard.json
```

Chunking und Embedding sind strikt getrennt: erst wird der Text zerlegt, dann wird jeder Chunk vektorisiert. So lassen sich beide Achsen unabhängig variieren.

---

## 3. Datenmodell in Weaviate

Pro Kombination aus Modell und Strategie existieren zwei Collections:

- `AIAct_<Modell>_<Strategie>_Article` — das Gesamtdokument mit Metadaten
- `AIAct_<Modell>_<Strategie>_Chunk` — die Textabschnitte, je mit Referenz `ofArticle` auf ihr Dokument

Beispiel: `AIAct_3Small_Fixed_Chunk`.

Eigenschaften der Chunk-Objekte:

| Feld | Typ | Zweck |
|---|---|---|
| `content` | Text | der Chunk-Text (wird vektorisiert) |
| `chunk_index` | Int | Position innerhalb des Dokuments |
| `chunk_total` | Int | Gesamtzahl Chunks dieses Dokuments |
| `article_number` | Int | Nummer von Artikel/Anhang/Erwägungsgrund |
| `source_type` | Text | `artikel`, `anhang` oder `erwaegungsgrund` |
| `title` | Text | Überschrift des Dokuments |

`source_type` ist wichtig, damit Treffer eindeutig zugeordnet werden: Artikel 5 und Erwägungsgrund 5 sind verschiedene Dokumente.

---

## 4. Module im Detail

### scraper.py
Lädt Artikel, Anhänge und Erwägungsgründe. Prüft per Last-Modified-Header, ob neu geladen werden muss, sonst Cache. Speichert je Dokumenttyp eine JSON-Datei unter `data/`. Ein Typ wird nur dann neu gecacht, wenn mindestens 80 Prozent der erwarteten Dokumente geladen wurden.

### chunking.py
Vier Strategien, einheitliche Rückgabe als Liste von Strings:

| Funktion | Strategie | Bibliothek |
|---|---|---|
| `chunk_fixed` | Fixed Character (1000/150) | reines Python-Slicing |
| `chunk_sentence` | Sentence (8 Sätze, 1 Overlap) | NLTK (deutscher Tokenizer) |
| `chunk_recursive` | Recursive Character (1000/150) | LangChain RecursiveCharacterTextSplitter |
| `chunk_semantic` | Semantic (Threshold 0.75) | LangChain SemanticChunker |

Der `SemanticChunker` nutzt `text-embedding-3-small` als Hilfsmodell, um Themenwechsel zu erkennen. Dieser Aufruf kostet OpenAI-Guthaben, unabhängig vom später gewählten Embedding-Modell.

### embeddings.py
Einheitliche Funktion `embed(model_key, texts, is_query=False)`:

| Modell | Quelle | Dimensionen | Besonderheit |
|---|---|---|---|
| 3Small | OpenAI API | 1536 | server-seitige Vektorisierung durch Weaviate |
| 3Large | OpenAI API | 3072 | server-seitige Vektorisierung durch Weaviate |
| SBERT | lokal (sentence-transformers) | 768 | Modell `paraphrase-multilingual-mpnet-base-v2` |
| E5Large | lokal (sentence-transformers) | 1024 | Modell `intfloat/multilingual-e5-large`, Prefix nötig |

E5-Modelle erwarten zwingend einen Prefix: `passage: ` für Dokumente, `query: ` für Suchanfragen. Ohne diesen Prefix sinkt die Retrieval-Qualität deutlich.

### database_manager.py
- `setup_collection` — legt Schema an. Für OpenAI-Modelle wird der Weaviate-Vectorizer `text2vec-openai` konfiguriert. Für lokale Modelle `Vectorizer.none()`, die Vektoren werden beim Insert direkt mitgegeben.
- `import_articles` — chunkt jedes Dokument einzeln, vektorisiert (bei lokalen Modellen) und schreibt Article- und Chunk-Objekte in Batches.
- `retrieve_chunks` — Suche. Bei OpenAI-Modellen läuft Dense-Suche über `near_text` (Weaviate vektorisiert die Anfrage selbst). Bei lokalen Modellen wird der Query-Vektor lokal berechnet und über `near_vector` gesucht.
- `get_status` — zählt Dokumente und Chunks je Collection.

### eval_ir.py
Reine Metrik-Funktionen ohne Weaviate-Abhängigkeit:
- `precision_at_k` — Anteil relevanter Treffer in den Top-K
- `mrr_at_k` — Rang des ersten relevanten Treffers (Mean Reciprocal Rank)
- `ndcg_at_k` — positionsgewichtete Bewertung (Normalized Discounted Cumulative Gain)
- `evaluate_query` — berechnet alle drei Metriken für eine Frage

### run_eval.py
Orchestriert die Evaluation: lädt den Goldstandard, iteriert über alle befüllten Collections und die drei Retrieval-Methoden, mittelt die Metriken über alle Fragen und schreibt das Ergebnis. Doppelt nutzbar: als importierbares Modul (`run_full_eval`) für die Streamlit-App und als eigenständiges Skript (`python run_eval.py`).

### admin_app.py
Streamlit-Frontend mit vier Tabs:
1. Daten & Import
2. Suche & Ausgabe
3. Datenbank-Browser
4. Evaluation

Login über bis zu drei Admin-Konten aus den Umgebungsvariablen.

---

## 5. Retrieval-Methoden

| Methode | Prinzip | Stärke |
|---|---|---|
| BM25 | lexikalisch, Termfrequenz und inverse Dokumenthäufigkeit | exakte Fachbegriffe, Artikelnummern |
| Dense (semantisch) | Vektor-Ähnlichkeit (Cosine) | inhaltlich verwandte Formulierungen, Synonyme |
| Hybrid | Kombination aus BM25 und Dense, gesteuert über `alpha` | ausgewogen, meist beste Gesamtleistung |

Der Hybrid-Parameter `alpha` steht auf 0.5 (gleichgewichtet). `alpha=0.0` wäre reines BM25, `alpha=1.0` rein semantisch.

---

## 6. Evaluation

Die Evaluation vergleicht die Top-5-Treffer jeder Konfiguration gegen den Goldstandard (`eval/goldstandard.json`, 30 Frage-Artikel-Paare).

- **Precision@5** — wie viele der fünf Treffer relevant sind
- **MRR@5** — wie weit oben der erste relevante Treffer steht
- **NDCG@5** — wie gut das gesamte Ranking ist (höhere Positionen zählen mehr)

Ergänzend ist eine ragas-Bewertung der Generierungsqualität geplant (Faithfulness, Context Precision, Context Recall, Answer Relevancy).

Bekannte Einschränkung: Der Goldstandard verweist auf Artikelnummern. Liefert die Suche einen Erwägungsgrund mit derselben Nummer, kann dies fälschlich als Treffer gewertet werden. Die `source_type`-Information ist in den Chunks vorhanden und kann bei Bedarf in die Eval-Logik einbezogen werden.

---

## 7. Fachbegriffe

**RAG (Retrieval-Augmented Generation)** — Antwortgenerierung mit vorgeschaltetem Wissensabruf: erst relevante Textstellen suchen, dann an ein Sprachmodell als Kontext übergeben.

**Embedding** — numerische Repräsentation eines Textes als Vektor. Semantisch ähnliche Texte liegen im Vektorraum nah beieinander.

**Chunk** — ein Textabschnitt. Lange Dokumente werden zerlegt, damit Vektoren präzise bleiben und Treffer fein lokalisiert werden können.

**BM25** — klassisches lexikalisches Ranking-Verfahren nach dem TF-IDF-Prinzip. Seltene Begriffe wiegen schwerer als häufige.

**HNSW** — Hierarchical Navigable Small World, der Graph-Index, mit dem Weaviate die Approximate-Nearest-Neighbor-Suche im Vektorraum effizient macht.

**Cosine Similarity** — Ähnlichkeitsmaß zwischen zwei Vektoren, Wertebereich 0 bis 1.

**Token** — kleinste Verarbeitungseinheit eines Sprachmodells, etwa 0,75 Wörter im Deutschen.

---

## 8. Bezug zur Vorlesung (Prof. Gawron)

| Vorlesungsinhalt | Umsetzung im System | Code-Stelle |
|---|---|---|
| BOW / TF-IDF / Zipfsches Gesetz | BM25-Suche | `retrieve_chunks` (Methode `bm25`) |
| Word Embeddings (Word2Vec) | konzeptionelle Grundlage der Vektorsuche | Theoriekapitel |
| Transformer / BERT | SBERT und OpenAI-Modelle basieren darauf | `embeddings.py` |
| Sentence-BERT | Open-Source-Modell SBERT multilingual | `embeddings.py` (SBERT) |
| RAG | Gesamtsystem Retrieval + Generierung | `admin_app.py`, Tab Suche |
| Tokenisierung | Satz-Chunking via NLTK | `chunking.py` (`chunk_sentence`) |
| Evaluation / IR-Metriken | Precision@5, MRR@5, NDCG@5 | `eval_ir.py` |

---

## 9. Vorstudie

Das System knüpft an die Datenbanksysteme-Hausarbeit der Autorinnen (Januar 2026) an. Dort wurde qualitativ gezeigt, dass Chunking die Retrieval-Qualität beeinflusst (Beispiel Artikel 5: relevante Passagen wurden erst durch Chunking zuverlässig gefunden). Die vorliegende NLP-Arbeit verallgemeinert diesen Befund systematisch über vier Chunking-Strategien und vier Embedding-Modelle mit quantitativen Metriken.
