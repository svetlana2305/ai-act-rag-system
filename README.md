# EU AI Act RAG-System

Seminararbeit im Masterstudiengang bei Prof. Dr. Christian Gawron.

**Thema:** Einfluss von Chunking-Strategie und Embedding-Modell auf die Retrieval-Qualität in RAG-Systemen am Beispiel des EU AI Act.

**Team:**  Ricarda  ·   Svetlana

**Abgabe:** 29. September 2026

---

## Worum geht es?

Das System scrapt den EU AI Act, zerlegt ihn in Textabschnitte (Chunks) und vergleicht systematisch Chunking-Strategien und Embedding-Modelle anhand standardisierter Retrieval-Metriken.

### Forschungsdesign

Faktorielles Experiment mit zweistufiger Analyse:

| Variable | Ausprägungen |
|---|---|
| Chunking-Strategie | Fixed Character (1000/150), Sentence (8/1), Recursive Character (1000/150), Semantic (Threshold 0.75) |
| Embedding-Modell | text-embedding-3-small, text-embedding-3-large, paraphrase-multilingual-mpnet-base-v2 (SBERT), intfloat/multilingual-e5-large |
| Retrieval-Methode | BM25, Dense, Hybrid (alpha=0.5) |

**Hauptanalyse:** 4 × 4 = 16 Konfigurationen mit Hybrid-Retrieval.
**Tiefenanalyse:** beste Konfiguration über alle 3 Retrieval-Methoden.

### Metriken

- **Information Retrieval:** Precision@5, MRR@5, NDCG@5
- **Generierungsqualität:** ragas Faithfulness, Context Precision, Context Recall, Answer Relevancy

### Korpus

EU AI Act, 306 Dokumente: 113 Artikel + 13 Anhänge + 180 Erwägungsgründe.
Quelle: artificialintelligenceact.eu/de

---

## Lokal starten

**Voraussetzung:** Docker Desktop

```bash
