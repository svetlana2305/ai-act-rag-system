# EU AI Act RAG-System

Seminararbeit im Masterstudiengang bei Prof. Dr. Christian Gawron.

**Thema:** Einfluss von Chunking-Strategie und Embedding-Modell auf die Retrieval-Qualität in RAG-Systemen am Beispiel des EU AI Act.

**Team:** Ricarda und Svetlana

**Abgabe:** 29. September 2026

**Live-App:** https://ai-act-rag-system.sliplane.app/

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

## Nutzung der Live-App

1. https://ai-act-rag-system.sliplane.app/ öffnen
2. Mit Admin-Credentials einloggen
3. Sidebar prüfen: Weaviate sollte online sein
4. **Tab "Suche & Ausgabe":** Frage stellen, Modell + Chunking-Strategie + Methode wählen, Top-K Treffer ansehen oder generierte Antwort lesen
5. **Tab "Datenbank-Browser":** durch importierte Artikel browsen

Auf der Live-Instanz sind die Collections vorbefüllt. Re-Scrape oder Re-Import läuft über die Sidebar "System-Wartung".

### Eval-Funktion (geplant Etappe 2)

In Streamlit integrierter Batch-Modus erlaubt mehrere Testfragen gleichzeitig. Ergebnisse werden als Markdown exportiert. Vollständige Eval läuft als separates Skript (siehe unten).

---

## Reproduzierbarkeit der Eval

Wird in Etappe 2-6 implementiert. Geplanter Ablauf:

```bash
# Voraussetzung: alle Collections auf Sliplane befüllt
python run_eval.py
```

Output:
- `results_main.csv` — 16 Konfigurationen × 3 IR-Metriken
- `results_retrieval.csv` — Tiefenanalyse mit BM25/Dense/Hybrid
- `results_ragas.csv` — Top-3 Konfigurationen × 4 ragas-Metriken

Testdatensatz: `data/goldstandard.json`. Aktueller Stand: 10 manuell annotierte Frage-Artikel-Paare, Ausbau auf 20 in Etappe 1 geplant.

---

## Deployment auf Sliplane

Die App läuft auf Sliplane mit Docker-Compose-basiertem Stack (Weaviate + Streamlit). Auto-Deploy ist auf Branch `main` konfiguriert.

### Initial-Setup (einmalig, bereits erfolgt)

1. Repository in Sliplane verbinden: *Add Service → From GitHub → ai-act-rag-system*
2. Sliplane erkennt `docker-compose.yml` automatisch. Im Dashboard laufen zwei Services: `weaviate` und `streamlit-app`.
3. Umgebungsvariablen im Sliplane-Dashboard setzen (siehe unten).
4. Volume `weaviate_data` persistiert über Container-Restarts hinweg.

### Workflow für Code-Änderungen

```
Feature-Branch  →  Pull Request  →  Review  →  Merge in main  →  Auto-Deploy Sliplane
```

Branch `feature/nlp-hausarbeit` ist aktiv in Entwicklung. Hauptstand: Tag `stable-2026-06-15` auf `main`.

---

## Lokal starten (optional)

Nur bei größeren Code-Änderungen oder zum Debuggen. Voraussetzung: Docker Desktop.

```bash
git clone https://github.com/svetlana2305/ai-act-rag-system.git
cd ai-act-rag-system
cp .env.example .env
# .env befüllen
docker compose up --build
```

- Admin-App: http://localhost:8501
- Weaviate: http://localhost:8080

---

## Umgebungsvariablen

| Variable | Pflicht | Beschreibung |
|---|---|---|
| `OPENAI_API_KEY` | ja | OpenAI-Key für Embeddings und Generierung |
| `ADMIN_USER_1` / `ADMIN_PASS_1` | ja | Svetlanas Login (Username `Svetlana`) |
| `ADMIN_USER_2` / `ADMIN_PASS_2` | nein | Ricardas Login (Username `Ricarda`) |
| `WEAVIATE_URL` | nein | Standard `http://weaviate.internal:8080` auf Sliplane, `http://weaviate:8080` lokal |
| `DATA_DIR` | nein | Pfad für Scraping-Cache, Standard `/app/data` |

---

## Projektstruktur

```
admin_app.py         Streamlit-Frontend (Login, Scrape, Import, Suche, Eval)
database_manager.py  Weaviate-Logik (Schema, Import, Status, Retrieval)
scraper.py           Web-Scraper mit Last-Modified-Prüfung
chunking.py          Chunking-Strategien (geplant Etappe 4)
embeddings.py        Embedding-Adapter OpenAI + Sentence-Transformers (geplant Etappe 5)
eval_ir.py           IR-Metriken Precision@5, MRR@5, NDCG@5 (geplant Etappe 2)
eval_ragas.py        ragas-Pipeline (geplant Etappe 3)
run_eval.py          Orchestrierung der Eval-Läufe (geplant Etappe 2)
docker-compose.yml   Weaviate + Streamlit als Docker-Stack
Dockerfile           Image für die Streamlit-App
requirements.txt     Python-Abhängigkeiten
data/
  goldstandard.json  Manuell annotierter Testdatensatz (aktuell 10 Fragen, Ziel 20 in Etappe 1)
  cache_*.json       Gecachte Scraping-Ergebnisse
```

---

## Roadmap

- [x] Etappe 0: Sicherungs-Setup (Git-Tag + Feature-Branch)
- [ ] Etappe 1: Goldstandard-Testdatensatz (10 → 20 Fragen, kategorisiert)
- [ ] Etappe 2: Eval-Skripte (Precision@5, MRR@5, NDCG@5)
- [ ] Etappe 3: ragas-Pipeline
- [ ] Etappe 4: Vier Chunking-Strategien (Fixed, Sentence, Recursive, Semantic)
- [ ] Etappe 5: SBERT + E5-Large Embeddings
- [ ] Etappe 6: Vollständige Eval-Läufe
- [ ] Etappe 7: Hausarbeit schreiben

---

## Vorstudie

Aufbauend auf der Datenbanksysteme-Hausarbeit der Autorinnen (Januar 2026), die qualitativ den Einfluss von Chunking auf die Retrieval-Qualität untersucht hat. Die vorliegende NLP-Arbeit verallgemeinert diese Erkenntnis systematisch über mehrere Chunking-Strategien und Embedding-Modelle mit quantitativen Metriken.
