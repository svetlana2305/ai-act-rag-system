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

Faktorielles Experiment:

| Variable | Ausprägungen |
|---|---|
| Chunking-Strategie | Fixed Character (1000/150), Sentence (8/1), Recursive Character (1000/150), Semantic (Threshold 0.75) |
| Embedding-Modell | text-embedding-3-small, text-embedding-3-large, paraphrase-multilingual-mpnet-base-v2 (SBERT), intfloat/multilingual-e5-large |
| Retrieval-Methode | BM25, Dense, Hybrid (alpha=0.5) |

**Zwei getrennte Verarbeitungsphasen:**
1. Chunking (`chunking.py`) — Text wird je Strategie in Abschnitte zerlegt
2. Embedding (`embeddings.py`) — jeder Chunk wird je Modell vektorisiert

**Matrix:** 4 Chunking × 4 Embedding = 16 Collections. Jede wird mit BM25, Dense und Hybrid abgefragt.

### Metriken

- **Information Retrieval:** Precision@5, MRR@5, NDCG@5
- **Generierungsqualität (geplant):** ragas Faithfulness, Context Precision, Context Recall, Answer Relevancy

### Korpus

EU AI Act, 306 Dokumente: 113 Artikel + 13 Anhänge + 180 Erwägungsgründe.
Quelle: artificialintelligenceact.eu/de

---

## Nutzung der Live-App

Vier Tabs:

1. **Daten & Import** — Korpus scrapen, Collections befüllen (einzeln oder alle), Status aller 16 Collections
2. **Suche & Ausgabe** — Frage stellen, Konfigurationen + Methoden wählen, Treffer ansehen oder Antwort generieren, Protokoll als Markdown exportieren
3. **Datenbank-Browser** — importierte Dokumente und Volltexte einsehen
4. **Evaluation** — Precision@5, MRR@5, NDCG@5 über alle befüllten Collections, Ergebnis als CSV

---

## Reproduzierbarkeit der Eval

In der Live-App Tab "Evaluation" → "Eval starten". Alternativ als Skript:

```bash
python run_eval.py
```

Output: `results_baseline.csv` — alle befüllten Konfigurationen × 3 IR-Metriken.

Testdatensatz: `eval/goldstandard.json` (30 manuell annotierte Frage-Artikel-Paare, kategorisiert nach definition/verbot/pflicht/transparenz/gpai/sanktion).

---

## Deployment auf Sliplane

Docker-Compose-Stack (Weaviate + Streamlit). Auto-Deploy auf Branch `main`.

### Server-Anforderung

Die Open-Source-Modelle (SBERT, E5-Large) laden lokal via `sentence-transformers` und brauchen Arbeitsspeicher. Empfehlung: **mindestens 4 GB RAM** (Server-Typ "Medium" oder größer). Auf 1-2 GB crasht der Import von E5-Large.

### Health-Check beim Import

Lange Importe blockieren den Streamlit-Thread, wodurch der Health-Check fehlschlagen kann. Falls Sliplane den Container während des Imports neu startet: Health-Check Grace Period erhöhen oder temporär deaktivieren.

### Workflow für Code-Änderungen

```
Feature-Branch  ->  Pull Request  ->  Review  ->  Merge in main  ->  Auto-Deploy Sliplane
```

Branch `feature/nlp-hausarbeit` aktiv in Entwicklung. Hauptstand: Tag `stable-2026-06-15` auf `main`.

---

## Umgebungsvariablen

| Variable | Pflicht | Beschreibung |
|---|---|---|
| `OPENAI_API_KEY` | ja | OpenAI-Key für 3Small/3Large-Embeddings, Semantic-Chunking und Antwortgenerierung |
| `ADMIN_USER_1` / `ADMIN_PASS_1` | ja | Login Svetlana |
| `ADMIN_USER_2` / `ADMIN_PASS_2` | nein | Login Ricarda |
| `WEAVIATE_URL` | nein | Standard `http://weaviate.internal:8080` auf Sliplane |
| `DATA_DIR` | nein | Pfad für Scraping-Cache, Standard `/app/data` |

Open-Source-Modelle (SBERT, E5) brauchen keinen API-Key, werden lokal im Container gerechnet.

---

## Projektstruktur

```
admin_app.py          Streamlit-Frontend (4 Tabs: Daten/Suche/Browser/Evaluation)
database_manager.py   Weaviate-Logik (Schema, Import mit Chunking+Embedding, Retrieval, Status)
chunking.py           Vier Chunking-Strategien (Fixed, Sentence, Recursive, Semantic)
embeddings.py         Embedding-Adapter OpenAI + Sentence-Transformers (mit E5-Prefix)
scraper.py            Web-Scraper mit Last-Modified-Pruefung
eval_ir.py            IR-Metriken Precision@5, MRR@5, NDCG@5
run_eval.py           Orchestrierung der Eval-Laeufe
docker-compose.yml    Weaviate + Streamlit als Docker-Stack
Dockerfile            Image fuer die Streamlit-App
requirements.txt      Python-Abhaengigkeiten (inkl. torch, sentence-transformers, langchain)
eval/
  goldstandard.json   Testdatensatz (30 Fragen, kategorisiert)
data/
  cache_*.json        Gecachte Scraping-Ergebnisse (via Volume persistiert)
```

---

## Import-Reihenfolge (Empfehlung)

Wegen der Rechenlast lokaler Modelle nicht direkt "ALLE importieren", sondern stufenweise:

1. `3Small / Fixed` (OpenAI, schnell) — prueft die Pipeline
2. `SBERT / Fixed` (erstes lokales Modell, laedt ~1 GB beim ersten Mal)
3. `E5Large / Fixed` (groesstes Modell, RAM-Stresstest)
4. Wenn alle drei laufen: restliche Collections oder "ALLE importieren" (Modelle sind dann gecacht)

Lokale Modelle vektorisieren auf der geteilten CPU langsam (Chunk fuer Chunk, einige Sekunden pro Chunk). Eine Collection mit ~850 Chunks dauert mehrere Minuten.

---

## Roadmap

- [x] Etappe 0: Sicherungs-Setup (Git-Tag + Feature-Branch)
- [x] Etappe 1: Goldstandard-Testdatensatz (30 Fragen, kategorisiert)
- [x] Etappe 2: Eval-Skripte (Precision@5, MRR@5, NDCG@5)
- [x] Etappe 4: Vier Chunking-Strategien
- [x] Etappe 5: SBERT + E5-Large Embeddings
- [ ] Etappe 6: Vollstaendige Eval-Laeufe auf allen 16 Collections
- [ ] Etappe 3: ragas-Pipeline (Generierungsqualitaet)
- [ ] Etappe 7: Hausarbeit schreiben

---

## Vorstudie

Aufbauend auf der Datenbanksysteme-Hausarbeit der Autorinnen (Januar 2026), die qualitativ den Einfluss von Chunking auf die Retrieval-Qualität untersucht hat. Die vorliegende NLP-Arbeit verallgemeinert diese Erkenntnis systematisch über mehrere Chunking-Strategien und Embedding-Modelle mit quantitativen Metriken.
