# EU AI Act RAG-System

Seminararbeit im Masterstudiengang bei Prof. Dr. Christian Gawron.

**Thema:** Einfluss von Chunking-Strategie und Embedding-Modell auf die Retrieval-Qualität in RAG-Systemen am Beispiel des EU AI Act.

**Team:** Svetlana (Infrastruktur, Deployment, Embedding) · Ricarda (Chunking, Evaluation, Testdatensatz)

---

## Worum geht es?

Das System scrapt den EU AI Act, zerlegt ihn in Textabschnitte (Chunks) und speichert diese mit drei verschiedenen OpenAI-Embedding-Modellen parallel in Weaviate. So lässt sich später direkt vergleichen, welches Modell beim Retrieval am besten abschneidet.

**Collections in Weaviate:**
- `AIAct_OpenAI_3Small` → text-embedding-3-small
- `AIAct_OpenAI_3Large` → text-embedding-3-large
- `AIAct_OpenAI_Ada` → text-embedding-ada-002

Jede Collection hat ein `Article`- und ein `Chunk`-Objekt. Chunks referenzieren über `ofArticle` auf ihren Artikel — so hat das LLM später nicht nur den Chunk, sondern den vollen Kontext.

---

## Lokal starten

**Voraussetzung:** Docker Desktop

```bash
git clone https://github.com/voelker-consulting/ai-act-rag-system.git
cd ai-act-rag-system
cp .env.example .env
# .env befüllen (OPENAI_API_KEY, ADMIN_PASSWORD)
docker compose up --build
```

- Admin-App: http://localhost:8501
- Weaviate: http://localhost:8080

---

## Erste Nutzung

1. http://localhost:8501 öffnen und anmelden
2. Sidebar prüfen: Weaviate sollte als "verbunden" angezeigt werden
3. **Schritt 1:** "Web-Update prüfen & scrapen" — lädt den EU AI Act herunter
4. **Schritt 2:** Collection auswählen (Modell) → "In Weaviate importieren"

Den Vorgang für alle drei Modelle wiederholen, um parallele Collections zu befüllen.

---

## Deployment auf Sliplane

1. Repository in Sliplane verbinden: *Add Service → From GitHub → ai-act-rag-system*
2. Sicherstellen, dass Sliplane **docker-compose.yml** als Quelle erkennt (nicht Dockerfile). Im Dashboard sollten nach dem Deploy zwei Services laufen: `weaviate` und `streamlit-app`.
3. Umgebungsvariablen setzen:

| Variable | Wert |
|---|---|
| `OPENAI_API_KEY` | sk-... |
| `ADMIN_PASSWORD` | frei wählbar |

`WEAVIATE_URL` muss nicht gesetzt werden — Standard ist `http://weaviate:8080`.

---

## Umgebungsvariablen

| Variable | Pflicht | Standard |
|---|---|---|
| `OPENAI_API_KEY` | ja | — |
| `ADMIN_PASSWORD` | ja | — |
| `WEAVIATE_URL` | nein | `http://weaviate:8080` |
| `DATA_DIR` | nein | `data/` |

---

## Projektstruktur

```
admin_app.py         Streamlit-Frontend (Login, Scrape, Import)
database_manager.py  Weaviate-Logik (Schema, Import, Status)
scraper.py           Web-Scraper mit Last-Modified-Prüfung
docker-compose.yml   Weaviate + Streamlit als Docker-Stack
Dockerfile           Image für die Streamlit-App
requirements.txt     Python-Abhängigkeiten
.env.example         Vorlage für die .env-Datei
```

---

## Für Ricarda

Repository klonen und lokal starten wie oben beschrieben. Die `.env` muss selbst befüllt werden (OpenAI-Key und Passwort).

Für Änderungen am Code: normaler Git-Workflow über `main`. Svetlana deployt auf Sliplane.
