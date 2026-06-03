# Anleitung — EU AI Act RAG Admin-System

Seminararbeit · Prof. Dr. Christian Gawron · FH Südwestfalen  
Team: Svetlana (Infrastruktur) + Ricarda (Chunking/Evaluierung)

---

## Übersicht

Das System besteht aus zwei Diensten auf Sliplane:

| Dienst | Zugang | Zweck |
|--------|--------|-------|
| Streamlit Admin-App | öffentliche URL, Login erforderlich | Daten verwalten, evaluieren |
| Weaviate | intern (Port 8080, nicht öffentlich) | Vektordatenbank |

Die Admin-App ist in drei Tabs aufgeteilt: **Daten → Datenbank durchsuchen → Evaluierung**

---

## Schritt 1 — Erstmaliges Setup (einmalig)

### 1.1 Anmelden

Öffne die Sliplane-URL der Streamlit-App. Du siehst ein Login-Formular.  
Benutzerdaten sind in den Sliplane-Umgebungsvariablen hinterlegt:
- `ADMIN_USER_1` / `ADMIN_PASS_1`
- `ADMIN_USER_2` / `ADMIN_PASS_2`

### 1.2 Verbindung prüfen

In der linken Seitenleiste steht entweder **"Weaviate verbunden"** (grün) oder **"Weaviate nicht erreichbar"** (rot).  
Nur wenn Weaviate verbunden ist, können Daten importiert und abgefragt werden.

---

## Schritt 2 — Daten scrapen (Tab "Daten", linke Spalte)

Klicke auf **"EU AI Act scrapen"**.

Das System lädt automatisch:
- 113 Artikel (`/de/article/1/` bis `/de/article/113/`)
- 13 Anhänge (`/de/annex/1/` bis `/de/annex/13/`)
- 180 Erwägungsgründe (`/de/recital/1/` bis `/de/recital/180/`)

**Fortschritt:** Während des Scrapens wird pro Dokumenttyp live der Fortschritt angezeigt (z.B. "Artikel: 45/113").

**Cache:** Die heruntergeladenen Texte werden lokal gespeichert. Bei erneutem Klick prüft das System per `Last-Modified`-Header ob eine Aktualisierung nötig ist. Wenn die Seite unverändert ist, werden die Daten aus dem Cache geladen (schnell).

**Force-Rescrape:** Wenn du sicher neu laden willst (z.B. nach einem Scraper-Fix), setze das Häkchen bei **"Cache leeren und neu scrapen"** vor dem Klicken.

**Fertig wenn:** Grüne Meldung mit "306 Dokumente bereit — 113 Artikel, 13 Anhänge, 180 Erwägungsgründe".

---

## Schritt 3 — In Weaviate importieren (Tab "Daten", rechte Spalte)

Für jede Collection (Modell × Chunking-Strategie) einmal durchführen.  
Es gibt **9 Collections** (3 Modelle × 3 Strategien):

| Modell | Beschreibung |
|--------|-------------|
| `3Small` | text-embedding-3-small (günstig, schnell) |
| `3Large` | text-embedding-3-large (teurer, präziser) |
| `Ada` | text-embedding-ada-002 (älteres Modell, Vergleich) |

| Strategie | Beschreibung |
|-----------|-------------|
| `Artikel` | Ganzes Dokument als ein Chunk (kein Splitting) |
| `500` | 500 Tokens, 50 Token Überlapp |
| `2000` | 2000 Tokens, 200 Token Überlapp |

### Vorgehen

1. Modell und Strategie aus den Dropdown-Menüs wählen
2. Auf **"In Weaviate importieren"** klicken
3. Warten bis "Import abgeschlossen" erscheint
4. Wiederholen für alle 9 Kombinationen

**Achtung:** Jeder Import kostet OpenAI-API-Guthaben (Embeddings). Für alle 9 Collections mit 306 Dokumenten fallen ca. 2–5 USD an (je nach Modell).

**Schema verwalten:** Unter dem aufklappbaren Bereich "Schema verwalten" kannst du einzelne Collections löschen und neu aufbauen — nützlich wenn etwas schiefgelaufen ist.

**Tipp:** Linke Seitenleiste zeigt nach dem Import die Chunk-Zahlen pro Collection.

---

## Schritt 4 — Datenbank prüfen (Tab "Datenbank durchsuchen")

Hier kannst du einzelne Collections durchsuchen und die importierten Dokumente sowie ihre Chunks einsehen.

1. Modell und Strategie wählen
2. Anzahl der anzuzeigenden Dokumente einstellen (Slider)
3. **"Laden"** klicken
4. Jedes Dokument aufklappen → Volltext sichtbar
5. **"Chunks laden"** klicken → alle Chunks dieses Artikels sehen

---

## Schritt 5 — Evaluierung (Tab "Evaluierung")

Hier findet der wissenschaftliche Vergleich statt. Dies ist der Kern der Seminararbeit.

### 5.1 Konfiguration

**Testfragen:** 10 vorausgefüllte Fragen zum EU AI Act. Du kannst eigene Fragen hinzufügen oder bestehende ändern (eine pro Zeile).

**Top-K:** Wie viele Chunks pro Frage abgerufen werden (Standard: 3).

**Suchmethoden** (eine oder mehrere ankreuzen):
- **Semantisch (Vektor):** Dense Embeddings, versteht inhaltliche Ähnlichkeit
- **BM25 (Keyword / TF-IDF):** Klassische Keyword-Suche, bevorzugt seltene relevante Wörter (Zipfsches Gesetz)
- **Hybrid (BM25 + Semantisch):** Kombination beider Ansätze, Alpha-Slider steuert Gewichtung

Beim Alpha-Slider: `0.0` = reines BM25, `1.0` = rein semantisch, `0.5` = gleichgewichtet.

**Collections:** Welche der 9 Collections abgefragt werden. Nutze **"Alle"** / **"Keine"** um schnell alle aus- oder abzuwählen.

### 5.2 Evaluierung starten

Klicke auf **"Evaluierung starten"**. Der Fortschritt wird angezeigt.

**Achtung bei vielen Kombinationen:** 10 Fragen × 9 Collections × 3 Methoden = 270 Abfragen. Das dauert mehrere Minuten und kostet OpenAI-Guthaben für die semantischen Abfragen.

**Empfehlung für ersten Test:** Erst mit 1–2 Collections und einer Methode testen.

### 5.3 Ergebnisse lesen

Nach dem Lauf:

- **Einzelfragen aufklappen:** Zeigt pro Methode den besten Treffer je Collection mit Score und Chunk-Vorschau
- **Zusammenfassungstabelle:** Ø-Score je Collection und Methode nebeneinander

**Hinweis zu den Scores:**
- Semantisch/Hybrid: Score zwischen 0 und 1 (höher = ähnlicher)
- BM25: Rohwert (unbegrenzt positiv, höher = relevanter) — **nicht direkt mit den anderen Werten vergleichbar**
- Wenn BM25 zusammen mit anderen Methoden gewählt ist, erscheint automatisch ein Hinweis

### 5.4 Protokoll exportieren

Klicke auf **"Protokoll als Markdown herunterladen"**.

Die `.md`-Datei enthält:
- Alle Konfigurationsparameter (Datum, Collections, Methoden, Top-K)
- Tabelle je Frage und Methode mit bestem Treffer
- Zusammenfassung mit Ø-Scores
- Hinweis zur Score-Vergleichbarkeit (falls zutreffend)

Diese Datei kann direkt in die Seminararbeit übernommen werden.

---

## Häufige Probleme

| Problem | Ursache | Lösung |
|---------|---------|--------|
| "Weaviate nicht erreichbar" | Weaviate-Service gestartet? | In Sliplane prüfen ob der Weaviate-Dienst läuft |
| Import schlägt fehl | OpenAI-Key fehlt oder ungültig | `OPENAI_API_KEY` in Sliplane-Umgebungsvariablen prüfen |
| Keine Dokumente nach Scrape | Webseite nicht erreichbar | Erneut versuchen, ggf. Force-Rescrape aktivieren |
| Evaluierung zeigt "Art. 0" | Alte Collection vor dem Bug-Fix | Collection löschen, neu importieren |
| Streamlit-Fehler nach Neustart | Session abgelaufen | Seite neu laden und neu einloggen |

---

## Für die Seminararbeit

### Empfohlene Evaluierungsläufe

1. **Methoden-Vergleich:** Eine Collection (z.B. `3Large / 500 Tokens`), alle drei Methoden → zeigt BM25 vs. Semantisch vs. Hybrid
2. **Modell-Vergleich:** Eine Methode (Semantisch), alle drei Modelle mit gleicher Strategie → zeigt ada vs. 3-small vs. 3-large
3. **Strategie-Vergleich:** Ein Modell (3Large), eine Methode (Semantisch), alle drei Strategien → zeigt Artikel-weise vs. 500 vs. 2000 Token

Für jeden Lauf ein Markdown-Protokoll herunterladen und als Anhang zur Arbeit beifügen.

### Bezug zur Vorlesung (Prof. Gawron)

| Vorlesungsinhalt | Entsprechung im System |
|-----------------|------------------------|
| BOW / TF-IDF / Zipfsches Gesetz (NLP_01/02) | BM25-Suchmethode |
| Dense Embeddings / Word2Vec → OpenAI (NLP_02) | Semantische Suchmethode |
| Transformer / BERT (NLP_03) | OpenAI text-embedding-3 Modelle |
| RAG (NLP_06) | Gesamtsystem: Retrieval + Weaviate |
