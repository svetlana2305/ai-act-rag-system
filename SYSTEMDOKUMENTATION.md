r Seminararbeit.

#### Was wird hier gemacht?

Verschiedene Suchmethoden und Collection-Konfigurationen werden mit denselben Testfragen verglichen. So kann man sagen: "BM25 findet bei dieser Frage Artikel 5, Semantisch findet Artikel 50 — welches ist das richtige?"

#### Schritt-für-Schritt Evaluierung

**1. Testfragen** (linke Seite):
- Es sind bereits 10 Fragen vorgegeben
- Du kannst sie ändern oder eigene hinzufügen
- Eine Frage pro Zeile

**2. Einstellungen** (rechte Seite):

*Top-K* — Wie viele Chunks soll die Suche pro Frage zurückgeben?
- Wert 1: Nur den besten Treffer
- Wert 3 (Standard): Die 3 besten Treffer
- Wert 5: Die 5 besten Treffer

*Suchmethoden* — Welche Methoden sollen verglichen werden?
- ☑ Semantisch (Vektor) — versteht Bedeutung
- ☑ BM25 (Keyword / TF-IDF) — zählt Wörter
- ☑ Hybrid (BM25 + Semantisch) — kombiniert

*Hybrid-Alpha* (erscheint nur wenn Hybrid gewählt):
- 0.0 = reines BM25
- 0.5 = ausgeglichen
- 1.0 = rein semantisch

*Collections* — Welche der 9 Konfigurationen sollen abgefragt werden?
- **"Alle"** Button: wählt alle 9 aus
- **"Keine"** Button: wählt alle ab
- Oder einzeln ankreuzen

**3. Evaluierung starten:**
- Klicke **"Evaluierung starten"**
- Fortschrittsanzeige läuft (z.B. "27/90 …")
- Je mehr Kombinationen, desto länger (alle 9 × alle 3 Methoden × 10 Fragen = 270 Abfragen)

**4. Ergebnisse lesen:**

*Einzelfragen-Ansicht:*
- Klappe jede Frage auf
- Pro Methode siehst du: bester Treffer, Artikel-Nummer, Score, Textvorschau

*Zusammenfassungstabelle:*
- Zeigt den Durchschnitts-Score pro Collection und Methode
- Achtung: BM25-Scores sind auf einer anderen Skala als Semantisch/Hybrid!
  - Semantisch: 0–1 (höher = ähnlicher)
  - BM25: unbegrenzt positiv (höher = relevanter)

**5. Protokoll herunterladen:**
- Klicke **"Protokoll als Markdown herunterladen"**
- Speichert eine `.md`-Datei mit allen Ergebnissen
- Diese Datei kann direkt in die Seminararbeit eingefügt werden

---

### Empfohlene Evaluierungsläufe für die Seminararbeit

#### Lauf 1: Methoden vergleichen
**Ziel:** Zeigen wie sich BM25, Semantisch und Hybrid unterscheiden

- Methoden: ☑ alle drei
- Collections: nur `3Large / 500 Token` (eine reicht)
- Top-K: 3
- → Protokoll als `eval_methoden.md` speichern

#### Lauf 2: Modelle vergleichen
**Ziel:** Zeigen ob ada-002, 3-small oder 3-large besser ist

- Methode: nur Semantisch
- Collections: `3Small/500`, `3Large/500`, `Ada/500` (alle mit gleicher Strategie)
- Top-K: 3
- → Protokoll als `eval_modelle.md` speichern

#### Lauf 3: Chunking-Strategien vergleichen
**Ziel:** Zeigen ob Artikel-weise, 500 Token oder 2000 Token besser ist

- Methode: nur Semantisch
- Collections: `3Large/Artikel`, `3Large/500`, `3Large/2000` (alle mit gleichem Modell)
- Top-K: 3
- → Protokoll als `eval_strategien.md` speichern

---

## 6. Fachbegriffe erklärt

### RAG — Retrieval-Augmented Generation
Wörtlich: "Antwortgenerierung mit Wissensabruf"

Das RAG-Prinzip funktioniert so:
1. **Retrieval (Abruf):** Relevante Textabschnitte aus der Datenbank suchen
2. **Augmented (Angereichert):** Diese Abschnitte als Kontext an ein Sprachmodell übergeben
3. **Generation:** Das Sprachmodell formuliert eine Antwort basierend auf den Abschnitten

**Warum RAG statt nur ChatGPT?** ChatGPT kennt den EU AI Act vielleicht grob, aber nicht die genauen aktuellen Texte. Mit RAG geben wir dem Modell den exakten Text — die Antworten werden präziser und nachvollziehbar ("laut Artikel 5 Absatz 1...").

### Embedding / Vektor
Ein Embedding ist ein Text der in Zahlen umgewandelt wurde — eine Liste von ~1500 Zahlen (bei text-embedding-3-small: 1536 Zahlen).

```
"Verbotene KI-Praktiken"  →  [-0.021, 0.044, -0.006, 0.034, ...]
                                        1536 Zahlen
```

Ähnliche Texte haben ähnliche Vektoren. "Manipulation" und "Täuschung" liegen im Vektorraum näher beieinander als "Manipulation" und "Haushalt".

### Kosinus-Ähnlichkeit
Wie die Ähnlichkeit zweier Vektoren gemessen wird. Ergebnis: 0 (komplett verschieden) bis 1 (identisch).

Die semantische Suche findet Texte mit hoher Kosinus-Ähnlichkeit zur Suchanfrage.

### BM25
"Best Match 25" — ein klassischer Suchalgorithmus aus den 1990ern.

Prinzip (vereinfacht):
- Seltene Wörter zählen mehr als häufige (TF-IDF-Prinzip)
- "das" erscheint überall → wenig wert
- "Biometrische Fernidentifizierung" erscheint selten → sehr wertvoll
- Kurze Dokumente werden leicht bevorzugt

**Zipfsches Gesetz** (aus der Vorlesung): In natürlicher Sprache ist das häufigste Wort ~doppelt so häufig wie das zweithäufigste, ~dreimal so häufig wie das dritte usw. BM25 nutzt diese Eigenschaft.

### Chunking
Der EU AI Act-Text ist zu lang um ihn als Ganzes mit einem Embedding darzustellen (Modelle haben ein Token-Limit). Deshalb teilen wir ihn in kleinere Abschnitte (Chunks).

**Chunking-Strategien im Vergleich:**

| Strategie | Vorteil | Nachteil |
|-----------|---------|----------|
| Artikel-weise | Ganzer Kontext erhalten | Zu lang für präzise Vektoren |
| 500 Token | Präzise Vektoren | Kontext kann abschneiden |
| 2000 Token | Guter Mittelweg | Etwas unschärfer als 500 |

### Token
OpenAI teilt Text nicht in Wörter, sondern in Tokens. Ein Token ist ca. 0,75 Wörter (auf Englisch; Deutsch etwas weniger da zusammengesetzte Wörter länger sind).

Beispiele:
- "Hochrisiko" = 3 Tokens
- "KI" = 1 Token
- "Konformitätsbewertungsverfahren" = 6 Tokens

### Hybrid-Alpha
Der Alpha-Parameter steuert die Mischung bei der Hybrid-Suche:
- α = 0.0 → 100% BM25, 0% Semantisch
- α = 0.5 → 50% BM25, 50% Semantisch
- α = 1.0 → 0% BM25, 100% Semantisch

Für die Seminararbeit interessant: Gibt es einen Alpha-Wert der besser als die reinen Methoden ist?

### Collection
Ein "Ordner" in Weaviate. Jede unserer 9 Kombinationen (Modell × Strategie) hat eine eigene Collection. Vergleichbar mit einer Tabelle in einer normalen Datenbank.

### Schema
Die Struktur einer Collection — welche Felder existieren und welchen Typ haben sie (Text, Zahl, Referenz...). Muss einmalig definiert werden bevor Daten hineingelegt werden können.

### Weaviate
Open-Source Vektordatenbank aus den Niederlanden. Besonderheit: Sie vektorisiert automatisch beim Einfügen und sucht nach Vektorähnlichkeit. Wir müssen keine eigene Suchlogik schreiben.

---

## 7. Häufige Fehler und Lösungen

### "Weaviate nicht erreichbar" (roter Hinweis)

**Ursache:** Der Weaviate-Docker-Container läuft nicht.

**Lösung:**
1. Auf Sliplane einloggen
2. Den Weaviate-Dienst prüfen und neu starten
3. Seite im Browser neu laden
4. Jetzt sollte "Weaviate verbunden" erscheinen

### Import schlägt fehl mit "OpenAI"-Fehlermeldung

**Ursache:** Der OpenAI-API-Key fehlt oder ist ungültig.

**Lösung:**
1. Sliplane → Umgebungsvariablen
2. Prüfen ob `OPENAI_API_KEY` gesetzt ist
3. Den Schlüssel ggf. neu eingeben (beginnt mit `sk-`)

### "Keine Dokumente bereit" — Import-Button ausgegraut

**Ursache:** Noch kein Scraping durchgeführt.

**Lösung:** Zuerst Tab "Daten" → "EU AI Act scrapen" klicken.

### Evaluierung zeigt "Art. 0" bei Suchergebnissen

**Ursache:** Alte Collection-Daten vor einem Schema-Fix.

**Lösung:**
1. Tab "Daten" → rechte Spalte → "Schema verwalten" aufklappen
2. "Schema löschen" klicken
3. Neu importieren mit "In Weaviate importieren"

### Seite reagiert nicht mehr / dreht sich endlos

**Ursache:** Streamlit wird neu gestartet (nach Code-Updates) oder der Server ist überlastet.

**Lösung:** Browser-Seite neu laden (`F5` oder `Strg+R`). Danach neu einloggen.

### Scraping bricht nach wenigen Artikeln ab

**Ursache:** Netzwerk-Timeout oder die Webseite ist vorübergehend nicht erreichbar.

**Lösung:** Einfach nochmal auf "EU AI Act scrapen" klicken. Das System nutzt automatisch bereits gespeicherte Daten aus dem Cache — nur die fehlenden werden nachgeladen.

### "Mindestens eine Collection auswählen"

**Ursache:** Im Evaluierungsbereich wurde keine Collection angekreuzt.

**Lösung:** Collections ankreuzen oder den **"Alle"**-Button klicken.

---

## 8. Bezug zur Vorlesung

Diese Tabelle hilft beim Schreiben der Seminararbeit — die Verbindung zwischen unserem Code und den Vorlesungsinhalten:

| Vorlesungsinhalt (Prof. Gawron) | Wo im System | Code-Stelle |
|----------------------------------|-------------|-------------|
| **BOW / TF-IDF** (NLP_01) | BM25-Suche | `col.query.bm25()` in `database_manager.py` |
| **Zipfsches Gesetz** (NLP_02) | BM25-Scoring: seltene Wörter zählen mehr | Weaviate intern |
| **Word Embeddings** (NLP_02) | Semantische Suche mit Vektoren | `col.query.near_text()` |
| **Transformer / BERT** (NLP_03) | OpenAI text-embedding-3 basiert auf Transformer-Architektur | `MODELS` Dict in `database_manager.py` |
| **RAG** (NLP_06) | Gesamtsystem: Suche → Kontext → Antwort | Evaluierungstab + spätere Antwortgenerierung |
| **Tokenisierung** (NLP_01) | Chunking mit tiktoken | `split_tokens()` in `database_manager.py` |
| **Semantische Ähnlichkeit** | Kosinus-Distanz in Weaviate | `MetadataQuery(distance=True)` |
| **Hybride Retrieval-Ansätze** | Hybrid-Suche (alpha-Parameter) | `col.query.hybrid()` |

### Empfohlene Formulierungen für die Seminararbeit

**Zum Chunking:**
> "Der EU AI Act-Text wurde in drei Granularitätsstufen aufgeteilt: artikelweise (kein Splitting), in Segmente von 500 Token mit 50 Token Überlapp sowie in Segmente von 2000 Token mit 200 Token Überlapp. Die Überlappung verhindert Informationsverluste an Segmentgrenzen."

**Zur Evaluierung:**
> "Drei Retrievalmethoden wurden verglichen: die klassische BM25-Suche (nach dem TF-IDF-Prinzip, Zipfsches Gesetz), die vektorbasierte semantische Suche (Dense Embeddings via OpenAI text-embedding-3) sowie eine hybride Kombination beider Ansätze."

**Zu den Modellen:**
> "Für die Vektorisierung wurden drei OpenAI-Embedding-Modelle verglichen: text-embedding-ada-002 (ältere Generation), text-embedding-3-small (aktuell, kostengünstig) und text-embedding-3-large (aktuell, höhere Dimensionalität)."

---

*Dokument erstellt für das Seminararbeits-Team · Letzte Aktualisierung: Mai 2026*
