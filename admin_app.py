import datetime
import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from database_manager import (
    MODELS, STRATEGIES, collection_prefix,
    browse_collection, delete_collection, get_chunks_for_article,
    get_client, get_status, import_articles, retrieve_chunks, setup_collection,
)
from scraper import check_and_scrape, clear_cache, load_cached_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

st.set_page_config(page_title="EU AI Act RAG", page_icon="⚖️", layout="wide")

# Drei konfigurierbare Nutzerkonten über Umgebungsvariablen
USERS = {}
for _i in ("1", "2", "3"):
    _u = os.getenv(f"ADMIN_USER_{_i}", "")
    _p = os.getenv(f"ADMIN_PASS_{_i}", "")
    if _u and _p:
        USERS[_u] = _p


# Login — Weaviate wird hier bewusst NICHT initialisiert, damit die Seite sofort lädt

if not st.session_state.get("auth"):
    st.title("⚖️ EU AI Act RAG — Admin")
    st.markdown("---")
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login"):
            username = st.text_input("Benutzername")
            pw = st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden", use_container_width=True):
                if not USERS:
                    st.error("Keine Benutzerkonten konfiguriert (ADMIN_USER_1/ADMIN_PASS_1 fehlen).")
                elif USERS.get(username) == pw:
                    st.session_state["auth"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Unbekannter Benutzername oder falsches Passwort.")
    st.stop()


# Ab hier: nur eingeloggte Nutzer

@st.cache_resource(show_spinner=False)
def weaviate_client():
    return get_client()


client = weaviate_client()

# Beim Start aus dem lokalen Cache laden, falls vorhanden
if "documents" not in st.session_state:
    cached = load_cached_corpus()
    if cached:
        st.session_state["documents"] = cached


# Sidebar

with st.sidebar:
    st.title("⚙️ Einstellungen")
    st.markdown("---")

    try:
        ready = client.is_ready()
    except Exception:
        ready = False

    if ready:
        st.success("Weaviate verbunden")
    else:
        st.error("Weaviate nicht erreichbar")

    st.markdown("---")

    # Status aller 9 Collections (3 Modelle × 3 Strategien)
    st.subheader("Collections")
    if ready:
        try:
            status = get_status(client)
            for model_key in MODELS:
                st.markdown(f"**{model_key}**")
                for strategy_key in STRATEGIES:
                    counts = status.get((model_key, strategy_key), {"documents": 0, "chunks": 0})
                    if counts["documents"] > 0:
                        st.caption(
                            f"  {strategy_key}: {counts['documents']} Dok., {counts['chunks']} Chunks"
                        )
                    else:
                        st.caption(f"  {strategy_key}: leer")
        except Exception as e:
            st.caption(f"Status-Fehler: {e}")

    st.markdown("---")

    logged_in_as = st.session_state.get("username", "")
    if logged_in_as:
        st.caption(f"Eingeloggt als: **{logged_in_as}**")

    if st.button("Abmelden", use_container_width=True):
        st.session_state["auth"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.markdown("---")
    st.caption("Version 0.2 · Stand: 03.06.2026")


# Hauptbereich — Tabs

st.title("⚖️ EU AI Act — RAG Admin")
st.caption("Seminararbeit · Masterstudiengang · Prof. Dr. Christian Gawron")
st.markdown("---")

tab_daten, tab_db, tab_eval = st.tabs(["Daten", "Datenbank durchsuchen", "Evaluierung"])


# ── Tab 1: Scrapen + Importieren ──────────────────────────────────────────────

with tab_daten:
    col_scrape, col_import = st.columns(2, gap="large")

    with col_scrape:
        st.subheader("Schritt 1 — Scrapen")
        st.markdown(
            "Lädt alle drei Dokumenttypen von `artificialintelligenceact.eu` herunter:\n"
            "- **113 Artikel** (`/de/article/X/`)\n"
            "- **13 Anhänge** (`/de/annex/X/`)\n"
            "- **180 Erwägungsgründe** (`/de/recital/X/`)\n\n"
            "Pro Typ wird per `Last-Modified` geprüft ob ein Update nötig ist."
        )

        force_rescrape = st.checkbox(
            "Cache leeren und neu scrapen",
            help="Erzwingt einen vollständigen Neu-Download aller Dokumente, "
                 "auch wenn laut Last-Modified keine Änderung erkannt wird.",
        )

        if st.button("EU AI Act scrapen", type="primary", use_container_width=True):
            if force_rescrape:
                clear_cache()
            with st.status("Scrape läuft …", expanded=True) as s:
                try:
                    # Live-Fortschrittsanzeige pro Dokumenttyp
                    ph = {
                        "artikel":         st.empty(),
                        "anhang":          st.empty(),
                        "erwaegungsgrund": st.empty(),
                    }
                    labels = {
                        "artikel":         "Artikel",
                        "anhang":          "Anhänge",
                        "erwaegungsgrund": "Erwägungsgründe",
                    }
                    totals = {"artikel": 113, "anhang": 13, "erwaegungsgrund": 180}

                    for src_type, placeholder in ph.items():
                        placeholder.write(f"{labels[src_type]}: 0/{totals[src_type]}")

                    def on_progress(src_type, _label, nr, total):
                        ph[src_type].write(f"{labels[src_type]}: {nr}/{total}")

                    scraped, docs = check_and_scrape(progress_callback=on_progress, force=force_rescrape)
                    st.session_state["documents"] = docs

                    arten = {}
                    for d in docs:
                        arten[d["source_type"]] = arten.get(d["source_type"], 0) + 1

                    if scraped:
                        s.update(label=f"Fertig — {len(docs)} Dokumente geladen", state="complete")
                    else:
                        s.update(label=f"Aus Cache — {len(docs)} Dokumente bereit", state="complete")
                except Exception as e:
                    st.write(f"Fehler: {e}")
                    s.update(label="Fehler", state="error")

        if st.session_state.get("documents"):
            docs = st.session_state["documents"]
            arten = {}
            for d in docs:
                arten[d["source_type"]] = arten.get(d["source_type"], 0) + 1
            st.success(
                f"{len(docs)} Dokumente bereit — "
                f"{arten.get('artikel', 0)} Artikel, "
                f"{arten.get('anhang', 0)} Anhänge, "
                f"{arten.get('erwaegungsgrund', 0)} Erwägungsgründe"
            )
            with st.expander("Vorschau (erste 5)"):
                for d in docs[:5]:
                    st.markdown(f"**{d['title']}** `{d['source_type']}`")
                    st.caption(d["full_text"][:300] + ("…" if len(d["full_text"]) > 300 else ""))

    with col_import:
        st.subheader("Schritt 2 — In Weaviate importieren")

        model_key = st.selectbox(
            "Embedding-Modell",
            options=list(MODELS.keys()),
            format_func=lambda k: f"{k}  —  {MODELS[k]}",
        )

        strategy_key = st.selectbox(
            "Chunking-Strategie",
            options=list(STRATEGIES.keys()),
            format_func=lambda k: STRATEGIES[k]["label"],
        )

        target = collection_prefix(model_key, strategy_key)
        st.info(f"Ziel-Collection: `{target}`")

        has_docs = bool(st.session_state.get("documents"))
        if not has_docs:
            st.caption("Zuerst Dokumente scrapen.")

        with st.expander("Schema verwalten"):
            st.warning("Löschen entfernt alle Daten dieser Collection unwiderruflich.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Schema erstellen", use_container_width=True):
                    try:
                        setup_collection(client, model_key, strategy_key)
                        st.success("Erstellt.")
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(str(e))
            with c2:
                if st.button("Schema löschen", use_container_width=True):
                    try:
                        delete_collection(client, model_key, strategy_key)
                        st.success("Gelöscht.")
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(str(e))

        if st.button("In Weaviate importieren", type="primary", use_container_width=True, disabled=not has_docs):
            documents = st.session_state.get("documents", [])
            bar = st.progress(0, text="Starte …")
            try:
                with st.status("Importiere …", expanded=True) as s:
                    st.write("Schema sicherstellen …")
                    setup_collection(client, model_key, strategy_key)
                    bar.progress(20, text="Schema bereit …")

                    strategie_label = STRATEGIES[strategy_key]["label"]
                    st.write(f"Importiere {len(documents)} Dokumente mit Strategie '{strategie_label}' …")
                    result = import_articles(client, documents, model_key, strategy_key)
                    bar.progress(100, text="Fertig.")

                    msg = f"{result['documents']} Dokumente, {result['chunks']} Chunks importiert."
                    st.write(msg)
                    s.update(label="Import abgeschlossen", state="complete")

                st.success(msg)
                st.cache_resource.clear()

            except Exception as e:
                bar.empty()
                st.error(f"Import fehlgeschlagen: {e}")
                logging.exception("Import-Fehler")


# ── Tab 2: Datenbank-Browser ──────────────────────────────────────────────────

with tab_db:
    st.subheader("Datenbank durchsuchen")

    if not ready:
        st.error("Weaviate nicht erreichbar.")
    else:
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            db_model = st.selectbox(
                "Modell",
                options=list(MODELS.keys()),
                format_func=lambda k: f"{k}  —  {MODELS[k]}",
                key="db_model",
            )
        with db_col2:
            db_strategy = st.selectbox(
                "Strategie",
                options=list(STRATEGIES.keys()),
                format_func=lambda k: STRATEGIES[k]["label"],
                key="db_strategy",
            )

        db_limit = st.slider("Anzahl Dokumente", 5, 50, 20, 5, key="db_limit")

        # Cache-Schlüssel enthält Modell, Strategie und Limit — verhindert veraltete Anzeige
        db_cache_key = f"db_rows_{db_model}_{db_strategy}_{db_limit}"

        if st.button("Laden", use_container_width=True):
            with st.spinner("Lade Dokumente …"):
                try:
                    rows = browse_collection(client, db_model, db_strategy, limit=db_limit)
                    st.session_state[db_cache_key] = rows
                except Exception as e:
                    st.error(f"Fehler: {e}")

        rows = st.session_state.get(db_cache_key, [])
        if rows:
            st.caption(f"{len(rows)} Dokumente aus `{collection_prefix(db_model, db_strategy)}`")
            for row in rows:
                chunk_info = f"{row['chunk_count']} Chunks" if row["chunk_count"] else "keine Chunks"
                label = (
                    f"**{row['title']}** — `{row['source_type']}` "
                    f"Nr. {row['article_number']} · {chunk_info}"
                )
                with st.expander(label):
                    st.markdown("**Volltext:**")
                    st.text_area(
                        "Volltext",
                        value=row["full_text"],
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"text_{row['uuid']}",
                    )

                    if st.button(f"Chunks laden", key=f"chunks_{row['uuid']}"):
                        try:
                            chunks = get_chunks_for_article(
                                client, db_model, db_strategy, row["uuid"]
                            )
                            if chunks:
                                st.markdown(f"**{len(chunks)} Chunks:**")
                                for c in chunks:
                                    st.markdown(
                                        f"*Chunk {c['index'] + 1}/{c['total']}:* {c['content'][:300]}…"
                                    )
                            else:
                                st.info("Keine Chunks gefunden.")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
        elif st.session_state.get("db_rows") is not None:
            st.info("Collection ist leer oder existiert noch nicht.")


# ── Tab 3: Evaluierung ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_goldstandard() -> dict:
    """Lädt data/goldstandard.json und gibt ein Dict {frage: [artikel_nummern]} zurück."""
    path = os.path.join(os.path.dirname(__file__), "data", "goldstandard.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        entry["frage"]: [int(a) for a in entry["relevante_artikel"]]
        for entry in data.get("fragen", [])
    }


def _precision_at_k(hits: list, relevant_articles: list) -> float:
    """Anteil der gefundenen Chunks, deren Artikel-Nummer in relevant_articles enthalten ist."""
    if not hits or not relevant_articles:
        return 0.0
    relevant_set = set(relevant_articles)
    correct = sum(1 for h in hits if int(h.get("article_number", 0)) in relevant_set)
    return round(correct / len(hits), 4)


_STANDARD_FRAGEN = """\
Welche KI-Systeme sind laut EU AI Act vollständig verboten?
Welche Pflichten haben Anbieter von Hochrisiko-KI-Systemen?
Was sind die Transparenzanforderungen für KI-Systeme?
Wie werden KI-Systeme mit hohem Risiko klassifiziert?
Welche Sanktionen drohen bei Verstößen gegen den EU AI Act?
Welche Anforderungen gelten für Trainingsdaten von Hochrisiko-KI?
Welche Rolle spielt menschliche Aufsicht bei KI-Systemen?
Was sind Allzweck-KI-Modelle und welche Pflichten gelten für sie?
Wie läuft das Konformitätsbewertungsverfahren ab?
Welche Rechte haben betroffene Personen gegenüber automatisierten Entscheidungen?\
"""

_METHODEN = {
    "semantic": "Semantisch (Vektor)",
    "bm25":     "BM25 (Keyword / TF-IDF)",
    "hybrid":   "Hybrid (BM25 + Semantisch)",
}

with tab_eval:
    st.subheader("Evaluierung — Retrieval-Vergleich")
    st.markdown(
        "Testfragen werden gegen ausgewählte Collections und Suchmethoden gestellt. "
        "Drei Methoden im Vergleich: **BM25** (klassisch, Zipfsches Gesetz / TF-IDF-Prinzip), "
        "**Semantisch** (Dense Embeddings, OpenAI) und **Hybrid** (beide kombiniert)."
    )

    if not ready:
        st.error("Weaviate nicht erreichbar.")
    else:
        ev_col1, ev_col2 = st.columns([2, 1])

        with ev_col1:
            fragen_text = st.text_area(
                "Testfragen (eine pro Zeile)",
                value=_STANDARD_FRAGEN,
                height=200,
                key="ev_fragen",
            )

        with ev_col2:
            ev_top_k = st.slider("Top-K Chunks pro Frage", 1, 5, 3, key="ev_topk")

            st.markdown("**Suchmethoden:**")
            ev_methoden = []
            for mkey, mlabel in _METHODEN.items():
                if st.checkbox(mlabel, key=f"ev_m_{mkey}", value=(mkey == "semantic")):
                    ev_methoden.append(mkey)

            ev_alpha = 0.5
            if "hybrid" in ev_methoden:
                ev_alpha = st.slider(
                    "Hybrid-Alpha (0 = BM25, 1 = Semantisch)",
                    0.0, 1.0, 0.5, 0.1,
                    key="ev_alpha",
                )

            st.markdown("**Collections:**")
            btn_a, btn_k = st.columns(2)
            with btn_a:
                if st.button("Alle", key="ev_alle", use_container_width=True):
                    for _mk in MODELS:
                        for _sk in STRATEGIES:
                            st.session_state[f"ev_{_mk}_{_sk}"] = True
            with btn_k:
                if st.button("Keine", key="ev_keine", use_container_width=True):
                    for _mk in MODELS:
                        for _sk in STRATEGIES:
                            st.session_state[f"ev_{_mk}_{_sk}"] = False

            ev_selected = []
            for mk in MODELS:
                for sk in STRATEGIES:
                    col_name = collection_prefix(mk, sk)
                    label = f"{mk} / {STRATEGIES[sk]['label']}"
                    if st.checkbox(label, key=f"ev_{mk}_{sk}"):
                        ev_selected.append((mk, sk, col_name))

        if st.button("Evaluierung starten", type="primary", use_container_width=True):
            fragen = [f.strip() for f in fragen_text.splitlines() if f.strip()]
            if not fragen:
                st.warning("Keine Fragen eingegeben.")
            elif not ev_methoden:
                st.warning("Mindestens eine Suchmethode auswählen.")
            elif not ev_selected:
                st.warning("Mindestens eine Collection auswählen.")
            else:
                # Ergebnisse: {frage: {(col_name, method): [hits]}}
                ergebnisse = {}
                gesamt = len(fragen) * len(ev_selected) * len(ev_methoden)
                bar = st.progress(0, text="Starte …")
                schritt = 0

                for frage in fragen:
                    ergebnisse[frage] = {}
                    for mk, sk, col_name in ev_selected:
                        for mkey in ev_methoden:
                            try:
                                hits = retrieve_chunks(
                                    client, mk, sk, frage,
                                    top_k=ev_top_k,
                                    method=mkey,
                                    hybrid_alpha=ev_alpha,
                                )
                            except Exception:
                                hits = []
                            ergebnisse[frage][(col_name, mkey)] = hits
                            schritt += 1
                            bar.progress(schritt / gesamt, text=f"{schritt}/{gesamt} …")

                bar.empty()
                st.session_state["ev_ergebnisse"] = ergebnisse
                st.session_state["ev_selected"]   = ev_selected
                st.session_state["ev_methoden"]   = ev_methoden
                st.session_state["ev_alpha_last"]  = ev_alpha
                st.session_state["ev_top_k"]      = ev_top_k

        # ── Ergebnisse anzeigen ──────────────────────────────────────────────

        ev_res     = st.session_state.get("ev_ergebnisse", {})
        ev_cols    = st.session_state.get("ev_selected", [])
        ev_methods = st.session_state.get("ev_methoden", [])
        ev_a       = st.session_state.get("ev_alpha_last", 0.5)
        ev_k       = st.session_state.get("ev_top_k", 3)

        if ev_res and ev_cols and ev_methods:
            st.markdown("---")
            methoden_str = " · ".join(_METHODEN[m] for m in ev_methods)
            st.markdown(f"**{len(ev_res)} Fragen · {len(ev_cols)} Collections · {methoden_str}**")

            for frage, key_hits in ev_res.items():
                with st.expander(f"❓ {frage}"):
                    for mkey in ev_methods:
                        st.markdown(f"##### {_METHODEN[mkey]}")
                        for mk, sk, col_name in ev_cols:
                            hits = key_hits.get((col_name, mkey), [])
                            label = f"{mk} / {STRATEGIES[sk]['label']}"
                            if hits:
                                bester = hits[0]
                                score_val = bester["score"]
                                vorschau = bester["content"][:160].replace("\n", " ")
                                st.caption(
                                    f"**{label}** — Art. {bester['article_number']} "
                                    f"({bester['title']}) Score: **{score_val}** — {vorschau}…"
                                )
                            else:
                                st.caption(f"**{label}** — keine Treffer")

            # ── Zusammenfassungstabelle ──────────────────────────────────────

            # BM25-Rohwert und Kosinus-Ähnlichkeit sind verschiedene Skalen.
            # Spalte = Methodenname + Score-Typ, damit jede Kombination eindeutig bleibt
            # (z.B. semantic + hybrid würden sonst beide "Ähnlichkeit (0–1)" heißen).
            def _score_spalte(mkey: str) -> str:
                kurz = {"semantic": "Semantisch", "bm25": "BM25", "hybrid": "Hybrid"}
                typ  = "Score (Rohwert)" if mkey == "bm25" else "Ähnlichkeit (0–1)"
                return f"{kurz.get(mkey, mkey)} — {typ}"

            # Hinweis wenn BM25 mit Semantisch/Hybrid gemischt wird
            hat_bm25   = "bm25" in ev_methods
            hat_andere = any(m != "bm25" for m in ev_methods)
            if hat_bm25 and hat_andere:
                st.info(
                    "Hinweis: BM25-Scores (Rohwert, unbegrenzt positiv) und Ähnlichkeits-Scores "
                    "(0–1) sind nicht direkt vergleichbar — sie messen auf unterschiedlichen Skalen. "
                    "Die Spaltenbezeichnungen in der Tabelle machen das kenntlich."
                )

            goldstandard = _load_goldstandard()

            st.markdown("---")
            st.markdown("**Ø-Score je Collection und Methode:**")

            summary_data = []
            for mk, sk, col_name in ev_cols:
                row = {
                    "Collection": col_name,
                    "Modell":     mk,
                    "Strategie":  STRATEGIES[sk]["label"],
                }
                for mkey in ev_methods:
                    scores = []
                    for key_hits in ev_res.values():
                        hits = key_hits.get((col_name, mkey), [])
                        if hits:
                            scores.append(sum(h["score"] for h in hits) / len(hits))
                    row[_score_spalte(mkey)] = round(sum(scores) / len(scores), 4) if scores else "—"

                    if goldstandard:
                        pk_scores = []
                        for frage, key_hits in ev_res.items():
                            relevant = goldstandard.get(frage)
                            if relevant is None:
                                continue
                            hits = key_hits.get((col_name, mkey), [])
                            pk_scores.append(_precision_at_k(hits, relevant))
                        kurz = {"semantic": "Semantisch", "bm25": "BM25", "hybrid": "Hybrid"}
                        pk_col = f"{kurz.get(mkey, mkey)} — Precision@K"
                        row[pk_col] = round(sum(pk_scores) / len(pk_scores), 4) if pk_scores else "—"

                summary_data.append(row)

            st.dataframe(summary_data, use_container_width=True, hide_index=True)

            if goldstandard:
                annotiert = sum(1 for f in ev_res if f in goldstandard)
                st.caption(
                    f"Precision@K basiert auf {annotiert} von {len(ev_res)} Fragen mit "
                    f"Goldstandard-Annotation (data/goldstandard.json)."
                )

            # ── Markdown-Export ──────────────────────────────────────────────
            datum = datetime.date.today().isoformat()
            col_labels = [f"{mk}/{STRATEGIES[sk]['label']}" for mk, sk, _ in ev_cols]
            alpha_info = f", Alpha={ev_a}" if "hybrid" in ev_methods else ""

            md_lines = [
                "# Evaluierungs-Protokoll — EU AI Act RAG",
                "",
                f"**Datum:** {datum}  ",
                f"**Collections:** {', '.join(col_labels)}  ",
                f"**Suchmethoden:** {', '.join(_METHODEN[m] for m in ev_methods)}{alpha_info}  ",
                f"**Top-K:** {ev_k}",
                "",
            ]

            if hat_bm25 and hat_andere:
                md_lines += [
                    "> **Hinweis zur Score-Vergleichbarkeit:** BM25-Scores (Rohwert, unbegrenzt "
                    "positiv) und Ähnlichkeits-Scores (0–1, Kosinus-Distanz) messen auf "
                    "unterschiedlichen Skalen und sind nicht direkt miteinander vergleichbar.",
                    "",
                ]

            md_lines += ["---", ""]

            kurz_methode = {"semantic": "Semantisch", "bm25": "BM25", "hybrid": "Hybrid"}

            for frage, key_hits in ev_res.items():
                md_lines.append(f"## {frage}")
                relevant_gs = goldstandard.get(frage) if goldstandard else None
                if relevant_gs:
                    md_lines.append(
                        f"*Goldstandard-Artikel: {', '.join(f'Art. {a}' for a in relevant_gs)}*"
                    )
                md_lines.append("")
                for mkey in ev_methods:
                    score_label = _score_spalte(mkey)
                    pk_header = " | Precision@K" if relevant_gs else ""
                    md_lines.append(f"### {_METHODEN[mkey]}")
                    md_lines.append("")
                    md_lines.append(f"| Collection | Artikel | {score_label} | Chunk-Vorschau{pk_header} |")
                    sep_pk = " | ---" if relevant_gs else ""
                    md_lines.append(f"|------------|---------|{'-' * len(score_label)}|----------------{sep_pk}|")
                    for mk, sk, col_name in ev_cols:
                        hits = key_hits.get((col_name, mkey), [])
                        clabel = f"{mk}/{STRATEGIES[sk]['label']}"
                        pk_cell = ""
                        if relevant_gs:
                            pk_val = _precision_at_k(hits, relevant_gs)
                            pk_cell = f" | {pk_val}"
                        if hits:
                            best = hits[0]
                            prev = best["content"][:120].replace("\n", " ").replace("|", "\\|")
                            md_lines.append(
                                f"| {clabel} | Art. {best['article_number']} — {best['title']} "
                                f"| {best['score']} | {prev}…{pk_cell} |"
                            )
                        else:
                            md_lines.append(f"| {clabel} | — | — | leer{pk_cell} |")
                    md_lines.append("")

            md_lines += [
                "---",
                "",
                "## Zusammenfassung — Ø-Score je Collection und Methode",
                "",
            ]
            if goldstandard:
                annotiert_md = sum(1 for f in ev_res if f in goldstandard)
                md_lines += [
                    f"*Precision@K basiert auf {annotiert_md} von {len(ev_res)} Fragen "
                    f"mit Goldstandard-Annotation.*",
                    "",
                ]

            score_cols_header = [_score_spalte(m) for m in ev_methods]
            pk_cols_header = (
                [f"{kurz_methode.get(m, m)} — Precision@K" for m in ev_methods]
                if goldstandard else []
            )
            all_col_headers = score_cols_header + pk_cols_header
            methoden_header = " | ".join(all_col_headers)
            methoden_sep    = " | ".join("---" for _ in all_col_headers)
            md_lines.append(f"| Collection | Modell | Strategie | {methoden_header} |")
            md_lines.append(f"|------------|--------|-----------|{methoden_sep}|")
            for r in summary_data:
                all_vals = [str(r.get(c, "—")) for c in all_col_headers]
                scores_cols = " | ".join(all_vals)
                md_lines.append(
                    f"| {r['Collection']} | {r['Modell']} | {r['Strategie']} | {scores_cols} |"
                )
            md_lines.append("")

            md_content = "\n".join(md_lines)
            dateiname = f"evaluierung_{datum}_top{ev_k}.md"
            st.download_button(
                label="Protokoll als Markdown herunterladen",
                data=md_content,
                file_name=dateiname,
                mime="text/markdown",
                use_container_width=True,
            )
