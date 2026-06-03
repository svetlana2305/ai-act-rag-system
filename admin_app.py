import datetime
import logging
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from database_manager import (
    MODELS, STRATEGIES, collection_prefix, browse_collection, 
    delete_collection, get_chunks_for_article, get_client, 
    get_status, import_articles, retrieve_chunks
)
from scraper import check_and_scrape, clear_cache, load_cached_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
st.set_page_config(page_title="EU AI Act Master RAG", page_icon="⚖️", layout="wide")

# --- LOGIN ---
USERS = {os.getenv(f"ADMIN_USER_{i}"): os.getenv(f"ADMIN_PASS_{i}") for i in ("1", "2", "3") if os.getenv(f"ADMIN_USER_{i}")}
if not st.session_state.get("auth"):
    st.title("⚖️ EU AI Act RAG — Admin Login")
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login"):
            u, p = st.text_input("Nutzer"), st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden", use_container_width=True):
                if USERS.get(u) == p:
                    st.session_state.update({"auth": True, "username": u})
                    st.rerun()
                else: st.error("Login-Daten falsch.")
    st.stop()

@st.cache_resource
def get_weaviate_client(): return get_client()

client = get_weaviate_client()
corpus = load_cached_corpus()
is_scraped = len(corpus) >= 306

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ System-Status")
    ready = client.is_ready()
    if ready: st.success("Weaviate online")
    else: st.error("Weaviate offline")
    
    if is_scraped: st.success(f"Daten: {len(corpus)} Dokumente")
    else: st.warning("Cache leer")

    with st.expander("🛠️ Wartung"):
        if st.button("Neu Scrapen erzwingen"):
            check_and_scrape(force=True)
            st.rerun()
        if st.button("Daten-Cache löschen"):
            clear_cache()
            st.rerun()
    
    if st.button("Abmelden"):
        st.session_state["auth"] = False
        st.rerun()

# --- TABS ---
st.title("⚖️ EU AI Act — RAG Admin Dashboard")
tab_sync, tab_db, tab_eval = st.tabs(["🔄 Synchronisierung", "🔍 Datenbank-Browser", "Ausgabe"])

# TAB 1: SMART SYNC
with tab_sync:
    if not is_scraped:
        st.info("👋 Willkommen! Bitte starte das initiale Scraping.")
        if st.button("Scraping starten", type="primary"):
            with st.status("Lade Daten..."): check_and_scrape()
            st.rerun()
    else:
        st.subheader("Collection Management")
        db_status = get_status(client)
        cols = st.columns(2)
        for i, m_key in enumerate(MODELS):
            with cols[i]:
                st.markdown(f"### Modell: **{m_key}**")
                for s_key, cfg in STRATEGIES.items():
                    stat = db_status.get((m_key, s_key), {"documents": 0, "chunks": 0, "exists": False})
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        c1.markdown(f"**{cfg['label']}**")
                        if stat["documents"] > 0:
                            c1.caption(f"✅ {stat['documents']} Dok. / {stat['chunks']} Chunks")
                            if c2.button("Löschen", key=f"del_{m_key}_{s_key}"):
                                delete_collection(client, m_key, s_key)
                                st.rerun()
                        else:
                            c1.caption("❌ Fehlt")
                            if c2.button("Import", key=f"imp_{m_key}_{s_key}"):
                                with st.spinner("Embedding..."): import_articles(client, corpus, m_key, s_key)
                                st.rerun()

# TAB 2: BROWSER
with tab_db:
    st.subheader("Datenbank durchsuchen")
    c1, c2 = st.columns(2)
    db_m = c1.selectbox("Modell", list(MODELS.keys()), key="db_sel_m")
    db_s = c2.selectbox("Strategie", list(STRATEGIES.keys()), key="db_sel_s")
    limit = st.slider("Limit", 5, 50, 20)
    
    if st.button("Daten laden", use_container_width=True):
        st.session_state["db_rows"] = browse_collection(client, db_m, db_s, limit)

    for row in st.session_state.get("db_rows", []):
        with st.expander(f"Art. {row['article_number']}: {row['title']} ({row['chunk_count']} Chunks)"):
            st.text_area("Volltext", row["full_text"], height=200, disabled=True, key=f"ta_{row['uuid']}")
            if st.button("Chunks anzeigen", key=f"btn_ch_{row['uuid']}"):
                for c in get_chunks_for_article(client, db_m, db_s, row["uuid"]):
                    st.info(f"Chunk {c['index']+1}/{c['total']}: {c['content']}")

# TAB 3: EVALUIERUNG & MD-DOWNLOAD
_STANDARD_FRAGEN = """Welche KI-Systeme sind laut EU AI Act vollständig verboten?
Welche Pflichten haben Anbieter von Hochrisiko-KI-Systemen?
Was sind die Transparenzanforderungen für KI-Systeme?
Welche Sanktionen drohen bei Verstößen?"""

with tab_eval:
    st.subheader("🔍 Such-Konfiguration")
    
    # --- 1. DATEN-AUSWAHL ---
    st.markdown("**1. Welche Datenbank-Konfigurationen sollen abgefragt werden?**")
    
    status = get_status(client)
    col1, col2 = st.columns(2)
    sel_cols = []
    
    with col1:
        st.markdown("#### OpenAI 3-Small")
        with st.container(border=True):
            if status.get(("3Small", "Artikel"), {}).get("documents", 0) > 0:
                if st.checkbox("Ganze Artikel", key="chk_s_art"): sel_cols.append(("3Small", "Artikel"))
            if status.get(("3Small", "500"), {}).get("documents", 0) > 0:
                if st.checkbox("500 Tokens (Fein)", key="chk_s_500"): sel_cols.append(("3Small", "500"))
            if status.get(("3Small", "2000"), {}).get("documents", 0) > 0:
                if st.checkbox("2000 Tokens (Grob)", key="chk_s_2000"): sel_cols.append(("3Small", "2000"))

    with col2:
        st.markdown("#### OpenAI 3-Large")
        with st.container(border=True):
            if status.get(("3Large", "Artikel"), {}).get("documents", 0) > 0:
                if st.checkbox("Ganze Artikel", key="chk_l_art"): sel_cols.append(("3Large", "Artikel"))
            if status.get(("3Large", "500"), {}).get("documents", 0) > 0:
                if st.checkbox("500 Tokens (Fein)", key="chk_l_500"): sel_cols.append(("3Large", "500"))
            if status.get(("3Large", "2000"), {}).get("documents", 0) > 0:
                if st.checkbox("2000 Tokens (Grob)", key="chk_l_2000"): sel_cols.append(("3Large", "2000"))

    if not sel_cols:
        st.info("Bitte importiere zuerst Daten unter 'Synchronisierung' oder wähle oben mindestens eine Checkbox aus.")

    st.divider()

    # --- 2. SUCH-PARAMETER ---
    col_param1, col_param2 = st.columns([1, 1])
    
    with col_param1:
        st.markdown("**2. Wie soll gesucht werden?**")
        meths = []
        if st.checkbox("Bedeutung (Semantisch)", value=True): meths.append("semantic")
        if st.checkbox("Stichwörter (BM25)"): meths.append("bm25")
        if st.checkbox("Kombination (Hybrid)"): meths.append("hybrid")

    with col_param2:
        st.markdown("**3. Anzeige**")
        top_k = st.slider("Wie viele Treffer sollen pro Frage angezeigt werden?", 1, 5, 3)

    # --- 3. EVALUATION (RICARDA) ---
    with st.expander("🛠️ Erweiterte Parameter & Evaluation (Für Ricarda)"):
        st.markdown("Hier werden später die Test-Datensätze und Scoring-Funktionen integriert.")
        fragen = st.text_area("Testfragen (eine pro Zeile)", value=_STANDARD_FRAGEN, height=150).splitlines()
        hybrid_alpha = st.slider("Hybrid-Gewichtung (0 = reines Stichwort, 1 = reine Bedeutung)", 0.0, 1.0, 0.5)

    st.divider()

    # --- 4. EXECUTION ---
    if st.button("Ausgabe starten", type="primary", use_container_width=True):
        if not sel_cols: 
            st.warning("Bitte wähle oben mindestens eine Konfiguration aus.")
        elif not meths:
            st.warning("Bitte wähle mindestens eine Suchmethode aus.")
        else:
            ergebnisse = {}
            bar = st.progress(0)
            step = 0
            total_steps = len(fragen) * len(sel_cols) * len(meths)
            
            for f in fragen:
                ergebnisse[f] = {}
                for mk, sk in sel_cols:
                    for m in meths:
                        hits = retrieve_chunks(client, mk, sk, f.strip(), top_k, m, hybrid_alpha)
                        ergebnisse[f][(collection_prefix(mk, sk), m)] = hits
                        step += 1
                        bar.progress(step / total_steps)
            
            st.session_state["eval_data"] = ergebnisse
            st.success("Abfrage abgeschlossen!")

    # --- 5. RESULT-RENDERING ---
    if "eval_data" in st.session_state:
        data = st.session_state["eval_data"]
        
        # Markdown Generierung
        md = ["# Ausgabe EU AI Act RAG", f"Datum: {datetime.date.today()}", "---"]
        for q, res in data.items():
            md.append(f"## Frage: {q}")
            for (col, meth), hits in res.items():
                md.append(f"### {col} ({meth})")
                if hits:
                    md.append(f"Bester Treffer: Art. {hits[0]['article_number']} (Score: {hits[0]['score']})")
                    md.append(f"> {hits[0]['content'][:250]}...")
                md.append("")
        
        md_final = "\n".join(md)
        st.download_button("📥 Protokoll (.md) herunterladen", md_final, file_name=f"eval_{datetime.date.today()}.md")
        
        # Visuelle Darstellung
        st.markdown("### Suchergebnisse")
        for q, res in data.items():
            with st.expander(f"Frage: {q}"):
                for (col_name, meth), hits in res.items():
                    st.markdown(f"**Konfiguration:** `{col_name}` | **Modus:** `{meth}`")
                    if not hits:
                        st.caption("Keine Treffer gefunden.")
                        continue
                    
                    for hit in hits:
                        st.info(f"**Artikel {hit['article_number']}** (Score: {hit['score']})\n\n{hit['content']}")
                    st.divider()
