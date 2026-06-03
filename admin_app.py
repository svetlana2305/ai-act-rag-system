import datetime
import logging
import os
import streamlit as st
from dotenv import load_dotenv
import requests

load_dotenv()

from database_manager import (
    MODELS, STRATEGIES, collection_prefix, browse_collection, 
    delete_collection, get_chunks_for_article, get_client, 
    get_status, import_articles, retrieve_chunks
)
from scraper import check_and_scrape, clear_cache, load_cached_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
st.set_page_config(page_title="EU AI Act Master RAG", layout="wide")

# --- LOGIN ---
USERS = {os.getenv(f"ADMIN_USER_{i}"): os.getenv(f"ADMIN_PASS_{i}") for i in ("1", "2", "3") if os.getenv(f"ADMIN_USER_{i}")}
if not st.session_state.get("auth"):
    st.title("EU AI Act RAG — Admin Login")
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
db_status = get_status(client)

# --- SIDEBAR ---
with st.sidebar:
    st.title("System-Status")
    if client.is_ready(): st.success("Weaviate online")
    else: st.error("Weaviate offline")
    
    with st.expander("System-Wartung"):
        if st.button("Neu Scrapen", use_container_width=True):
            with st.status("Lade Daten..."): check_and_scrape()
            st.rerun()
        if st.button("Cache löschen", use_container_width=True):
            clear_cache()
            st.rerun()
    
    if st.button("Abmelden", use_container_width=True):
        st.session_state["auth"] = False
        st.rerun()

# --- HAUPTSEITE ---
st.title("EU AI Act — RAG Forschungs-App")
tab_eval, tab_browser = st.tabs(["Suche & Ausgabe", "Datenbank-Browser"])

with tab_eval:
    mode = st.radio("Modus", ["Einzelne Frage", "Batch-Testreihe"], horizontal=True)
    user_query = st.text_input("Deine Frage:") if mode == "Einzelne Frage" else ""
    fragen = [user_query.strip()] if mode == "Einzelne Frage" and user_query else []
    
    if mode == "Batch-Testreihe":
        fragen_text = st.text_area("Testfragen (eine pro Zeile)", height=150)
        fragen = [f.strip() for f in fragen_text.splitlines() if f.strip()]

    with st.expander("⚙️ Parameter", expanded=True):
        c1, c2 = st.columns(2)
        sel_cols = []
        with c1:
            st.markdown("**Konfigurationen:**")
            for m in MODELS:
                for s in STRATEGIES:
                    if db_status.get((m, s), {}).get("documents", 0) > 0:
                        if st.checkbox(f"{m} / {s}", key=f"chk_{m}_{s}"): sel_cols.append((m, s))
        with c2:
            st.markdown("**Methoden:**")
            meths = []
            if st.checkbox("Bedeutung", True): meths.append("semantic")
            if st.checkbox("Stichwort (BM25)"): meths.append("bm25")
            if st.checkbox("Hybrid"): meths.append("hybrid")
            top_k = st.slider("Treffer (Top-K)", 1, 5, 3)

    if st.button("Suchen & Generieren", type="primary"):
        ergebnisse = {}
        for f in fragen:
            ergebnisse[f] = {}
            for mk, sk in sel_cols:
                for m in meths:
                    ergebnisse[f][(collection_prefix(mk, sk), m)] = retrieve_chunks(client, mk, sk, f, top_k, m, 0.5)
        st.session_state["eval_data"] = ergebnisse
        
        # Generierung
        if mode == "Einzelne Frage" and ergebnisse and sel_cols:
            first_key = list(ergebnisse[fragen[0]].keys())[0]
            kontext = ergebnisse[fragen[0]][first_key]
            if kontext:
                kontext_text = "\n".join([f"[{i+1}] Art. {h.get('article_number', 'Unbekannt')}: {h['content']}" for i, h in enumerate(kontext)])
                prompt = f"Beantworte die Frage basierend auf dem Kontext. Zitiere Quellen als [1], [2].\n\nKontext:\n{kontext_text}\n\nFrage: {fragen[0]}"
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}, 
                                     json={"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]})
                st.session_state["rag_answer"] = resp.json()["choices"][0]["message"]["content"]
                st.session_state["rag_sources"] = kontext
        st.rerun()

    # Ausgabe
    if "eval_data" in st.session_state:
        if "rag_answer" in st.session_state:
            st.markdown("### 🤖 Antwort")
            st.info(st.session_state["rag_answer"])
            st.markdown("#### Quellen")
            
            # Bereinigte Quellenanzeige
            for i, h in enumerate(st.session_state.get("rag_sources", []), 1):
                # Fallback-Logik für saubere Darstellung
                art_nr = h.get('article_number')
                art_title = h.get('title')
                
                label = f"Art. {art_nr}" if art_nr and art_nr != "None" else "Relevantes Dokument"
                sub_label = f": {art_title}" if art_title and art_title != "None" else ""
                
                st.markdown(f"**[{i}]** {label}{sub_label}")
            st.divider()

        data = st.session_state["eval_data"]
        md = ["# Ergebnis", f"Datum: {datetime.date.today()}"]
        for q, res in data.items():
            md.append(f"## Frage: {q}")
            for (col, meth), hits in res.items():
                for i, h in enumerate(hits, 1):
                    md.append(f"**[{i}] Art. {h.get('article_number', 'Unbekannt')}**: {h['content']}")
        
        st.download_button("📥 Protokoll (.md) laden", "\n".join(md), file_name="rag.md", key="dwn_1")
        for q, res in data.items():
            with st.expander(f"Quellen für: {q}"):
                for (col, meth), hits in res.items():
                    for h in hits:
                        st.info(f"**Art. {h.get('article_number', 'Unbekannt')}**: {h['content']}")

with tab_browser:
    st.subheader("Datenbank durchsuchen")
    c1, c2 = st.columns(2)
    db_m = c1.selectbox("Modell", list(MODELS.keys()))
    db_s = c2.selectbox("Strategie", list(STRATEGIES.keys()))
    if st.button("Laden"): st.session_state["db_rows"] = browse_collection(client, db_m, db_s, 20)
    for row in st.session_state.get("db_rows", []):
        with st.expander(f"Art. {row.get('article_number', 'Unbekannt')}: {row.get('title', 'Ohne Titel')}"):
            st.text_area("Volltext", row.get("full_text", ""), disabled=True)
