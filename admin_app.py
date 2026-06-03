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

# --- SIDEBAR (WARTUNG & SYNCHRONISATION) ---
with st.sidebar:
    st.title("System-Status")
    ready = client.is_ready()
    if ready: st.success("Weaviate online")
    else: st.error("Weaviate offline")
    
    if is_scraped: st.success(f"Daten: {len(corpus)} Dokumente")
    else: st.warning("Cache leer")

    with st.expander("System-Wartung & DB-Setup"):
        st.markdown("### 1. Web-Scraper")
        if st.button("Scraping starten/prüfen", use_container_width=True):
            with st.status("Lade Daten..."): check_and_scrape()
            st.rerun()
        if st.button("Lokal gecachte Dateien löschen", use_container_width=True):
            clear_cache()
            st.rerun()
            
        st.divider()
        st.markdown("### 2. Weaviate Vektor-Index")
        
        is_infra_admin = st.session_state.get("username") == "Svetlana"
        unlock_delete = False
        
        if is_infra_admin:
            unlock_delete = st.checkbox("Lösch-Buttons freischalten")
        else:
            st.caption("Index-Löschung für diesen Account gesperrt.")

        for m_key in MODELS:
            st.markdown(f"**Modell: {m_key}**")
            for s_key, cfg in STRATEGIES.items():
                stat = db_status.get((m_key, s_key), {"documents": 0, "chunks": 0, "exists": False})
                c1, c2 = st.columns([2, 1])
                
                if stat["documents"] > 0:
                    c1.caption(f"Aktiv: {cfg['label']} ({stat['chunks']} Chks)")
                    if unlock_delete and c2.button("Löschen", key=f"del_{m_key}_{s_key}"):
                        delete_collection(client, m_key, s_key)
                        st.rerun()
                else:
                    c1.caption(f"Fehlt: {cfg['label']}")
                    if c2.button("Import", key=f"imp_{m_key}_{s_key}"):
                        with st.spinner("Embedding..."): import_articles(client, corpus, m_key, s_key)
                        st.rerun()
    
    if st.button("Abmelden", use_container_width=True):
        st.session_state["auth"] = False
        st.rerun()

# --- HAUPTSEITE TABS ---
st.title("EU AI Act — RAG Forschungs-App")
tab_eval, tab_browser = st.tabs(["Suche & Ausgabe", "Datenbank-Browser"])

_STANDARD_FRAGEN = """Welche KI-Systeme sind laut EU AI Act vollständig verboten?
Welche Pflichten haben Anbieter von Hochrisiko-KI-Systemen?
Was sind die Transparenzanforderungen für KI-Systeme?
Welche Sanktionen drohen bei Verstößen?"""

# TAB 1: SUCHE & AUSGABE
with tab_eval:
    st.subheader("Parameter-Auswahl")
    
    st.markdown("**1. Welche Datenbank-Konfigurationen sollen abgefragt werden?**")
    col_matrix1, col_matrix2 = st.columns(2)
    sel_cols = []
    
    with col_matrix1:
        st.markdown("#### OpenAI 3-Small")
        with st.container(border=True):
            if db_status.get(("3Small", "Artikel"), {}).get("documents", 0) > 0:
                if st.checkbox("Ganze Artikel", key="chk_s_art"): sel_cols.append(("3Small", "Artikel"))
            if db_status.get(("3Small
