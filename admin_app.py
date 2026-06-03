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
    
    # 1. HAUPT-EINGABEBEREICH (Fokus des Nutzers)
    st.markdown("### 🔍 Frag den EU AI Act")
    mode = st.radio("Modus", ["Einzelne Frage", "Wissenschaftliche Testreihe (Batch)"], horizontal=True, label_visibility="collapsed")
    
    fragen = []
    
    if mode == "Einzelne Frage":
        user_query = st.text_input("Deine Frage:", placeholder="z.B. Welche KI-Systeme sind vollständig verboten?", label_visibility="collapsed")
        if user_query:
            fragen = [user_query.strip()]
    else:
        st.info("Batch-Modus: Führt alle Fragen gegen alle unten ausgewählten Konfigurationen aus. (Ohne LLM-Generierung zur Kostenersparnis)")
        fragen_text = st.text_area("Testfragen (Eine Frage pro Zeile)", value=_STANDARD_FRAGEN, height=150)
        fragen = [f.strip() for f in fragen_text.splitlines() if f.strip()]

    st.write("") # Abstand

    # 2. ERWEITERTE PARAMETER (Versteckt im Expander)
    with st.expander("⚙️ Such- und Datenbankparameter (Erweitert)", expanded=False):
        st.markdown("**1. Datenbank-Konfigurationen**")
        col_matrix1, col_matrix2 = st.columns(2)
        sel_cols = []
        
        with col_matrix1:
            st.markdown("#### OpenAI 3-Small")
            with st.container(border=True):
                if db_status.get(("3Small", "Artikel"), {}).get("documents", 0) > 0:
                    if st.checkbox("Ganze Artikel", key="chk_s_art"): sel_cols.append(("3Small", "Artikel"))
                if db_status.get(("3Small", "500"), {}).get("documents", 0) > 0:
                    if st.checkbox("500 Tokens (Feines Chunking)", key="chk_s_500"): sel_cols.append(("3Small", "500"))
                if db_status.get(("3Small", "2000"), {}).get("documents", 0) > 0:
                    if st.checkbox("2000 Tokens (Grobes Chunking)", key="chk_s_2000"): sel_cols.append(("3Small", "2000"))

        with col_matrix2:
            st.markdown("#### OpenAI 3-Large")
            with st.container(border=True):
                if db_status.get(("3Large", "Artikel"), {}).get("documents", 0) > 0:
                    if st.checkbox("Ganze Artikel", key="chk_l_art"): sel_cols.append(("3Large", "Artikel"))
                if db_status.get(("3Large", "500"), {}).get("documents", 0) > 0:
                    if st.checkbox("500 Tokens (Feines Chunking)", key="chk_l_500"): sel_cols.append(("3Large", "500"))
                if db_status.get(("3Large", "2000"), {}).get("documents", 0) > 0:
                    if st.checkbox("2000 Tokens (Grobes Chunking)", key="chk_l_2000"): sel_cols.append(("3Large", "2000"))

        st.divider()

        col_param1, col_param2 = st.columns(2)
        with col_param1:
            st.markdown("**2. Suchlogik**")
            meths = []
            if st.checkbox("Bedeutungssuche (Semantisch / Vektor)", value=True): meths.append("semantic")
            if st.checkbox("Stichwortsuche (BM25 / Keyword)"): meths.append("bm25")
            if st.checkbox("Kombination (Hybrid-Suche)"): meths.append("hybrid")
            
            hybrid_alpha = 0.5
            if "hybrid" in meths:
                hybrid_alpha = st.slider("Hybrid-Gewichtung (0.0 = reines Keyword, 1.0 = rein Semantisch)", 0.0, 1.0, 0.5)
        
        with col_param2:
            st.markdown("**3. Anzahl der Treffer**")
            top_k = st.slider("Menge der Textabschnitte (Top-K)", 1, 5, 3, label_visibility="collapsed")

    # 3. AUSFÜHRUNG
    st.write("")
    if st.button("Suchen & Generieren" if mode == "Einzelne Frage" else "Testreihe starten", type="primary", use_container_width=True):
        if not sel_cols:
            st.warning("Bitte klappe die '⚙️ Such- und Datenbankparameter' auf und wähle mindestens eine Konfiguration aus.")
        elif not meths:
            st.warning("Bitte wähle in den Parametern mindestens eine Suchlogik aus.")
        elif not fragen:
            st.warning("Bitte gib eine Frage ein.")
        else:
            ergebnisse = {}
            bar = st.progress(0)
            step = 0
            total_steps = len(fragen) * len(sel_cols) * len(meths)
            
            for f in fragen:
                ergebnisse[f] = {}
                for mk, sk in sel_cols:
                    for m in meths:
                        hits = retrieve_chunks(client, mk, sk, f, top_k, m, hybrid_alpha)
                        ergebnisse[f][(collection_prefix(mk, sk), m)] = hits
                        step += 1
                        bar.progress(step / total_steps)
            
            st.session_state["eval_data"] = ergebnisse

            # RAG: Generierung der KI-Antwort (nur bei Einzelfragen)
            if mode == "Einzelne Frage":
                first_config_key = list(ergebnisse[fragen[0]].keys())[0]
                kontext_treffer = ergebnisse[fragen[0]][first_config_key]
                
                if not kontext_treffer:
                    st.session_state["rag_answer"] = "⚠️ Das System konnte keine passenden Textstellen im Gesetz finden. Es wird keine KI-Antwort generiert (Vermeidung von Halluzinationen)."
                else:
                    with st.spinner("Generiere KI-Antwort (RAG)..."):
                        kontext_text = "\n\n".join([h['content'] for h in kontext_treffer])
                        prompt = f"Beantworte die folgende Frage ausschließlich auf Basis des bereitgestellten Kontexts aus dem EU AI Act. Wenn der Kontext die Frage nicht beantwortet, sage das klar.\n\nKontext:\n{kontext_text}\n\nFrage: {fragen[0]}"
                        
                        headers = {
                            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", 
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "gpt-4o", 
                            "messages": [{"role": "user", "content": prompt}]
                        }
                        
                        try:
                            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                            resp.raise_for_status()
                            st.session_state["rag_answer"] = resp.json()["choices"][0]["message"]["content"]
                        except Exception as e:
                            st.session_state["rag_answer"] = f"Fehler bei der Antwortgenerierung: {e}"
            else:
                if "rag_answer" in st.session_state:
                    del st.session_state["rag_answer"]

            st.success("Abfrage erfolgreich beendet.")

    # 4. ERGEBNIS-ANZEIGE
    if "eval_data" in st.session_state:
        data = st.session_state["eval_data"]
        
        if "rag_answer" in st.session_state:
            st.markdown("### 🤖 Generierte Antwort")
            st.info(st.session_state["rag_answer"])
            st.divider()

        md = ["# Ausgabe EU AI Act RAG", f"Datum: {datetime.date.today()}", "---"]
        for q, res in data.items():
            md.append(f"## Frage: {q}")
            for (col, meth), hits in res.items():
                meth_de = "Bedeutung" if meth=="semantic" else "Stichwort" if meth=="bm25" else "Hybrid"
                md.append(f"### Setup: {col} | Suchlogik: {meth_de}")
                
                if not hits:
                    md.append("*Keine relevanten Textstellen gefunden.*\n")
                else:
                    for i, hit in enumerate(hits, 1):
                        art_nr = hit.get('article_number') or "Unbekannt"
                        title = hit.get('title') or "Kein Titel"
                        md.append(f"**Treffer {i}: Artikel {art_nr} - {title}** (Score: {hit['score']})")
                        md.append(f"{hit['content']}\n")
                md.append("---\n")
        
        md_final = "\n".join(md)
        st.download_button("📥 Protokoll (.md) herunterladen", md_final, file_name=f"rag_ausgabe_{datetime.date.today()}.md", key="download_md_btn")
        
        st.markdown("### 📋 Gefundene Textstellen (Kontext-Datenbank)")
        for q, res in data.items():
            with st.expander(f"Quellen für: {q}", expanded=True):
                for (col_name, meth), hits in res.items():
                    meth_de = "Bedeutung" if meth=="semantic" else "Stichwort" if meth=="bm25" else "Hybrid"
                    st.markdown(f"**Setup:** `{col_name}` | **Suchlogik:** `{meth_de}`")
                    
                    if not hits:
                        st.caption("Keine relevanten Textstellen in dieser Konfiguration gefunden.")
                    else:
                        for hit in hits:
                            art_nr = hit.get('article_number') or "Unbekannt"
                            title = hit.get('title') or "Kein Titel"
                            st.info(f"**Artikel {art_nr}: {title}** (Relevanz-Score: {hit['score']})\n\n{hit['content']}")
                    st.divider()

# TAB 2: BROWSER
with tab_browser:
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
