import json
import logging
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# Drei Dokumenttypen mit ihren URL-Schemata und Anzahlen
_SOURCES = [
    {
        "type":  "artikel",
        "url":   "https://artificialintelligenceact.eu/de/article/{}/",
        "count": 113,
        "label": "Artikel",
    },
    {
        "type":  "anhang",
        "url":   "https://artificialintelligenceact.eu/de/annex/{}/",
        "count": 13,
        "label": "Anhang",
    },
    {
        "type":  "erwaegungsgrund",
        "url":   "https://artificialintelligenceact.eu/de/recital/{}/",
        "count": 180,
        "label": "Erwägungsgrund",
    },
]


def _timestamp_file(source_type: str) -> Path:
    return DATA_DIR / f"last_scraped_{source_type}.txt"


def _cache_file(source_type: str) -> Path:
    return DATA_DIR / f"cache_{source_type}.json"


def _load_timestamp(source_type: str):
    f = _timestamp_file(source_type)
    if not f.exists():
        return None
    try:
        return datetime.fromisoformat(f.read_text().strip())
    except ValueError:
        return None


def _save_timestamp(source_type: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _timestamp_file(source_type).write_text(datetime.now(timezone.utc).isoformat())


def _load_cache(source_type: str) -> list:
    f = _cache_file(source_type)
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_cache(source_type: str, docs: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cache_file(source_type).write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _source_needs_update(source: dict) -> bool:
    # Ersten Eintrag stellvertretend per HEAD prüfen
    check_url = source["url"].format(1)
    try:
        r = requests.head(check_url, headers=HEADERS, timeout=15, allow_redirects=True)
        r.raise_for_status()
        lm = r.headers.get("Last-Modified")
    except requests.RequestException as e:
        logger.warning("%s HEAD fehlgeschlagen: %s — lade trotzdem.", source["label"], e)
        return True

    if not lm:
        return True

    try:
        remote = parsedate_to_datetime(lm).astimezone(timezone.utc)
    except Exception:
        return True

    local = _load_timestamp(source["type"])
    return local is None or remote > local


def _extract_text(soup: BeautifulSoup) -> str:
    # Divi-Theme: Artikelinhalt liegt immer in et_pb_post_content, direkt abrufbar
    container = soup.find(class_="et_pb_post_content")
    if container:
        lines = [l.strip() for l in container.get_text(separator="\n").splitlines() if l.strip()]
        return "\n".join(lines).strip()

    # Fallback für den Fall dass sich die Seitenstruktur ändert:
    # Volltext nehmen und Disclaimer-Abschnitte entfernen
    body = soup.get_text(separator="\n")
    if "HINWEIS:" in body:
        body = body.split("HINWEIS:", 1)[1]
        # Die Disclaimer-Zeile selbst überspringen (endet bei einem Zeilenumbruch)
        lines_raw = body.splitlines()
        # Erste nicht-leere Zeile ist der Rest des Disclaimers — entfernen
        content_lines = []
        disclaimer_done = False
        for line in lines_raw:
            stripped = line.strip()
            if not disclaimer_done:
                if not stripped:
                    continue  # Leerzeilen nach HINWEIS überspringen
                # Zeilen die eindeutig noch zum Disclaimer gehören
                if any(phrase in stripped for phrase in (
                    "maschinell", "offizielle Übersetzung", "Europäischen Parlaments",
                    "hier finden", "Weiter", "Nächste", "Naechste", "Feedback",
                )):
                    continue
                disclaimer_done = True
            if stripped:
                content_lines.append(stripped)
        return "\n".join(content_lines).strip()

    lines = [l.strip() for l in body.splitlines() if l.strip()]
    return "\n".join(lines).strip()


def _scrape_source(source: dict, progress_callback=None) -> list:
    docs = []
    label = source["label"]

    for nr in range(1, source["count"] + 1):
        url = source["url"].format(nr)
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("%s %d: Fehler — %s", label, nr, e)
            continue

        soup = BeautifulSoup(r.text, "lxml")
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else f"{label} {nr}"
        text = _extract_text(soup)

        if not text:
            logger.debug("%s %d leer, übersprungen.", label, nr)
            continue

        docs.append({
            "article_number": nr,
            "title":          title,
            "full_text":      text,
            "source_type":    source["type"],
        })

        if progress_callback:
            progress_callback(source["type"], source["label"], len(docs), source["count"])

        logger.info("%s %d/%d: %s", label, nr, source["count"], title)
        time.sleep(0.3)

    logger.info("%d %s-Dokumente gescrapt.", len(docs), label)
    return docs


def load_cached_corpus() -> list:
    """Lädt alle zwischengespeicherten Dokumente aus dem lokalen Cache."""
    all_docs = []
    for source in _SOURCES:
        all_docs.extend(_load_cache(source["type"]))
    return all_docs


def clear_cache():
    """Löscht alle lokalen Cache- und Zeitstempel-Dateien."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source in _SOURCES:
        for f in (_cache_file(source["type"]), _timestamp_file(source["type"])):
            if f.exists():
                f.unlink()
                logger.info("Cache gelöscht: %s", f)


def check_and_scrape(progress_callback=None, force: bool = False) -> tuple:
    """
    Prüft je Dokumenttyp ob eine Aktualisierung nötig ist.
    Mit force=True wird die Last-Modified-Prüfung übersprungen.
    Neu gescrapte Typen werden im Cache gespeichert.
    Gibt immer den vollständigen Korpus zurück (306 Dokumente),
    auch wenn kein Typ aktualisiert wurde.
    """
    any_scraped = False

    for source in _SOURCES:
        if force or _source_needs_update(source):
            docs = _scrape_source(source, progress_callback=progress_callback)
            # Vollständigkeitsprüfung: mindestens 80% der erwarteten Dokumente
            expected = source["count"]
            if len(docs) >= int(expected * 0.8):
                _save_cache(source["type"], docs)
                _save_timestamp(source["type"])
                any_scraped = True
                logger.info("%s: %d/%d Dokumente gescrapt und gecacht.", source["label"], len(docs), expected)
            elif docs:
                # Unvollständig — Cache wird nicht überschrieben
                logger.warning(
                    "%s: nur %d/%d Dokumente gescrapt — Cache bleibt unverändert.",
                    source["label"], len(docs), expected,
                )
            else:
                logger.warning("%s: keine Dokumente gescrapt.", source["label"])
        else:
            logger.info("%s: unverändert, lade aus Cache.", source["label"])

    # Immer den vollständigen Korpus aus dem Cache zurückgeben
    all_docs = load_cached_corpus()
    return any_scraped, all_docs
