"""
News-Analyse Modul — vollständig unabhängig, keine externen APIs benötigt.
Quellen: yfinance .news + Yahoo Finance RSS
Sentiment: regelbasiert mit Finanz-Lexikon (kein ML-Modell nötig)
"""

import re
import time
import logging
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

# ── FINANZ-SENTIMENT-LEXIKON ─────────────────────────────────────────────────
# Kuratierte Wörter speziell für Aktiennachrichten

BULLISH_WORDS = {
    # Ergebnisse & Wachstum
    "beat": 2, "beats": 2, "record": 2, "record-high": 3, "all-time high": 3,
    "exceeded": 2, "surpassed": 2, "topped": 2, "outperformed": 2,
    "raised guidance": 3, "raised outlook": 3, "raised forecast": 3,
    "upgrade": 2, "upgraded": 2, "buy rating": 3, "strong buy": 3,
    "outperform": 2, "overweight": 2,
    # Kurs & Momentum
    "surge": 2, "surges": 2, "surged": 2, "rally": 2, "rallied": 2,
    "breakout": 2, "momentum": 1, "bullish": 3, "upside": 2,
    "52-week high": 3, "new high": 2, "multiyear high": 3,
    # Geschäft
    "acquisition": 1, "buyback": 2, "share repurchase": 2, "dividend": 1,
    "dividend increase": 3, "special dividend": 2, "profit": 1,
    "revenue growth": 2, "earnings growth": 2, "margin expansion": 2,
    "strong demand": 2, "robust demand": 2, "contract win": 2,
    "partnership": 1, "expansion": 1, "launch": 1,
    "positive": 1, "growth": 1, "strong": 1, "solid": 1, "robust": 1,
}

BEARISH_WORDS = {
    # Ergebnisse & Ausblick
    "miss": 2, "missed": 2, "misses": 2, "disappointed": 2, "disappoints": 2,
    "below expectations": 3, "cut guidance": 3, "lowered guidance": 3,
    "lowered outlook": 3, "lowered forecast": 3, "warning": 2,
    "profit warning": 3, "revenue warning": 3,
    "downgrade": 2, "downgraded": 2, "sell rating": 3, "underperform": 2,
    "underweight": 2, "bearish": 3,
    # Kurs & Risiko
    "plunge": 2, "plunges": 2, "plunged": 2, "crash": 2, "sell-off": 2,
    "selloff": 2, "drop": 1, "drops": 1, "decline": 1, "fell": 1,
    "tumble": 2, "tumbled": 2, "slump": 2, "slumped": 2,
    "52-week low": 3, "new low": 2,
    # Geschäft & Risiko
    "lawsuit": 2, "investigation": 2, "sec investigation": 3, "fraud": 3,
    "recall": 2, "layoffs": 2, "restructuring": 1, "job cuts": 2,
    "bankruptcy": 3, "default": 3, "debt": 1, "loss": 2, "losses": 2,
    "fine": 1, "penalty": 2, "regulation": 1, "ban": 2,
    "weak demand": 2, "slowing growth": 2, "margin pressure": 2,
    "competition": 1, "tariff": 1, "tariffs": 2,
    "negative": 1, "concern": 1, "risk": 1, "uncertainty": 1,
}

def _score_text(text: str) -> float:
    """Gibt einen Sentiment-Score zurück: positiv > 0, negativ < 0."""
    text_lower = text.lower()
    score = 0.0
    for phrase, weight in BULLISH_WORDS.items():
        if phrase in text_lower:
            score += weight
    for phrase, weight in BEARISH_WORDS.items():
        if phrase in text_lower:
            score -= weight
    return score


def _fetch_yfinance_news(ticker: str) -> list[dict]:
    """
    Holt Nachrichten über yfinance .news
    Unterstützt alle bekannten Formate:
      - Alt (< 0.2.40): item hat direkt 'title', 'publisher', 'providerPublishTime'
      - Neu (≥ 0.2.40): item hat 'content' dict mit 'title', 'provider', 'pubDate', 'summary'
      - Neu2: item hat 'content' → 'clickThroughUrl', 'provider' → 'displayName'
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = t.news or []
        results = []
        cutoff = datetime.now() - timedelta(days=14)

        for item in raw[:20]:
            # ── Format erkennen ──────────────────────────────────────────
            # Neues Format: alles in item["content"]
            content = item.get("content") or item

            # Titel
            title = (content.get("title")
                     or content.get("headline")
                     or item.get("title")
                     or "")

            # Publisher / Source
            provider = content.get("provider") or {}
            if isinstance(provider, dict):
                source = provider.get("displayName") or provider.get("name") or ""
            else:
                source = str(provider)
            if not source:
                source = (content.get("publisher")
                          or item.get("publisher")
                          or item.get("source")
                          or "Yahoo Finance")

            # Zusammenfassung
            summary = (content.get("summary")
                       or content.get("description")
                       or item.get("summary")
                       or "")

            # Datum
            pub_raw = (content.get("pubDate")
                       or content.get("publishedAt")
                       or item.get("providerPublishTime")
                       or item.get("published")
                       or "")
            try:
                if isinstance(pub_raw, (int, float)) and pub_raw > 0:
                    pub = datetime.fromtimestamp(pub_raw)
                elif isinstance(pub_raw, str) and pub_raw:
                    # ISO-Format: "2025-03-05T14:32:00Z"
                    pub_raw_clean = pub_raw.replace("Z", "+00:00")
                    from datetime import timezone
                    pub = datetime.fromisoformat(pub_raw_clean).replace(tzinfo=None)
                else:
                    pub = datetime.now()
            except Exception:
                pub = datetime.now()

            if pub < cutoff:
                continue
            # URL
            url = (content.get("canonicalUrl", {}) or {}).get("url") or \
                  (content.get("clickThroughUrl", {}) or {}).get("url") or \
                  content.get("url") or item.get("link") or item.get("url") or ""

            if not title:
                continue

            results.append({
                "title":   title.strip(),
                "summary": summary.strip(),
                "source":  source.strip(),
                "date":    pub.strftime("%d.%m.%Y"),
                "url":     url.strip(),
            })

        return results
    except Exception as e:
        log.debug(f"yfinance news error for {ticker}: {e}")
        return []


def _fetch_rss_news(ticker: str) -> list[dict]:
    """Fallback: Yahoo Finance RSS Feed — zuverlässig und immer verfügbar."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "xml")
        if not soup.find("item"):
            # Kein xml-Parser? Versuche html.parser
            soup = BeautifulSoup(resp.text, "html.parser")

        items = soup.find_all("item")[:15]
        results = []
        cutoff = datetime.now() - timedelta(days=14)
        for item in items:
            title   = item.find("title")
            desc    = item.find("description")
            pub_tag = item.find("pubDate")
            source  = item.find("source")

            title_text = title.get_text(strip=True) if title else ""
            desc_text  = desc.get_text(strip=True)  if desc  else ""
            src_text   = source.get_text(strip=True) if source else "Yahoo Finance"

            # Link-URL (aus <link> oder <guid>)
            link_tag  = item.find("link")
            guid_tag  = item.find("guid")
            link_url  = ""
            if link_tag:
                link_url = link_tag.get_text(strip=True) or (link_tag.next_sibling or "")
            if not link_url and guid_tag:
                link_url = guid_tag.get_text(strip=True)
            link_url = str(link_url).strip()

            # CDATA bereinigen
            title_text = re.sub(r'<!\[CDATA\[|\]\]>', '', title_text).strip()
            desc_text  = re.sub(r'<!\[CDATA\[|\]\]>', '', desc_text).strip()
            desc_text  = re.sub(r'<[^>]+>', '', desc_text).strip()  # HTML-Tags raus

            if not title_text:
                continue

            try:
                import email.utils
                pub = datetime(*email.utils.parsedate(pub_tag.text.strip())[:6]) if pub_tag else datetime.now()
            except Exception:
                pub = datetime.now()

            if pub < cutoff:
                continue

            results.append({
                "title":   title_text,
                "summary": desc_text,
                "source":  src_text,
                "date":    pub.strftime("%d.%m.%Y"),
                "url":     link_url,
            })
        return results
    except Exception as e:
        log.debug(f"RSS error for {ticker}: {e}")
        return []


def _fetch_google_news(ticker: str) -> list[dict]:
    """
    Google News RSS — breiteste Abdeckung, findet auch Artikel die Yahoo nicht hat.
    URL: https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en
    """
    try:
        query = f"{ticker} stock"
        url   = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp  = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []

        from bs4 import BeautifulSoup
        soup  = BeautifulSoup(resp.text, "xml")
        if not soup.find("item"):
            soup = BeautifulSoup(resp.text, "html.parser")

        items  = soup.find_all("item")[:15]
        results = []
        cutoff  = datetime.now() - timedelta(days=14)

        for item in items:
            title_tag  = item.find("title")
            link_tag   = item.find("link")
            pub_tag    = item.find("pubDate")
            source_tag = item.find("source")

            title_text = title_tag.get_text(strip=True) if title_tag else ""
            # Google News hängt " - Quelle" ans Ende des Titels
            source_text = source_tag.get_text(strip=True) if source_tag else ""
            if source_text and title_text.endswith(f" - {source_text}"):
                title_text = title_text[: -(len(source_text) + 3)].strip()

            link_url = ""
            if link_tag:
                link_url = link_tag.get_text(strip=True)
                # Google News gibt oft ./articles/... URLs — in volle URL umwandeln
                if link_url.startswith("./"):
                    link_url = "https://news.google.com/" + link_url[2:]

            try:
                import email.utils
                pub = datetime(*email.utils.parsedate(pub_tag.get_text(strip=True))[:6]) if pub_tag else datetime.now()
            except Exception:
                pub = datetime.now()

            if pub < cutoff or not title_text:
                continue

            results.append({
                "title":   title_text,
                "summary": "",
                "source":  source_text or "Google News",
                "date":    pub.strftime("%d.%m.%Y"),
                "url":     link_url,
            })

        return results
    except Exception as e:
        log.debug(f"Google News error for {ticker}: {e}")
        return []


# ── SEC EDGAR ────────────────────────────────────────────────────────────────
# Globaler CIK-Cache damit nicht bei jedem Ticker neu geladen werden muss
_cik_cache: dict = {}

def _get_cik(ticker: str) -> str | None:
    """Gibt die SEC CIK-Nummer für einen Ticker zurück (mit Cache)."""
    global _cik_cache
    if not _cik_cache:
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                timeout=15,
                headers={"User-Agent": "BreakoutScreener/1.0 research@example.com"},
            )
            if resp.status_code == 200:
                data = resp.json()
                _cik_cache = {
                    v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                    for v in data.values()
                }
        except Exception as e:
            log.debug(f"CIK-Cache Ladefehler: {e}")

    return _cik_cache.get(ticker.upper())


def _fetch_sec_filings(ticker: str) -> list[dict]:
    """
    SEC EDGAR — offizielle Pflichtmeldungen (8-K Events, Earnings etc.)
    Liefert: Form-Typ, Datum, Beschreibung, direkter Link zum Filing.
    """
    # Wichtige Form-Typen mit Erklärungen
    FORM_LABELS = {
        "8-K":    "Wichtiges Ereignis (8-K)",
        "8-K/A":  "Korrektur Ereignismeldung (8-K/A)",
        "10-Q":   "Quartalsbericht (10-Q)",
        "10-K":   "Jahresbericht (10-K)",
        "4":      "Insider-Transaktion (Form 4)",
        "SC 13G": "Großaktionär-Meldung (>5%)",
        "SC 13D": "Aktivistischer Investor (>5%)",
        "DEF 14A":"Hauptversammlungs-Proxy",
    }
    # 8-K Item-Codes → lesbare Beschreibungen
    ITEM_LABELS = {
        "1.01": "Wesentlicher Vertrag",
        "1.02": "Vertragsbeendigung",
        "1.03": "Insolvenzverfahren",
        "2.01": "Akquisition / Veräußerung",
        "2.02": "Quartalsergebnis",
        "2.05": "Stellenabbau / Restrukturierung",
        "2.06": "Wertminderung",
        "3.01": "Börsendelisting-Risiko",
        "4.01": "Wirtschaftsprüferwechsel",
        "4.02": "Rücknahme Finanzbericht",
        "5.01": "Kontrollwechsel",
        "5.02": "Management-Wechsel",
        "5.03": "Satzungsänderung",
        "7.01": "Regulatorische FD-Offenlegung",
        "8.01": "Sonstige Ereignisse",
        "9.01": "Finanzdaten / Anlagen",
    }

    try:
        cik = _get_cik(ticker)
        if not cik:
            return []

        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            timeout=15,
            headers={"User-Agent": "BreakoutScreener/1.0 research@example.com"},
        )
        if resp.status_code != 200:
            return []

        data     = resp.json()
        filings  = data.get("filings", {}).get("recent", {})
        forms    = filings.get("form", [])
        dates    = filings.get("filingDate", [])
        accnums  = filings.get("accessionNumber", [])
        items_l  = filings.get("items", [])          # 8-K Item-Codes
        primary  = filings.get("primaryDocument", [])

        cutoff  = datetime.now() - timedelta(days=30)
        results = []

        for i, form in enumerate(forms):
            if form not in FORM_LABELS:
                continue
            try:
                pub = datetime.strptime(dates[i], "%Y-%m-%d")
            except Exception:
                continue
            if pub < cutoff:
                continue

            # Link zum Filing
            acc_clean = accnums[i].replace("-", "")
            doc       = primary[i] if i < len(primary) else ""
            url       = (f"https://www.sec.gov/Archives/edgar/data/"
                         f"{int(cik)}/{acc_clean}/{doc}")

            # Beschreibung
            label = FORM_LABELS[form]
            item_code = items_l[i] if i < len(items_l) else ""
            if form == "8-K" and item_code:
                codes = [c.strip() for c in str(item_code).split(",")]
                descs = [ITEM_LABELS.get(c, c) for c in codes if c]
                if descs:
                    label = f"8-K: {', '.join(descs[:2])}"

            results.append({
                "title":   label,
                "summary": f"SEC Filing · {form} · {dates[i]}",
                "source":  "SEC EDGAR",
                "date":    pub.strftime("%d.%m.%Y"),
                "url":     url,
                "form":    form,
            })

            if len(results) >= 6:
                break

        return results
    except Exception as e:
        log.debug(f"SEC EDGAR error for {ticker}: {e}")
        return []


def analyze_ticker_news(ticker: str, row: dict) -> dict:
    """
    Holt Nachrichten aus allen Quellen und berechnet Sentiment + Empfehlung.
    Quellen: yfinance → Yahoo RSS → Google News + SEC EDGAR (immer parallel)
    """
    # ── News-Artikel: yfinance primär, Fallbacks ──────────────────────────
    articles = _fetch_yfinance_news(ticker)
    if not articles:
        articles = _fetch_rss_news(ticker)
    if not articles:
        articles = _fetch_google_news(ticker)

    # Google News immer ergänzend (andere Quellen als Yahoo)
    if articles:
        google = _fetch_google_news(ticker)
        # Deduplizierung: nur Artikel hinzufügen deren Titel noch nicht vorhanden
        existing_titles = {a["title"].lower()[:60] for a in articles}
        for g in google:
            if g["title"].lower()[:60] not in existing_titles:
                articles.append(g)
                existing_titles.add(g["title"].lower()[:60])

    # SEC EDGAR immer separat (strukturell anders, kein normales Sentiment)
    sec_filings = _fetch_sec_filings(ticker)

    if not articles and not sec_filings:
        return {
            "ticker": ticker,
            "articles": [],
            "sec_filings": [],
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "top_headlines": [],
            "recommendation": "ABWARTEN",
            "rec_color": "#ffd600",
            "summary": "Keine aktuellen Nachrichten gefunden.",
            "reason": "Rein technische Einschätzung auf Basis des Screener-Scores.",
            "n_articles": 0,
        }

    if not articles:
        # Nur SEC-Daten vorhanden
        return {
            "ticker":          ticker,
            "articles":        [],
            "sec_filings":     sec_filings,
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "top_headlines":   [],
            "recommendation":  "ABWARTEN",
            "rec_color":       "#ffd600",
            "summary":         "Keine Nachrichtenartikel, aber SEC-Filings vorhanden.",
            "reason":          "Rein technische Einschätzung auf Basis des Screener-Scores.",
            "n_articles":      0,
        }

    # Sentiment berechnen
    total_score = 0.0
    scored = []
    for a in articles:
        combined = a["title"] + " " + a["summary"]
        s = _score_text(combined)
        total_score += s
        scored.append((s, a))

    # Normalisieren
    avg_score = total_score / len(articles) if articles else 0

    if avg_score >= 1.5:
        sentiment_label = "positiv"
    elif avg_score <= -1.5:
        sentiment_label = "negativ"
    else:
        sentiment_label = "neutral"

    # Top-Headlines (die mit dem stärksten Sentiment)
    scored.sort(key=lambda x: abs(x[0]), reverse=True)
    top_headlines = [
        {"title": a["title"], "date": a["date"], "source": a["source"],
         "score": s, "url": a.get("url", "")}
        for s, a in scored[:4]
    ]

    # Empfehlung: kombiniere technischen Score + News-Sentiment
    tech_score  = float(row.get("score", 0))
    rsi         = float(row.get("rsi", 50))
    perf_30d    = float(row.get("perf_30d") or 0)

    # Gewichtetes Scoring
    tech_weight = tech_score / 100
    news_weight = max(-1, min(1, avg_score / 5))
    combined_w  = tech_weight * 0.6 + news_weight * 0.4
    rsi_penalty = 0.15 if rsi > 72 else 0
    final       = combined_w - rsi_penalty

    if final >= 0.45 and sentiment_label != "negativ":
        rec = "KAUFEN"
        rec_color = "#00c853"
    elif final <= 0.2 or sentiment_label == "negativ":
        rec = "MEIDEN"
        rec_color = "#ff1744"
    else:
        rec = "ABWARTEN"
        rec_color = "#ffd600"

    reason = _build_reason(ticker, rec, tech_score, avg_score, rsi, perf_30d, top_headlines)

    # Zusammenfassung
    pos_news = [a["title"] for s, a in scored if s > 0][:2]
    neg_news = [a["title"] for s, a in scored if s < 0][:2]
    neu_news = [a["title"] for s, a in scored if s == 0][:1]
    summary_parts = []
    if pos_news:
        summary_parts.append("📈 " + " · ".join(pos_news))
    if neg_news:
        summary_parts.append("📉 " + " · ".join(neg_news))
    if not pos_news and not neg_news and neu_news:
        summary_parts.append("📰 " + " · ".join(neu_news))
    summary = " &nbsp;|&nbsp; ".join(summary_parts) or "Keine eindeutigen Nachrichten."

    return {
        "ticker":          ticker,
        "articles":        articles,
        "sec_filings":     sec_filings,
        "sentiment_score": round(avg_score, 2),
        "sentiment_label": sentiment_label,
        "top_headlines":   top_headlines,
        "recommendation":  rec,
        "rec_color":       rec_color,
        "summary":         summary,
        "reason":          reason,
        "n_articles":      len(articles),
    }


def _build_reason(ticker, rec, tech_score, news_score, rsi, perf_30d, headlines) -> str:
    parts = []
    if tech_score >= 55:
        parts.append(f"Starkes technisches Signal (Score {tech_score:.0f}/100)")
    elif tech_score >= 40:
        parts.append(f"Moderates technisches Signal (Score {tech_score:.0f}/100)")
    else:
        parts.append(f"Schwaches technisches Signal (Score {tech_score:.0f}/100)")

    if news_score >= 2:
        parts.append("News-Sentiment klar positiv")
    elif news_score >= 0.5:
        parts.append("News-Sentiment leicht positiv")
    elif news_score <= -2:
        parts.append("News-Sentiment klar negativ — Vorsicht")
    elif news_score <= -0.5:
        parts.append("News-Sentiment leicht negativ")
    else:
        parts.append("News-Sentiment neutral")

    if rsi > 72:
        parts.append(f"RSI {rsi:.0f} — überkauft, Rücksetzer möglich")
    elif rsi < 35:
        parts.append(f"RSI {rsi:.0f} — überverkauft, mögliche Erholung")

    if perf_30d > 15:
        parts.append(f"Bereits +{perf_30d:.0f}% in 30T — erhöhtes Rückschlagrisiko")
    elif perf_30d < -10:
        parts.append(f"{perf_30d:.0f}% in 30T — schwache Kursentwicklung")

    return " · ".join(parts)


def analyze_news_top_n(df, n: int = 10) -> dict:
    """Analysiert die Top-N Aktien mit 4 parallelen Threads."""
    from tqdm import tqdm
    top = df.head(n)
    print(f"\n📰 Nachrichten-Analyse für Top {len(top)} Aktien (Yahoo Finance)...")

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(analyze_ticker_news, row["ticker"], row.to_dict()): row["ticker"]
            for _, row in top.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Nachrichten"):
            ticker = futures[fut]
            try:
                results[ticker] = fut.result()
            except Exception as e:
                results[ticker] = {"ticker": ticker, "error": str(e)}

    ok = sum(1 for v in results.values() if v.get("n_articles", 0) > 0)
    print(f"   ✓ {ok}/{len(top)} Ticker mit Nachrichten gefunden")
    return results


def build_news_html(news_results: dict, df) -> str:
    """Baut den HTML-Block für Nachrichten-Einschätzungen."""
    if not news_results:
        return ""

    cards = ""
    for _, row in df.head(len(news_results)).iterrows():
        ticker = row["ticker"]
        nr     = news_results.get(ticker, {})

        if not nr or "error" in nr:
            err = nr.get("error", "Fehler") if nr else "Keine Daten"
            cards += f'<div class="news-card"><div class="news-ticker">{ticker}</div><div style="color:#666;font-size:11px;padding:12px 0">⚠ {err}</div></div>'
            continue

        rec        = nr.get("recommendation", "ABWARTEN")
        rec_color  = nr.get("rec_color", "#ffd600")
        sentiment  = nr.get("sentiment_label", "neutral")
        sent_color = "#00c853" if sentiment == "positiv" else "#ff1744" if sentiment == "negativ" else "#ffd600"
        summary    = nr.get("summary", "—")
        reason     = nr.get("reason", "—")
        n_art      = nr.get("n_articles", 0)
        headlines  = nr.get("top_headlines", [])
        price      = row.get("price", 0)
        score      = row.get("score", 0)
        perf       = row.get("perf_30d")
        perf_s     = (f'<span style="color:{"#00c853" if (perf or 0)>=0 else "#ff1744"}">'
                      f'{("+" if (perf or 0)>=0 else "")}{perf:.1f}%</span>') if perf is not None else "—"

        # Top-Headlines als Liste
        hl_html = ""
        for h in headlines[:3]:
            s    = h.get("score", 0)
            c    = "#00c853" if s > 0 else "#ff4466" if s < 0 else "#888"
            dot  = "▲" if s > 0 else "▼" if s < 0 else "·"
            url  = h.get("url", "")
            title_part = (
                f'<a href="{url}" target="_blank" '
                f'style="color:#ccc;text-decoration:none;border-bottom:1px solid #333" '
                f'onmouseover="this.style.color=\'#00ff88\';this.style.borderBottomColor=\'#00ff88\'" '
                f'onmouseout="this.style.color=\'#ccc\';this.style.borderBottomColor=\'#333\'">'
                f'{h["title"]}</a>'
                if url else h["title"]
            )
            hl_html += (f'<div style="display:flex;gap:6px;margin-bottom:4px">'
                        f'<span style="color:{c};flex-shrink:0">{dot}</span>'
                        f'<span style="font-size:11px;line-height:1.5">'
                        f'{title_part}'
                        f'<span style="color:#555;margin-left:6px">{h["date"]} · {h["source"]}</span>'
                        f'</span></div>')

        cards += f"""
<div class="news-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <div>
      <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank"
         style="font-size:17px;font-weight:900;color:#fff;text-decoration:none;font-family:'Courier New',monospace"
         onmouseover="this.style.color='#00ff88'" onmouseout="this.style.color='#fff'">{ticker}</a>
      <span style="font-size:11px;color:#888;margin-left:10px">${price:.2f} · Score {score:.0f} · 30T: {perf_s}</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-shrink:0">
      <span style="font-size:9px;color:{sent_color};border:1px solid {sent_color}44;padding:2px 8px;border-radius:3px;text-transform:uppercase;letter-spacing:1px">{sentiment}</span>
      <span style="font-size:12px;font-weight:700;color:{rec_color};border:1px solid {rec_color}66;padding:4px 12px;border-radius:4px;letter-spacing:1px">{rec}</span>
    </div>
  </div>
  <div style="margin-bottom:10px">{hl_html}</div>
  <div style="font-size:10px;color:#555;border-top:1px solid #1a1a28;padding-top:8px;line-height:1.6">
    💡 {reason} &nbsp;<span style="color:#333">({n_art} Artikel)</span>
  </div>
</div>"""

    return f"""
<div style="margin-bottom:32px;font-family:'Courier New',monospace">
  <div style="font-size:11px;color:#888;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px">// Nachrichten &amp; Einschätzung</div>
  <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:6px;letter-spacing:1px">
    Top {len(news_results)} Aktien · Aktuelle Nachrichten
    <span style="font-size:11px;color:#555;font-weight:400;margin-left:10px">Yahoo Finance · Sentiment-Analyse</span>
  </div>
  <div style="font-size:10px;color:#555;margin-bottom:16px">
    ⚠ Automatische Analyse · Keine Anlageberatung · Eigene Recherche empfohlen
  </div>
  <div class="news-grid">{cards}</div>
</div>"""
