"""
╔══════════════════════════════════════════════════════════════════╗
║   S&P 500 · ADVANCED BREAKOUT SCREENER  v3.0                    ║
║   + S&P 500 MARKTGESUNDHEITS-ANALYSE (NEU)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  NEU in v3.0: S&P500 INDEX-ANALYSE                              ║
║  • Fear & Greed Index (live von CNN via API)                     ║
║  • Technische Chartanalyse des SPX (MA50/MA200, RSI, MACD)      ║
║  • Aktuelle Marktnachrichten-Zusammenfassung                     ║
║  • Gesamturteil: Kaufen / Abwarten / Vorsicht                   ║
║                                                                  ║
║  KERNPRINZIP v2.1:                                               ║
║  • Harte Vorfilter eliminieren schwache Aktien sofort            ║
║  • Failed Breakouts werden aktiv herausgefiltert                 ║
║  • Crossover nur gut wenn Preis DANACH weiter steigt             ║
║  • Kombinations-Boni für simultane Signalbestätigung             ║
║                                                                  ║
║  INDIKATOREN:                                                    ║
║  1. Trend-Qualität     (MA60 Crossover + Bestätigung)           ║
║  2. Breakout-Gesundheit(Hält der Breakout? Kein Failed Cross)   ║
║  3. Preisstruktur      (Höhere Hochs, Nähe zum Tageshoch)        ║
║  4. Momentum           (RSI-Anstieg, MACD-Kreuzung)             ║
║  5. Volumen-Qualität   (Spike bei steigendem Preis)              ║
║  6. Squeeze-Breakout   (BB-Squeeze + Ausbruch kombiniert)       ║
║  7. Relative Stärke    (besser als eigener Schnitt)              ║
║  8. Langfristtrend     (EMA200, 52W-Hoch-Nähe)                  ║
║                                                                  ║
║  VORFILTER (Aktie wird ignoriert wenn):                          ║
║  ✗ Preis > 15% unter MA60 (kein kurzfristiger Breakout möglich) ║
║  ✗ Letzter Schlusskurs < 1$ (Penny Stocks raus)                 ║
║  ✗ Zu wenig Handelsvolumen (Ø < 50.000 Stk/Tag)                 ║
║  ✗ Preis in den letzten 20 Tagen gesunken (negativer Trend)     ║
║  ✗ Preis in letzten 5 Tagen gefallen (kurzfr. Abschwung raus)   ║
║  ✗ Failed Breakout: Cross vor ≤5T aber seitdem >3% gefallen     ║
║                                                                  ║
║  Installation:                                                   ║
║    pip install yfinance pandas requests beautifulsoup4 tqdm      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# MODUL: S&P 500 MARKTGESUNDHEITS-ANALYSE  (NEU in v3.0)
# ══════════════════════════════════════════════════════════════════

def get_fear_greed_index() -> dict:
    """
    Lädt den aktuellen CNN Fear & Greed Index.
    Versucht mehrere Quellen – gibt dict mit value, label, description zurück.
    """
    result = {"value": None, "label": "N/A", "description": "Daten nicht verfügbar"}

    # Quelle 1: CNN direkt (inoffizielle API)
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score")
            rating = data.get("fear_and_greed", {}).get("rating", "")
            if score is not None:
                result["value"] = round(float(score), 1)
                result["label"] = rating.replace("_", " ").title()
                result["description"] = _fgi_description(result["value"])
                return result
    except Exception:
        pass

    # Quelle 2: Alternative feargreedmeter API
    try:
        url2 = "https://feargreedmeter.com/api/fgi"
        resp2 = requests.get(url2, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp2.status_code == 200:
            d2 = resp2.json()
            score = d2.get("value") or d2.get("score") or d2.get("fgi")
            if score is not None:
                result["value"] = round(float(score), 1)
                result["label"] = _fgi_label(result["value"])
                result["description"] = _fgi_description(result["value"])
                return result
    except Exception:
        pass

    # Quelle 3: Schätzung aus VIX-Daten
    try:
        vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if not vix.empty:
            vix_val = float(vix["Close"].iloc[-1])
            # VIX ↔ Fear/Greed: VIX 10 ≈ 85 Greed, VIX 20 ≈ 50 Neutral, VIX 40 ≈ 10 Fear
            estimated = max(0, min(100, round(110 - vix_val * 3, 1)))
            result["value"] = estimated
            result["label"] = _fgi_label(estimated) + " (VIX-Schätzung)"
            result["description"] = _fgi_description(estimated) + f" | VIX: {vix_val:.1f}"
            return result
    except Exception:
        pass

    return result


def _fgi_label(value: float) -> str:
    if value >= 75:   return "Extreme Gier"
    if value >= 55:   return "Gier"
    if value >= 45:   return "Neutral"
    if value >= 25:   return "Angst"
    return "Extreme Angst"


def _fgi_description(value: float) -> str:
    if value >= 75:
        return "Investoren sind übermäßig optimistisch – Vorsicht vor Überhitzung!"
    if value >= 55:
        return "Markt zeigt Kaufbereitschaft, moderate Risikofreude"
    if value >= 45:
        return "Ausgeglichene Stimmung, abwartende Haltung"
    if value >= 25:
        return "Investoren sind verunsichert – mögliche Kaufgelegenheiten entstehen"
    return "Panikstimmung – historisch oft ein Kontra-Indikator (Kaufzone)"


def analyze_sp500_health() -> dict:
    """
    Vollständige technische Analyse des S&P 500 Index (^GSPC).
    Berechnet: Trend, RSI, MACD, Bollinger Bands, Abstand zu MAs, VIX.
    Gibt ein strukturiertes dict zurück.
    """
    print("\n📊 Analysiere S&P 500 Marktgesundheit...")
    result = {
        "price":         None,
        "change_1d_pct": None,
        "change_5d_pct": None,
        "change_20d_pct":None,
        "ma50":          None,
        "ma200":         None,
        "above_ma50":    None,
        "above_ma200":   None,
        "golden_cross":  None,  # MA50 über MA200
        "pct_from_ma50": None,
        "pct_from_ma200":None,
        "rsi":           None,
        "rsi_trend":     None,
        "macd_hist":     None,
        "macd_bullish":  None,
        "bb_position":   None,  # 0..1 (wo im BB steht der Kurs)
        "vix":           None,
        "vix_label":     None,
        "trend_label":   None,
        "signals_bull":  [],
        "signals_bear":  [],
        "score":         None,  # 0-100: 50=neutral, >50=bullish, <50=bearish
        "verdict":       None,  # "KAUFEN" / "ABWARTEN" / "VORSICHT"
        "verdict_emoji": None,
        "verdict_color": None,
        "error":         None,
    }

    try:
        # ── Kursdaten laden ──────────────────────────────────────────────
        end   = datetime.today()
        start = end - timedelta(days=365)
        spx   = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
        vix_d = yf.download("^VIX",  start=start, end=end, progress=False, auto_adjust=True)

        if spx.empty or len(spx) < 60:
            result["error"] = "Zu wenig SPX-Daten"
            return result

        close = spx["Close"].squeeze()
        n = len(close)

        # ── Preisinfo ────────────────────────────────────────────────────
        price   = float(close.iloc[-1])
        p_1d    = float(close.iloc[-2])
        p_5d    = float(close.iloc[-6])  if n >= 6  else price
        p_20d   = float(close.iloc[-21]) if n >= 21 else price

        result["price"]          = round(price, 2)
        result["change_1d_pct"]  = round((price - p_1d)  / p_1d  * 100, 2)
        result["change_5d_pct"]  = round((price - p_5d)  / p_5d  * 100, 2)
        result["change_20d_pct"] = round((price - p_20d) / p_20d * 100, 2)

        # ── Moving Averages ──────────────────────────────────────────────
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        ma50_v  = float(ma50.iloc[-1])
        ma200_v = float(ma200.iloc[-1]) if n >= 200 else None

        result["ma50"]  = round(ma50_v, 2)
        result["ma200"] = round(ma200_v, 2) if ma200_v else None
        result["above_ma50"]  = price > ma50_v
        result["above_ma200"] = (price > ma200_v) if ma200_v else None
        result["pct_from_ma50"]  = round((price - ma50_v)  / ma50_v  * 100, 2)
        result["pct_from_ma200"] = round((price - ma200_v) / ma200_v * 100, 2) if ma200_v else None

        if ma200_v:
            result["golden_cross"] = ma50_v > ma200_v  # Golden Cross aktiv?

        # ── RSI ──────────────────────────────────────────────────────────
        delta    = close.diff()
        gain     = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
        loss     = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
        rs       = gain / loss.replace(0, np.nan)
        rsi_s    = 100 - (100 / (1 + rs))
        rsi_val  = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.iloc[-3])
        result["rsi"]       = round(rsi_val, 1)
        result["rsi_trend"] = "steigend" if rsi_val > rsi_prev else "fallend"
        # Verlauf (60 Tage)
        result["rsi_history"] = [round(float(v), 1) for v in rsi_s.iloc[-60:] if not np.isnan(v)]

        # ── MACD ─────────────────────────────────────────────────────────
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd_l   = ema12 - ema26
        macd_sig = macd_l.ewm(span=9, adjust=False).mean()
        macd_h   = macd_l - macd_sig
        hist_val = float(macd_h.iloc[-1])
        hist_prev= float(macd_h.iloc[-3])
        result["macd_hist"]    = round(hist_val, 2)
        result["macd_bullish"] = hist_val > 0 and hist_val > hist_prev
        # Verlauf (60 Tage)
        result["macd_history"] = [round(float(v), 2) for v in macd_h.iloc[-60:] if not np.isnan(v)]

        # ── Bollinger Bands ──────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_up = sma20 + 2 * std20
        bb_lo = sma20 - 2 * std20
        bb_range = float(bb_up.iloc[-1]) - float(bb_lo.iloc[-1])
        bb_pos = (price - float(bb_lo.iloc[-1])) / bb_range if bb_range > 0 else 0.5
        result["bb_position"] = round(bb_pos, 3)

        # ── VIX ──────────────────────────────────────────────────────────
        if not vix_d.empty:
            vix_close = vix_d["Close"].squeeze()
            vix_val = float(vix_close.iloc[-1])
            result["vix"] = round(vix_val, 2)
            if vix_val < 15:   result["vix_label"] = "Niedrig (Sorglosigkeit)"
            elif vix_val < 20: result["vix_label"] = "Normal"
            elif vix_val < 30: result["vix_label"] = "Erhöht (Unruhe)"
            elif vix_val < 40: result["vix_label"] = "Hoch (Angst)"
            else:              result["vix_label"] = "Extrem hoch (Panik)"
            # Verlauf (60 Tage)
            result["vix_history"] = [round(float(v), 2) for v in vix_close.iloc[-60:] if not np.isnan(v)]

        # ── Trend-Label (einfach) ────────────────────────────────────────
        if result["change_20d_pct"] and result["change_20d_pct"] > 3:
            result["trend_label"] = "📈 Aufwärtstrend"
        elif result["change_20d_pct"] and result["change_20d_pct"] < -3:
            result["trend_label"] = "📉 Abwärtstrend / Korrektur"
        else:
            result["trend_label"] = "↔ Seitwärtsbewegung"

        # ── SIGNAL-SAMMLUNG ──────────────────────────────────────────────
        bull = []
        bear = []

        if result["above_ma50"]:
            bull.append(f"Kurs über MA50 (+{result['pct_from_ma50']:.1f}%)")
        else:
            bear.append(f"Kurs unter MA50 ({result['pct_from_ma50']:.1f}%)")

        if result["above_ma200"] is not None:
            if result["above_ma200"]:
                bull.append(f"Kurs über MA200 (+{result['pct_from_ma200']:.1f}%)")
            else:
                bear.append(f"Kurs unter MA200 ({result['pct_from_ma200']:.1f}%)")

        if result["golden_cross"] is not None:
            if result["golden_cross"]:
                bull.append("Golden Cross aktiv (MA50 > MA200)")
            else:
                bear.append("Death Cross aktiv (MA50 < MA200)")

        if rsi_val > 50 and result["rsi_trend"] == "steigend":
            bull.append(f"RSI {rsi_val:.0f} steigend (bullisches Momentum)")
        elif rsi_val < 45:
            bear.append(f"RSI {rsi_val:.0f} – schwaches Momentum")
        elif rsi_val > 70:
            bear.append(f"RSI {rsi_val:.0f} – überkauft, Korrekturrisiko")

        if result["macd_bullish"]:
            bull.append("MACD-Histogramm positiv & steigend")
        else:
            bear.append("MACD-Histogramm negativ oder fallend")

        if bb_pos > 0.7:
            bear.append(f"Kurs im oberen BB-Bereich ({bb_pos*100:.0f}%) – überdehnt")
        elif bb_pos < 0.3:
            bull.append(f"Kurs im unteren BB-Bereich ({bb_pos*100:.0f}%) – mögliche Erholung")

        if result["vix"] is not None:
            if result["vix"] < 18:
                bull.append(f"VIX {result['vix']:.1f} – geringe Volatilität (ruhiger Markt)")
            elif result["vix"] > 25:
                bear.append(f"VIX {result['vix']:.1f} – erhöhte Angst im Markt")

        if result["change_5d_pct"] and result["change_5d_pct"] > 1.5:
            bull.append(f"5-Tages-Momentum +{result['change_5d_pct']:.1f}%")
        elif result["change_5d_pct"] and result["change_5d_pct"] < -1.5:
            bear.append(f"5-Tages-Momentum {result['change_5d_pct']:.1f}%")

        result["signals_bull"] = bull
        result["signals_bear"] = bear

        # ── GESAMT-SCORE & URTEIL ────────────────────────────────────────
        # Einfaches Scoring: jedes Bullish-Signal +1, jedes Bearish-Signal -1
        # Basis 50, dann normalisiert auf 0-100
        total_signals = len(bull) + len(bear)
        if total_signals > 0:
            raw_score = 50 + (len(bull) - len(bear)) / total_signals * 50
            score = round(max(0, min(100, raw_score)), 1)
        else:
            score = 50.0
        result["score"] = score

        if score >= 65:
            result["verdict"] = "KAUFEN"
            result["verdict_emoji"] = "🟢"
            result["verdict_color"] = "#00c853"
        elif score >= 48:
            result["verdict"] = "ABWARTEN"
            result["verdict_emoji"] = "🟡"
            result["verdict_color"] = "#ffd600"
        else:
            result["verdict"] = "VORSICHT"
            result["verdict_emoji"] = "🔴"
            result["verdict_color"] = "#ff1744"

    except Exception as e:
        result["error"] = str(e)

    return result


def print_market_health(spx: dict, fgi: dict) -> None:
    """Gibt die Marktanalyse schön formatiert in der Konsole aus."""
    W = 68
    sep = "═" * W

    print(f"\n╔{sep}╗")
    print(f"║{'S&P 500 MARKTGESUNDHEITS-ANALYSE · v3.0':^{W}}║")
    print(f"╠{sep}╣")

    # Preisinfo
    if spx["price"]:
        chg1 = spx["change_1d_pct"] or 0
        chg5 = spx["change_5d_pct"] or 0
        chg20= spx["change_20d_pct"]or 0
        arrow1 = "▲" if chg1 >= 0 else "▼"
        arrow5 = "▲" if chg5 >= 0 else "▼"
        arrow20= "▲" if chg20>= 0 else "▼"
        print(f"║  S&P 500 (^GSPC): ${spx['price']:,.2f}  "
              f"1T: {arrow1}{abs(chg1):.2f}%  "
              f"5T: {arrow5}{abs(chg5):.2f}%  "
              f"20T: {arrow20}{abs(chg20):.2f}%{' '*(W-62)}║")
    else:
        print(f"║  S&P 500 Daten nicht verfügbar{' '*(W-31)}║")

    # MA-Info
    if spx["ma50"]:
        ma50_sym  = "✓" if spx["above_ma50"]  else "✗"
        ma200_sym = "✓" if spx.get("above_ma200") else "✗" if spx.get("above_ma200") is False else "–"
        gc = "Golden Cross ✓" if spx.get("golden_cross") else ("Death Cross ✗" if spx.get("golden_cross") is False else "–")
        print(f"║  MA50: ${spx['ma50']:,.0f} [{ma50_sym}]  "
              f"MA200: ${spx['ma200']:,.0f} [{ma200_sym}]  "
              f"{gc}{' '*(W-60)}║")

    # RSI / MACD / VIX
    rsi_info  = f"RSI: {spx['rsi']:.0f} ({spx['rsi_trend']})" if spx["rsi"] else "RSI: N/A"
    macd_info = f"MACD-Hist: {'↑ Bullish' if spx['macd_bullish'] else '↓ Bearish'}" if spx["macd_hist"] is not None else ""
    vix_info  = f"VIX: {spx['vix']:.1f}" if spx["vix"] else ""
    print(f"║  {rsi_info}   {macd_info}   {vix_info}{' '*(W - len(rsi_info)-len(macd_info)-len(vix_info)-9)}║")

    print(f"╠{sep}╣")

    # Fear & Greed Index
    fgi_val   = fgi["value"]
    fgi_label = fgi["label"]
    fgi_bar   = _make_fgi_bar(fgi_val, width=30) if fgi_val is not None else "[N/A]"
    print(f"║  FEAR & GREED INDEX: {fgi_bar} {fgi_val or 'N/A'}  ← {fgi_label}{' '*(max(0, W-25-len(fgi_bar)-len(str(fgi_val or 'N/A'))-len(fgi_label)-3))}║")
    print(f"║  {fgi['description'][:W-4]}{' '*(W-4-len(fgi['description'][:W-4]))}║")

    print(f"╠{sep}╣")
    print(f"║  {'TECHNISCHE SIGNALE':^{W}}║")

    # Bullische Signale
    for s in spx["signals_bull"]:
        line = f"  ✅ {s}"
        print(f"║{line:<{W}}║")
    # Bärische Signale
    for s in spx["signals_bear"]:
        line = f"  ⚠  {s}"
        print(f"║{line:<{W}}║")

    print(f"╠{sep}╣")

    # Trend-Label
    tl = spx.get("trend_label", "")
    print(f"║  20T-Trend: {tl}{' '*(W-14-len(tl))}║")

    # Verdict
    v_emoji = spx.get("verdict_emoji", "")
    verdict = spx.get("verdict", "N/A")
    score   = spx.get("score", 50)
    score_bar = _make_score_bar(score, width=20)
    v_line = f"  {v_emoji} MARKT-URTEIL: {verdict}  [{score_bar}] {score:.0f}/100"
    print(f"║{v_line:<{W}}║")

    # Kaufempfehlung
    if verdict == "KAUFEN":
        tip = "  → Technisch intakter Markt: Breakout-Kandidaten sind chancenreich"
    elif verdict == "ABWARTEN":
        tip = "  → Unklare Lage: Nur Top-Setups mit starker Bestätigung kaufen"
    else:
        tip = "  → Vorsicht empfohlen: Positionen klein halten, Stops setzen!"
    print(f"║{tip:<{W}}║")

    print(f"╚{sep}╝\n")


def _make_fgi_bar(value: float, width: int = 30) -> str:
    """Erstellt einen farbfreien ASCII-Fortschrittsbalken für den FGI."""
    filled = int(round(value / 100 * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _make_score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    bar    = "█" * filled + "░" * (width - filled)
    return bar


def build_market_health_html(spx: dict, fgi: dict) -> str:
    """
    Erzeugt einen vollständigen HTML-Block für den Marktgesundheits-Abschnitt,
    der am Anfang des Breakout-Reports eingefügt wird.
    """
    if spx.get("error"):
        return f'<div class="mh-section"><p style="color:#666">Marktanalyse nicht verfügbar: {spx["error"]}</p></div>'

    # Verdict-Farbe
    verdict_color = spx.get("verdict_color", "#ffd600")
    verdict       = spx.get("verdict", "N/A")
    verdict_emoji = spx.get("verdict_emoji", "")
    score         = spx.get("score", 50)

    # FGI Gauge
    fgi_val   = fgi.get("value") or 50
    fgi_label = fgi.get("label", "N/A")
    fgi_desc  = fgi.get("description", "")
    fgi_color = ("#ff1744" if fgi_val < 25 else
                 "#ff6d00" if fgi_val < 45 else
                 "#ffd600" if fgi_val < 55 else
                 "#69f0ae" if fgi_val < 75 else
                 "#00c853")

    # Signale
    bull_html = "".join(f'<div class="mh-signal mh-bull">✅ {s}</div>' for s in spx.get("signals_bull", []))
    bear_html = "".join(f'<div class="mh-signal mh-bear">⚠ {s}</div>'  for s in spx.get("signals_bear", []))

    # MA-Info
    ma50_ok  = "✓" if spx.get("above_ma50") else "✗"
    ma200_ok = ("✓" if spx.get("above_ma200") else
                "✗"  if spx.get("above_ma200") is False else "—")
    gc_label = ("🌟 Golden Cross aktiv" if spx.get("golden_cross") else
                "💀 Death Cross aktiv"  if spx.get("golden_cross") is False else "—")

    chg1  = spx.get("change_1d_pct")  or 0
    chg5  = spx.get("change_5d_pct")  or 0
    chg20 = spx.get("change_20d_pct") or 0

    def _svg_line(values, width=200, height=56, color="#00ff88", fill_color=None, zero_line=False):
        """Erzeugt ein SVG-Liniendiagramm aus einer Werteliste."""
        if not values or len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn or 1
        pad = 4
        w, h = width, height
        pts = " ".join(
            f"{pad + (i / (len(values)-1)) * (w - 2*pad):.1f},"
            f"{pad + (1 - (v - mn) / rng) * (h - 2*pad):.1f}"
            for i, v in enumerate(values)
        )
        svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        # Optionale Nulllinie (für MACD)
        if zero_line and mn < 0 < mx:
            y0 = pad + (1 - (0 - mn) / rng) * (h - 2*pad)
            svg += f'<line x1="{pad}" y1="{y0:.1f}" x2="{w-pad}" y2="{y0:.1f}" stroke="#333" stroke-width="1" stroke-dasharray="3,3"/>'
        # Füllbereich
        if fill_color:
            first_x = pad
            last_x  = pad + (w - 2*pad)
            bottom  = h - pad
            svg += f'<polygon points="{first_x},{bottom} {pts} {last_x},{bottom}" fill="{fill_color}" opacity="0.12"/>'
        svg += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        # Letzter Punkt hervorheben
        last_x = pad + (w - 2*pad)
        last_y = pad + (1 - (values[-1] - mn) / rng) * (h - 2*pad)
        svg += f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" fill="{color}"/>'
        svg += '</svg>'
        return svg

    def _svg_bar(values, width=200, height=56):
        """MACD-Histogramm als Balken-Chart (grün/rot je nach Vorzeichen)."""
        if not values or len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn or 1
        w, h, pad = width, height, 4
        bar_w = max(1.5, (w - 2*pad) / len(values) - 0.8)
        svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        # Nulllinie
        y0 = pad + (1 - (0 - mn) / rng) * (h - 2*pad) if mn < 0 < mx else h - pad
        svg += f'<line x1="{pad}" y1="{y0:.1f}" x2="{w-pad}" y2="{y0:.1f}" stroke="#333" stroke-width="1"/>'
        for i, v in enumerate(values):
            x = pad + i / len(values) * (w - 2*pad)
            y_val = pad + (1 - (v - mn) / rng) * (h - 2*pad)
            bar_top  = min(y_val, y0)
            bar_h    = abs(y_val - y0)
            c = "#00c853" if v >= 0 else "#ff1744"
            svg += f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 0.5):.1f}" fill="{c}" opacity="0.8"/>'
        svg += '</svg>'
        return svg

    rsi_hist  = spx.get("rsi_history",  [])
    macd_hist = spx.get("macd_history", [])
    vix_hist  = spx.get("vix_history",  [])

    rsi_color  = "#ff9944" if spx.get("rsi", 50) > 70 else "#4488ff" if spx.get("rsi", 50) < 30 else "#00ff88"
    vix_color  = "#ff1744" if (spx.get("vix") or 0) > 30 else "#ffaa00" if (spx.get("vix") or 0) > 20 else "#00ff88"

    rsi_svg  = _svg_line(rsi_hist,  color=rsi_color, fill_color=rsi_color)
    macd_svg = _svg_bar(macd_hist)
    vix_svg  = _svg_line(vix_hist,  color=vix_color, fill_color=vix_color)

    rsi_cur  = spx.get("rsi", "—")
    vix_cur  = spx.get("vix", "—")
    macd_cur = spx.get("macd_hist", "—")
    macd_dir = "↑ bullish" if spx.get("macd_bullish") else "↓ bearish"

    def color_pct(v):
        c = "#69f0ae" if v >= 0 else "#ff1744"
        arrow = "▲" if v >= 0 else "▼"
        return f'<span style="color:{c}">{arrow}{abs(v):.2f}%</span>'

    return f"""
<style>
.mh-section {{
  background: #0d1117;
  border: 1px solid #222;
  border-radius: 12px;
  padding: 28px 32px;
  margin-bottom: 32px;
  font-family: 'Courier New', monospace;
}}
.mh-title {{
  font-size: 11px;
  color: #555;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 4px;
}}
.mh-header {{
  font-size: 22px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 24px;
  letter-spacing: 1px;
}}
.mh-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}}
.mh-card {{
  background: #111;
  border: 1px solid #1e1e1e;
  border-radius: 8px;
  padding: 16px;
}}
.mh-card-label {{
  font-size: 9px;
  color: #444;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 6px;
}}
.mh-card-top {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}}
.mh-card-val {{
  font-size: 24px;
  font-weight: 700;
  color: #fff;
}}
.mh-card-sub {{
  font-size: 11px;
  color: #aaa;
  margin-top: 6px;
}}
.mh-fgi-wrap {{
  background: #111;
  border: 1px solid #1e1e1e;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
}}
.mh-fgi-bar-bg {{
  height: 14px;
  background: linear-gradient(to right, #b71c1c, #e65100, #f9a825, #2e7d32, #1b5e20);
  border-radius: 7px;
  position: relative;
  margin: 12px 0 8px;
}}
.mh-fgi-needle {{
  position: absolute;
  top: -5px;
  width: 4px;
  height: 24px;
  background: #fff;
  border-radius: 2px;
  transform: translateX(-50%);
  left: {fgi_val}%;
  box-shadow: 0 0 6px rgba(255,255,255,0.8);
}}
.mh-fgi-labels {{
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: #444;
  letter-spacing: 1px;
}}
.mh-signals {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 24px;
}}
.mh-signal {{
  font-size: 11px;
  padding: 8px 12px;
  border-radius: 6px;
  line-height: 1.4;
}}
.mh-bull {{
  background: rgba(0,200,83,0.08);
  border: 1px solid rgba(0,200,83,0.2);
  color: #69f0ae;
}}
.mh-bear {{
  background: rgba(255,23,68,0.08);
  border: 1px solid rgba(255,23,68,0.2);
  color: #ff6d6d;
}}
.mh-verdict-wrap {{
  display: flex;
  align-items: center;
  gap: 24px;
  background: #111;
  border: 2px solid {verdict_color};
  border-radius: 10px;
  padding: 20px 24px;
}}
.mh-verdict-score-bar-bg {{
  height: 8px;
  background: #1e1e1e;
  border-radius: 4px;
  width: 180px;
  margin-top: 8px;
}}
.mh-verdict-score-bar-fill {{
  height: 8px;
  width: {score:.0f}%;
  background: {verdict_color};
  border-radius: 4px;
}}
.mh-verdict-main {{
  font-size: 28px;
  font-weight: 700;
  color: {verdict_color};
  letter-spacing: 2px;
}}
.mh-verdict-sub {{
  font-size: 11px;
  color: #555;
  margin-top: 4px;
  max-width: 420px;
  line-height: 1.6;
}}
.mh-ma-row {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.mh-ma-pill {{
  font-size: 11px;
  padding: 5px 12px;
  border-radius: 20px;
  border: 1px solid #222;
  color: #aaa;
  background: #111;
}}
.mh-chart-label {{
  font-size: 9px;
  color: #777;
  letter-spacing: 1px;
  margin-top: 4px;
  text-align: right;
}}
</style>

<div class="mh-section">
  <div class="mh-title">// S&P 500 Index-Analyse</div>
  <div class="mh-header">Marktgesundheits-Check <span style="font-size:14px;color:#888;font-weight:400">— Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')}</span></div>

  <!-- Zeile 1: Kurs + MAs + Urteil -->
  <div style="display:grid;grid-template-columns:auto 1fr auto;gap:20px;align-items:center;margin-bottom:20px;background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:16px 20px">
    <!-- Kurs -->
    <div>
      <div style="font-size:9px;color:#888;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">S&amp;P 500</div>
      <div style="font-size:28px;font-weight:700;color:#fff">${spx.get('price', 'N/A'):,.2f}</div>
      <div style="font-size:11px;color:#aaa;margin-top:2px">1T: {color_pct(chg1)} &nbsp;5T: {color_pct(chg5)} &nbsp;20T: {color_pct(chg20)}</div>
    </div>
    <!-- MA-Pills + FGI -->
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:0 16px">
      <div class="mh-ma-pill">MA50: ${spx.get('ma50','N/A'):,.0f} [{ma50_ok}] ({spx.get('pct_from_ma50',0):+.1f}%)</div>
      <div class="mh-ma-pill">MA200: ${spx.get('ma200','N/A'):,.0f} [{ma200_ok}] ({spx.get('pct_from_ma200',0):+.1f}%)</div>
      <div class="mh-ma-pill">{gc_label}</div>
      <div class="mh-ma-pill">{spx.get('trend_label','—')}</div>
      <div class="mh-ma-pill" style="color:{fgi_color};border-color:{fgi_color}44">F&amp;G: {fgi_val} — {fgi_label}</div>
    </div>
    <!-- Urteil -->
    <div style="text-align:right;border-left:1px solid #1e1e1e;padding-left:20px">
      <div style="font-size:9px;color:#888;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">Markt-Urteil</div>
      <div style="font-size:24px;font-weight:700;color:{verdict_color};letter-spacing:2px">{verdict_emoji} {verdict}</div>
      <div style="height:5px;background:#1e1e1e;border-radius:3px;width:120px;margin:6px 0 0 auto">
        <div style="height:5px;width:{score:.0f}%;background:{verdict_color};border-radius:3px"></div>
      </div>
      <div style="font-size:9px;color:#888;margin-top:3px">{score:.0f}/100</div>
    </div>
  </div>

  <!-- Zeile 2: RSI + MACD + VIX in einer Zeile -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">

    <div class="mh-card">
      <div class="mh-card-label">RSI (14) — 60 Tage</div>
      <div class="mh-card-top">
        <span class="mh-card-val" style="color:{rsi_color}">{rsi_cur}</span>
        <span style="font-size:11px;color:#aaa">RSI {spx.get('rsi_trend','')}</span>
      </div>
      {rsi_svg}
      <div class="mh-chart-label">
        <span style="color:#4488ff">▬ &lt;30 Überverkauft</span> &nbsp;
        <span style="color:#ff9944">▬ &gt;70 Überkauft</span>
      </div>
    </div>

    <div class="mh-card">
      <div class="mh-card-label">MACD-Histogramm — 60 Tage</div>
      <div class="mh-card-top">
        <span class="mh-card-val" style="color:{'#00c853' if spx.get('macd_bullish') else '#ff1744'}">{macd_cur}</span>
        <span style="font-size:11px;color:#aaa">{macd_dir}</span>
      </div>
      {macd_svg}
      <div class="mh-chart-label">
        <span style="color:#00c853">▬ Positiv (Bullish)</span> &nbsp;
        <span style="color:#ff1744">▬ Negativ (Bearish)</span>
      </div>
    </div>

    <div class="mh-card">
      <div class="mh-card-label">VIX — 60 Tage</div>
      <div class="mh-card-top">
        <span class="mh-card-val" style="color:{vix_color}">{vix_cur}</span>
        <span style="font-size:11px;color:#aaa">{spx.get('vix_label', '')}</span>
      </div>
      {vix_svg}
      <div class="mh-chart-label">
        <span style="color:#00ff88">▬ &lt;20 Normal</span> &nbsp;
        <span style="color:#ffaa00">▬ &gt;20 Erhöht</span> &nbsp;
        <span style="color:#ff1744">▬ &gt;30 Angst</span>
      </div>
    </div>

  </div>

  <!-- Fear & Greed (kompakt) -->
  <div class="mh-fgi-wrap" style="padding:14px 20px;margin-bottom:16px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div style="font-size:9px;color:#888;letter-spacing:2px;text-transform:uppercase">Fear &amp; Greed Index (CNN)</div>
      <div style="font-size:16px;font-weight:700;color:{fgi_color}">{fgi_val} — {fgi_label}</div>
    </div>
    <div class="mh-fgi-bar-bg" style="margin:0 0 6px">
      <div class="mh-fgi-needle"></div>
    </div>
    <div class="mh-fgi-labels">
      <span>0 EXTREME ANGST</span><span>25</span><span>50 NEUTRAL</span><span>75</span><span>100 EXTREME GIER</span>
    </div>
  </div>

  <!-- Signale -->
  <div class="mh-signals">
    {bull_html}
    {bear_html}
  </div>

  <!-- Verdict -->
  <div class="mh-verdict-wrap">
    <div>
      <div style="font-size:11px;color:#888;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">Markt-Urteil</div>
      <div class="mh-verdict-main">{verdict_emoji} {verdict}</div>
      <div class="mh-verdict-score-bar-bg">
        <div class="mh-verdict-score-bar-fill"></div>
      </div>
      <div style="font-size:9px;color:#888;margin-top:4px">{score:.0f}/100 Technische Stärke</div>
    </div>
    <div class="mh-verdict-sub">
      {"Technisch intakter Markt: Die Bedingungen für Breakout-Investitionen sind günstig. Starke Setups aus dem Screener haben erhöhte Erfolgswahrscheinlichkeit." if verdict == "KAUFEN" else
       "Unklare Marktlage: Nur hochwertige Setups mit mehrfacher Bestätigung in Betracht ziehen. Engere Stops setzen und Positionsgrößen reduzieren." if verdict == "ABWARTEN" else
       "Markt zeigt Schwächesignale oder befindet sich in einer Korrektur. Vorsicht bei Neukäufen! Bestehende Positionen schützen und auf klare Trendumkehr warten."}
    </div>
  </div>
</div>
"""

# ══════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════

LOOKBACK_DAYS  = 300
MA_PERIOD      = 60
MAX_TICKERS    = None   # None = alle
BATCH_SIZE     = 40
OUTPUT_CSV     = "breakout_results.csv"
OUTPUT_HTML    = "breakout_report.html"
TOP_N_CONSOLE  = 15

# ──────────────────────────────────────────────
# PERFORMANCE: PARALLELE DOWNLOADS + CACHE
# ──────────────────────────────────────────────
PARALLEL_WORKERS = 4          # Parallele Download-Threads
CACHE_DIR        = Path(".screener_cache")
CACHE_MAX_AGE_H  = 6          # Cache nach X Stunden ungültig

# ──────────────────────────────────────────────
# VORFILTER-SCHWELLWERTE
# ──────────────────────────────────────────────
MIN_PRICE               = 1.0    # Mindestpreis (Penny Stocks raus)
MIN_AVG_VOLUME          = 50_000 # Mindest-Durchschnittsvolumen
MAX_DIST_BELOW_MA60     = -0.15  # Max. 15% unter MA60 → sonst ignorieren
MIN_PRICE_TREND_20D     = -0.08  # Preis muss in 20T mind. > -8% sein (kein freier Fall)
MIN_PRICE_TREND_5D      = -0.05  # Preis muss in 5T > -5% sein
FAILED_BREAKOUT_THRESH  = -0.03  # Cross vor ≤5T aber seitdem > 3% gefallen → raus

# ──────────────────────────────────────────────
# SCORING-GEWICHTE  (gesamt max ~300 Punkte)
# KONZEPT: Kombinations-Boni machen den Unterschied
# ──────────────────────────────────────────────
WEIGHTS = {
    # ── TREND-SIGNAL (Kern) ──────────────────────────────────────
    "ma60_crossover_today":    60,   # Crossover genau HEUTE → Primärsignal
    "ma60_crossover_recent":   25,   # Crossover in letzten 1-3 Tagen (reduziert! Qualität entscheidet)
    "price_above_ma60_rising": 15,   # Über MA60 + MA60 selbst steigt (nachhaltiger Trend)

    # ── BREAKOUT-GESUNDHEIT (NEU) ────────────────────────────────
    # Entscheidet ob ein Recent Cross wirklich gut ist
    "breakout_holding":        35,   # Nach Cross: Preis heute >= Cross-Hoch (hält die Bewegung)
    "breakout_continuing":     20,   # Nach Cross: Preis steigt jeden Tag weiter (Continuation)

    # ── PREIS-STRUKTUR ───────────────────────────────────────────
    "close_near_day_high":     20,   # Schlusskurs > 70% der Tagesrange (kein Reversal!)
    "higher_highs_lows":       20,   # Höhere Hochs UND höhere Tiefs (Aufwärtsstruktur)
    "breakout_of_range":       25,   # Ausbruch aus mehrtägiger enger Range

    # ── MOMENTUM ────────────────────────────────────────────────
    "rsi_rising_above_50":     25,   # RSI > 50 UND steigt (nicht nur Niveau)
    "rsi_ideal_zone":          15,   # RSI 55–70 (ideal für Breakout-Continuation)
    "macd_fresh_crossover":    25,   # MACD kreuzt Signal nach oben (max 2 Tage alt)
    "macd_hist_accelerating":  15,   # MACD-Histogramm steigt 3 Tage hintereinander

    # ── VOLUMEN-QUALITÄT ─────────────────────────────────────────
    "volume_on_up_day":        30,   # Volumen-Spike (>150%) NUR an einem Up-Day
    "volume_dry_up_before":    15,   # Trocken vorher → Explosion heute
    "obv_new_high":            20,   # OBV auf neuem 30T-Hoch

    # ── SQUEEZE-BREAKOUT ─────────────────────────────────────────
    "squeeze_with_breakout":   30,   # BB-Squeeze + Preis bricht oberes BB
    "squeeze_building":        10,   # BB-Squeeze ohne Ausbruch (Vorbereitungsphase)

    # ── RELATIVE STÄRKE ──────────────────────────────────────────
    "outperforming_ma20":      15,   # Heute stärker als 20T-Durchschnitt
    "near_52w_high":           15,   # Innerhalb 5% vom 52W-Hoch
    "ema200_above":            10,   # Über EMA200 (Langfristtrend intakt)
    "adx_trending":            10,   # ADX > 25 (echter Trend)
}

# Kombinations-Bonus: Wenn mehrere Signale gleichzeitig feuern
COMBO_BONUSES = [
    # (Liste der benötigten Keys, Bonuspunkte, Label)
    (["ma60_crossover_today", "volume_on_up_day", "macd_fresh_crossover"],         40, "🚀 TRIPLE CONFIRM"),
    (["ma60_crossover_today", "volume_on_up_day"],                                 20, "⚡ CROSS+VOL"),
    (["squeeze_with_breakout", "volume_on_up_day"],                                20, "💥 SQUEEZE LAUNCH"),
    (["macd_fresh_crossover", "rsi_rising_above_50", "obv_new_high"],              15, "📊 FULL MOMENTUM"),
    (["ma60_crossover_recent", "breakout_holding", "breakout_continuing"],         25, "✅ CONFIRMED HOLD"),
    (["ma60_crossover_recent", "breakout_holding", "volume_on_up_day"],            20, "📈 HELD + VOL"),
    (["breakout_of_range", "volume_on_up_day"],                                    10, "📦 RANGE BREAK"),
]


# ══════════════════════════════════════════════
# SCHRITT 1: TICKER LADEN
# ══════════════════════════════════════════════

def get_tickers() -> list:
    print("\n📋 Lade S&P 500 Ticker-Liste...")
    tickers = _try_wikipedia() or _try_ishares_ivv() or _fallback_list()
    print(f"   ✓ {len(tickers)} Ticker geladen")
    return tickers


def _try_wikipedia() -> list:
    """S&P 500 Komponenten von Wikipedia – zuverlässigste Quelle."""
    try:
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        # Ticker stehen in der ersten Tabelle, Spalte 0
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if not table:
            return []
        tickers = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                t = cols[0].text.strip().replace(".", "-")  # BRK.B → BRK-B
                if t:
                    tickers.append(t)
        return tickers if len(tickers) > 400 else []
    except Exception:
        return []


def _try_ishares_ivv() -> list:
    """iShares IVV ETF Holdings = S&P 500 als Fallback."""
    url = ("https://www.ishares.com/us/products/239726/ISHARES-CORE-SP-500-ETF/"
           "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund")
    try:
        resp  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        lines = resp.text.split("\n")
        start = next((i for i, l in enumerate(lines)
                      if l.startswith("Ticker") or l.startswith('"Ticker"')), None)
        if start is None:
            return []
        tickers = []
        for line in lines[start + 1:]:
            t = line.split(",")[0].strip().strip('"')
            if t and 1 <= len(t) <= 5 and t != "Ticker":
                tickers.append(t)
        return tickers if len(tickers) > 400 else []
    except Exception:
        return []


def _fallback_list() -> list:
    """Hardcodierte S&P 500 Ticker-Liste (Stand 2024) als letzter Fallback."""
    print("   → Verwende eingebettete S&P 500-Liste...")
    return [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
        "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
        "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN",
        "APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET",
        "AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL",
        "BAC","BK","BBWI","BAX","BDX","WRB","BBY","BIO","TECH","BIIB","BLK","BX",
        "BA","BCR","BMY","AVGO","BR","BRO","BF.B","BLDR","BXP","CHRW","CDNS",
        "CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE",
        "CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR",
        "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME",
        "CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO",
        "CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS",
        "DHR","DRI","DVA","DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR",
        "DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN",
        "ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM",
        "EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE",
        "EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB",
        "FSLR","FE","FI","FLT","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX",
        "GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD",
        "GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES",
        "HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN",
        "HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP",
        "IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J",
        "JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI",
        "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN",
        "LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR",
        "MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET",
        "MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ",
        "MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM",
        "NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG",
        "NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL",
        "OTIS","PCAR","PKG","PLTR","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE",
        "PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR",
        "PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
        "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST",
        "RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS",
        "SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK",
        "SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
        "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV",
        "TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI",
        "UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI",
        "V","VST","VMC","WRK","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC",
        "WFC","WELL","WST","WDC","WHR","WMB","WTW","GWW","WYNN","XEL","XYL",
        "YUM","ZBRA","ZBH","ZTS",
    ]


# ══════════════════════════════════════════════
# SCHRITT 2: KURSDATEN LADEN
# ══════════════════════════════════════════════

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("screener")

# ── Cache-Hilfsfunktionen ────────────────────────────────────────────
def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"

def _cache_valid(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) / 3600 < CACHE_MAX_AGE_H

def _cache_load(key: str) -> dict | None:
    p = _cache_path(key)
    if not _cache_valid(p):
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if not data or len(data) < 400:
            return None
        # Jeder DataFrame bekommt eine eigene Kopie – verhindert
        # gemeinsame Speicherbereiche nach pickle.load()
        return {t: df.copy() for t, df in data.items()}
    except Exception:
        return None

def _cache_save(key: str, data: dict) -> None:
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        log.warning(f"Cache-Schreiben fehlgeschlagen: {e}")


def _extract_ticker_df(raw: pd.DataFrame, ticker: str, is_single: bool) -> pd.DataFrame | None:
    """
    Extrahiert einen einzelnen Ticker-DataFrame aus yf.download().
    Unterstützt altes Format (raw[ticker]) und neues MultiIndex-Format
    (Price, Ticker) aus yfinance ≥ 0.2.x.
    """
    try:
        if is_single:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        # Neues Format: MultiIndex Level-1 = Ticker
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(1):
                return raw.xs(ticker, level=1, axis=1).copy()
            if ticker in raw.columns.get_level_values(0):
                df = raw[ticker].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        # Altes Format
        if ticker in raw.columns:
            return raw[[ticker]].copy()
    except Exception:
        pass
    return None


import threading
_yf_lock = threading.Lock()  # yfinance ist nicht thread-safe


def _download_single(ticker: str, start_str: str, end_str: str) -> pd.DataFrame | None:
    """Lädt einen einzelnen Ticker — thread-safe durch Lock."""
    with _yf_lock:
        try:
            df = yf.download(ticker, start=start_str, end=end_str,
                             auto_adjust=True, progress=False)
        except Exception as e:
            log.debug(f"{ticker}: Download fehlgeschlagen: {e}")
            return None
    # Verarbeitung außerhalb des Locks
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(subset=["Close"])
        return df.copy() if len(df) >= 80 else None
    except Exception as e:
        log.debug(f"{ticker}: Verarbeitung fehlgeschlagen: {e}")
        return None


def _download_batch(batch: list, start_str: str, end_str: str) -> dict:
    result = {}
    for t in batch:
        df = _download_single(t, start_str, end_str)
        if df is not None:
            result[t] = df
    return result


def download_prices(tickers: list) -> dict:
    """
    Parallele Datenpipeline mit Tages-Cache.
    - Tageskerzen: Cache-first → parallele Batches (4 Threads)
    - Börse geschlossen: nur Tageskerzen (konsistente Vergleichsbasis)
    """
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]

    end       = datetime.today()
    start     = end - timedelta(days=LOOKBACK_DAYS)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Cache-Check ──────────────────────────────────────────────────
    cache_key  = f"daily_v3_{date.today().isoformat()}_{len(tickers)}"
    data       = _cache_load(cache_key)

    if data is not None:
        print(f"\n📥 Kursdaten aus Cache ({len(data)} Ticker) | {date.today()}")
        return data

    print(f"\n📥 Lade Kursdaten für {len(tickers)} Ticker ({start.date()} → {end.date()})...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    data    = {}

    # Downloads parallel (Lock in _download_single sorgt für yfinance thread-safety)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(_download_batch, b, start_str, end_str): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Downloading"):
            try:
                data.update(fut.result())
            except Exception as e:
                log.warning(f"Download-Future-Fehler: {e}")

    _cache_save(cache_key, data)
    print(f"   ✓ {len(data)} Ticker mit ausreichend Daten (Cache gespeichert)")
    return data


# ══════════════════════════════════════════════
# SCHRITT 3: INDIKATOREN
# ══════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series: pd.Series, period=20, std_dev=2.0):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bw    = (upper - lower) / sma
    return upper, lower, sma, bw


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def calc_adx(high, low, close, period=14):
    tr1  = high - low
    tr2  = (high - close.shift()).abs()
    tr3  = (low  - close.shift()).abs()
    tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, adjust=False).mean()
    up   = high - high.shift()
    down = low.shift() - low
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def pre_filter(ticker: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Harte Vorfilter: Gibt (True, "") zurück wenn Aktie OK ist,
    sonst (False, "Grund warum gefiltert").
    """
    close = df["Close"].squeeze()
    vol   = df.get("Volume", pd.Series(dtype=float))
    if isinstance(vol, pd.DataFrame):
        vol = vol.squeeze()
    n     = len(close)

    try:
        today_c = float(close.iloc[-1])
    except Exception:
        return False, "Close-Wert nicht lesbar"

    # 1) Mindestpreis
    if today_c < MIN_PRICE:
        return False, f"Preis zu niedrig (${today_c:.2f})"

    # 2) Mindestvolumen
    if vol is not None and len(vol) >= 20 and vol.sum() > 0:
        avg_vol = float(vol.iloc[-21:-1].mean())
        if avg_vol < MIN_AVG_VOLUME:
            return False, f"Volumen zu niedrig ({avg_vol:.0f})"

    # 3) Nicht zu weit unter MA60
    if n >= 60:
        ma60_val = float(close.rolling(60).mean().iloc[-1])
        dist = (today_c - ma60_val) / ma60_val
        if dist < MAX_DIST_BELOW_MA60:
            return False, f"Zu weit unter MA60 ({dist*100:.1f}%)"

    # 4) 20-Tage-Preis-Trend: kein freier Fall
    if n >= 20:
        price_20d_ago = float(close.iloc[-20])
        trend_20d = (today_c - price_20d_ago) / price_20d_ago
        if trend_20d < MIN_PRICE_TREND_20D:
            return False, f"Negativer 20T-Trend ({trend_20d*100:.1f}%)"

    # 5) NEU: 5-Tage-Trend – kein kurzfristiger Abschwung
    if n >= 5:
        price_5d_ago = float(close.iloc[-5])
        trend_5d = (today_c - price_5d_ago) / price_5d_ago
        if trend_5d < MIN_PRICE_TREND_5D:
            return False, f"Negativer 5T-Trend ({trend_5d*100:.1f}%)"

    # 6) NEU: Failed Breakout erkennen
    #    Suche MA60-Crossover in den letzten 5 Tagen.
    #    Wenn gefunden: prüfe ob Preis seitdem stark gefallen ist.
    if n >= 65:
        ma60_series = close.rolling(60).mean()
        for lookback in range(2, 6):   # 2 bis 5 Tage zurück
            if lookback + 1 >= n: continue
            c_before = float(close.iloc[-lookback - 1])
            c_at     = float(close.iloc[-lookback])
            m_before = float(ma60_series.iloc[-lookback - 1])
            m_at     = float(ma60_series.iloc[-lookback])
            was_cross = (c_before <= m_before) and (c_at > m_at)
            if was_cross:
                # Preis am Tag des Crossovers
                price_at_cross = c_at
                # Wie hat sich der Preis seitdem entwickelt?
                drawdown_since = (today_c - price_at_cross) / price_at_cross
                if drawdown_since < FAILED_BREAKOUT_THRESH:
                    return False, (f"Failed Breakout: Cross vor {lookback}T, "
                                   f"seitdem {drawdown_since*100:.1f}%")
                break  # Nur den letzten Crossover prüfen

    return True, ""


def analyze(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df = df.copy().sort_index()  # eigene Kopie – verhindert Seiteneffekte bei paralleler Analyse
        # MultiIndex-Schutz (yfinance ≥ 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df.get("Volume", pd.Series(dtype=float))
        if isinstance(vol, pd.DataFrame):
            vol = vol.squeeze()
        n     = len(close)
        if n < 80: return None

        # ── VORFILTER ────────────────────────────────────────────
        ok, reason = pre_filter(ticker, df)
        if not ok:
            return None

        # ── MOVING AVERAGES ──────────────────────────────────────
        ma60   = close.rolling(60).mean()
        ma20   = close.rolling(20).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        today_c    = float(close.iloc[-1])
        yest_c     = float(close.iloc[-2])
        today_h    = float(high.iloc[-1])
        today_l    = float(low.iloc[-1])
        today_ma60 = float(ma60.iloc[-1])
        yest_ma60  = float(ma60.iloc[-2])
        today_ema  = float(ema200.iloc[-1])

        if any(np.isnan([today_c, yest_c, today_ma60, yest_ma60])): return None

        # ── RSI ──────────────────────────────────────────────────
        rsi        = calc_rsi(close)
        today_rsi  = float(rsi.iloc[-1])
        yest_rsi   = float(rsi.iloc[-2])
        rsi_rising = today_rsi > yest_rsi

        # ── MACD ─────────────────────────────────────────────────
        macd_l, macd_sig, macd_hist = calc_macd(close)
        today_macd  = float(macd_l.iloc[-1])
        today_sig_v = float(macd_sig.iloc[-1])
        today_hist  = float(macd_hist.iloc[-1])
        yest_hist   = float(macd_hist.iloc[-2])
        prev_hist   = float(macd_hist.iloc[-3])
        yest_macd_l = float(macd_l.iloc[-2])
        yest_sig_l  = float(macd_sig.iloc[-2])

        # MACD Crossover in letzten 2 Tagen
        macd_cross_today  = (yest_macd_l <= yest_sig_l) and (today_macd > today_sig_v)
        macd_cross_yest   = (float(macd_l.iloc[-3]) <= float(macd_sig.iloc[-3])) and (yest_macd_l > yest_sig_l)
        macd_fresh_cross  = macd_cross_today or macd_cross_yest
        # Histogramm steigt 2 Tage hintereinander
        macd_accelerating = (today_hist > yest_hist) and (yest_hist > prev_hist)

        # ── BOLLINGER BANDS ──────────────────────────────────────
        bb_up, bb_lo, bb_mid, bb_bw = calc_bollinger(close)
        today_bw     = float(bb_bw.iloc[-1])
        hist_bw      = bb_bw.iloc[-60:-1]
        bw_pctile    = (hist_bw < today_bw).mean() if len(hist_bw) > 10 else 0.5
        # Squeeze = BW im untersten 20. Perzentil
        bb_squeeze   = bw_pctile < 0.20
        # Ausbruch über oberes BB
        bb_above_up  = today_c > float(bb_up.iloc[-1])

        # ── OBV ──────────────────────────────────────────────────
        obv_new_high_flag = False
        if vol is not None and len(vol) > 30 and vol.sum() > 0:
            obv          = calc_obv(close, vol)
            obv_30d_high = float(obv.iloc[-31:-1].max())
            obv_new_high_flag = float(obv.iloc[-1]) > obv_30d_high

        # ── ADX ──────────────────────────────────────────────────
        try:
            adx_s, plus_di, minus_di = calc_adx(high, low, close)
            adx_val    = float(adx_s.iloc[-1])
            plus_di_v  = float(plus_di.iloc[-1])
            minus_di_v = float(minus_di.iloc[-1])
        except Exception:
            adx_val    = 0.0
            plus_di_v  = 0.0
            minus_di_v = 0.0

        # ── VOLUMEN-ANALYSE ──────────────────────────────────────
        vol_ratio     = 0.0
        vol_spike_up  = False   # Spike NUR an Up-Day
        vol_dry_before = False   # Volumen vorher trocken, jetzt hoch

        if vol is not None and len(vol) >= 20 and vol.sum() > 0:
            avg_vol    = float(vol.iloc[-21:-1].mean())
            today_v    = float(vol.iloc[-1])
            vol_ratio  = today_v / avg_vol if avg_vol > 0 else 0.0
            is_up_day  = today_c > yest_c  # Preis ist heute gestiegen
            vol_spike_up = vol_ratio > 1.5 and is_up_day
            # Trocken = durchschnittliches Volumen der letzten 5 Tage war unter Durchschnitt
            avg_5d_prev = float(vol.iloc[-6:-1].mean())
            vol_dry_before = (avg_5d_prev < avg_vol * 0.8) and (today_v > avg_vol * 1.5)

        # ── 52W-HOCH ─────────────────────────────────────────────
        high_52w   = float(high.rolling(min(252, n)).max().iloc[-1])
        pct_52w    = (today_c / high_52w) if high_52w > 0 else 0.0
        near_52w   = pct_52w >= 0.95   # Innerhalb 5% (verschärft von 10% auf 5%)

        # ── PREIS-STRUKTUR ───────────────────────────────────────
        # 1) Schlusskurs nahe Tageshoch (kein Intraday-Reversal)
        day_range       = today_h - today_l
        close_vs_range  = (today_c - today_l) / day_range if day_range > 0 else 0.5
        close_near_high = close_vs_range >= 0.70   # Schlusskurs im oberen 30% der Tagesrange

        # 2) Höhere Hochs UND höhere Tiefs (echte Aufwärtsstruktur)
        highs_5 = [float(high.iloc[-i]) for i in range(1, 6)]
        lows_5  = [float(low.iloc[-i])  for i in range(1, 6)]
        higher_highs = highs_5[0] > highs_5[2] > highs_5[4]
        higher_lows  = lows_5[0]  > lows_5[2]  > lows_5[4]
        hh_hl        = higher_highs and higher_lows

        # 3) Ausbruch aus enger Range (Konsolidierung → Expansion)
        range_5d    = float(high.iloc[-6:-1].max()) - float(low.iloc[-6:-1].min())
        mid_5d      = float(close.iloc[-6:-1].mean())
        consol_5d   = range_5d / mid_5d if mid_5d > 0 else 999
        today_range = day_range / today_c if today_c > 0 else 0
        # War eng konsolidiert + heute expansion
        range_breakout = consol_5d < 0.04 and today_range > consol_5d * 1.5

        # 4) Konsolidierung allgemein (10 Tage)
        range_10d = float(high.iloc[-10:].max()) - float(low.iloc[-10:].min())
        mid_10d   = float(close.iloc[-10:].mean())
        consolidation = (range_10d / mid_10d if mid_10d > 0 else 999) < 0.05

        # ── MA-KREUZUNGEN ────────────────────────────────────────
        cross_up      = (yest_c <= yest_ma60) and (today_c > today_ma60)
        cross_down    = (yest_c >= yest_ma60) and (today_c < today_ma60)
        above_ma60    = today_c > today_ma60

        # MA60 selbst steigt (nachhaltiger Trend)
        ma60_rising   = float(ma60.iloc[-1]) > float(ma60.iloc[-5])

        # Crossover in letzten 1–3 Tagen (ohne heute)
        recent_cross = any(
            float(close.iloc[-j-1]) <= float(ma60.iloc[-j-1]) and
            float(close.iloc[-j])   > float(ma60.iloc[-j])
            for j in range(2, min(4, n-1))
        )

        # ── BREAKOUT-GESUNDHEIT ───────────────────────────────────
        # Für recent_cross: Wie hat sich die Aktie NACH dem Cross entwickelt?
        breakout_holding    = False   # Preis heute >= Preis am Cross-Tag
        breakout_continuing = False   # Preis steigt jeden Tag seit Cross

        if recent_cross and not cross_up:
            # Finde den genauen Cross-Tag
            cross_day_idx = None
            for j in range(2, min(4, n-1)):
                c_prev = float(close.iloc[-j-1])
                c_curr = float(close.iloc[-j])
                m_prev = float(ma60.iloc[-j-1])
                m_curr = float(ma60.iloc[-j])
                if c_prev <= m_prev and c_curr > m_curr:
                    cross_day_idx = -j
                    break

            if cross_day_idx is not None:
                price_at_cross = float(close.iloc[cross_day_idx])
                # Preis heute noch mindestens auf Cross-Niveau (Breakout hält)
                breakout_holding = today_c >= price_at_cross * 0.995

                # Preis ist jeden Tag seit dem Cross gestiegen (starke Continuation)
                days_since = abs(cross_day_idx)  # z.B. 2 oder 3
                if days_since >= 2:
                    prices_since = [float(close.iloc[cross_day_idx + i])
                                    for i in range(days_since + 1)]
                    breakout_continuing = all(
                        prices_since[i] <= prices_since[i+1]
                        for i in range(len(prices_since) - 1)
                    )

        # Relative Stärke: Heute-Performance vs. 5-Tage-Durchschnitt der Tagesrenditen
        returns_5d    = close.pct_change().iloc[-6:-1]
        today_ret     = (today_c - yest_c) / yest_c
        outperforming = today_ret > float(returns_5d.mean()) * 1.2

        # Über EMA200
        above_ema200  = today_c > today_ema and not np.isnan(today_ema)

        # ── SCORING ──────────────────────────────────────────────
        score   = 0
        details = {}

        def add(key, condition):
            nonlocal score
            pts = WEIGHTS.get(key, 0) if condition else 0
            score += pts
            details[key] = {"active": bool(condition), "pts": pts}

        # Trend
        add("ma60_crossover_today",    cross_up)
        add("ma60_crossover_recent",   recent_cross and not cross_up)
        add("price_above_ma60_rising", above_ma60 and ma60_rising
                                       and not cross_up and not recent_cross)

        # Breakout-Gesundheit (nur relevant bei recent_cross)
        add("breakout_holding",     breakout_holding)
        add("breakout_continuing",  breakout_continuing)

        # Preis-Struktur
        add("close_near_day_high",     close_near_high)
        add("higher_highs_lows",       hh_hl)
        add("breakout_of_range",       range_breakout)

        # Momentum
        add("rsi_rising_above_50",     today_rsi > 50 and rsi_rising)
        add("rsi_ideal_zone",          55 <= today_rsi <= 70)
        add("macd_fresh_crossover",    macd_fresh_cross)
        add("macd_hist_accelerating",  macd_accelerating)

        # Volumen
        add("volume_on_up_day",        vol_spike_up)
        add("volume_dry_up_before",    vol_dry_before)
        add("obv_new_high",            obv_new_high_flag)

        # Squeeze
        add("squeeze_with_breakout",   bb_squeeze and bb_above_up)
        add("squeeze_building",        bb_squeeze and not bb_above_up)

        # Relative Stärke / Langfrist
        add("outperforming_ma20",      outperforming)
        add("near_52w_high",           near_52w)
        add("ema200_above",            above_ema200)
        add("adx_trending",            adx_val > 25)

        # ── KOMBINATIONS-BONI ─────────────────────────────────────
        combo_score  = 0
        active_combos = []
        for keys, bonus, label in COMBO_BONUSES:
            if all(details.get(k, {}).get("active") for k in keys):
                combo_score += bonus
                active_combos.append(label)

        score += combo_score

        # ── SIGNAL-LABEL ──────────────────────────────────────────
        if cross_up:       signal = "CROSSOVER_UP"
        elif recent_cross: signal = "RECENT_CROSS"
        elif cross_down:   signal = "CROSSOVER_DOWN"
        elif above_ma60:   signal = "ABOVE_MA60"
        else:              signal = "BELOW_MA60"

        # ── SCORE NORMALISIERUNG (0–100) ──────────────────────────
        # Realistischer Max: bestes CROSSOVER_UP Szenario
        # (kann nicht gleichzeitig recent_cross UND cross_up haben)
        realistic_max = (
            WEIGHTS["ma60_crossover_today"]   +   # 60
            WEIGHTS["close_near_day_high"]    +   # 20
            WEIGHTS["higher_highs_lows"]      +   # 20
            WEIGHTS["breakout_of_range"]      +   # 25
            WEIGHTS["rsi_rising_above_50"]    +   # 25
            WEIGHTS["rsi_ideal_zone"]         +   # 15
            WEIGHTS["macd_fresh_crossover"]   +   # 25
            WEIGHTS["macd_hist_accelerating"] +   # 15
            WEIGHTS["volume_on_up_day"]       +   # 30
            WEIGHTS["volume_dry_up_before"]   +   # 15
            WEIGHTS["obv_new_high"]           +   # 20
            WEIGHTS["squeeze_with_breakout"]  +   # 30
            WEIGHTS["outperforming_ma20"]     +   # 15
            WEIGHTS["near_52w_high"]          +   # 15
            WEIGHTS["ema200_above"]           +   # 10
            WEIGHTS["adx_trending"]           +   # 10
            40 + 20 + 20                          # Top-3 Combo-Boni
        )  # ≈ 410 Punkte erreichbares Maximum
        score_100  = round(min(score / realistic_max * 100, 100.0), 1)

        if score_100 >= 60:   category = "🔥 SEHR STARK"
        elif score_100 >= 45: category = "📈 STARK"
        elif score_100 >= 30: category = "🟡 MITTEL"
        else:                 category = "⬇ SCHWACH"

        pct_from_ma60 = (today_c - today_ma60) / today_ma60 * 100
        perf_30d = (
            (today_c - float(close.iloc[-30])) / float(close.iloc[-30]) * 100
        ) if n >= 30 else None

        return {
            "ticker":        ticker,
            "signal":        signal,
            "category":      category,
            "score":         score_100,   # 0–100, das ist der Hauptwert
            "combo_score":   combo_score,
            "active_combos": ", ".join(active_combos),
            "price":          round(today_c, 2),
            "ma60":           round(today_ma60, 2),
            "ema200":         round(today_ema, 2) if not np.isnan(today_ema) else None,
            "pct_from_ma60":  round(pct_from_ma60, 2),
            "rsi":            round(today_rsi, 1),
            "macd_hist":      round(today_hist, 4),
            "adx":            round(adx_val, 1),
            "vol_ratio":      round(vol_ratio, 2),
            "bb_squeeze":     bb_squeeze,
            "obv_new_high":   obv_new_high_flag,
            "close_near_high": close_near_high,
            "breakout_holding":    breakout_holding,
            "breakout_continuing": breakout_continuing,
            "perf_30d":       round(perf_30d, 2) if perf_30d else None,
            "near_52w_high":  near_52w,
            "pct_52w":        round(pct_52w * 100, 1),
            "consolidation":  consolidation,
            "last_date":      df.index[-1].strftime("%Y-%m-%d"),
            "sparkline":      json.dumps([round(float(c), 2) for c in close.iloc[-30:]]),
            "details":        json.dumps(details),
        }

    except Exception:
        return None


def analyze_all(price_data: dict) -> pd.DataFrame:
    n = len(price_data)
    print(f"\n🔬 Analysiere {n} Ticker parallel ({PARALLEL_WORKERS} Threads)...")

    # Vorfilter zuerst (schnell, sequenziell) — mit Statistik
    passed, filter_reasons = [], {}
    for t, df in price_data.items():
        try:
            ok, reason = pre_filter(t, df)
            if ok:
                passed.append(t)
            else:
                key = reason.split("(")[0].strip()
                filter_reasons[key] = filter_reasons.get(key, 0) + 1
        except Exception as e:
            filter_reasons["Exception"] = filter_reasons.get("Exception", 0) + 1

    print(f"   📊 Vorfilter: {len(passed)} bestanden, {n - len(passed)} herausgefiltert")
    if filter_reasons:
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"      · {count:>3}× {reason}")

    if not passed:
        return pd.DataFrame()

    # Parallele Analyse der gefilterten Ticker
    passed_data = {t: price_data[t] for t in passed}
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(analyze, t, df): t for t, df in passed_data.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Analysiere"):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                log.warning(f"Analyse-Fehler: {e}")

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"   ✓ {len(df)} Aktien im Ergebnis")
    return df


# ══════════════════════════════════════════════
# SCHRITT 4: HTML-REPORT
# ══════════════════════════════════════════════

def build_html(df: pd.DataFrame, spx: dict = None, fgi: dict = None, total_tickers: int = 0) -> str:
    top     = df.head(150)
    # Marktgesundheits-Block (leer wenn nicht vorhanden)
    market_health_html = ""
    if spx is not None and fgi is not None:
        market_health_html = build_market_health_html(spx, fgi)

    def badge(signal):
        b = {
            "CROSSOVER_UP":   "background:rgba(0,255,136,0.15);color:#00ff88;border:1px solid rgba(0,255,136,0.45)",
            "RECENT_CROSS":   "background:rgba(0,255,136,0.08);color:#00dd77;border:1px solid rgba(0,255,136,0.2)",
            "ABOVE_MA60":     "background:rgba(68,136,255,0.12);color:#4488ff;border:1px solid rgba(68,136,255,0.3)",
            "BELOW_MA60":     "background:rgba(255,68,102,0.08);color:#ff4466;border:1px solid rgba(255,68,102,0.2)",
            "CROSSOVER_DOWN": "background:rgba(255,68,102,0.15);color:#ff4466;border:1px solid rgba(255,68,102,0.45)",
        }
        labels = {
            "CROSSOVER_UP":   "🔥 MA60 BREAKOUT ↑",
            "RECENT_CROSS":   "⚡ KÜRZL. BREAKOUT",
            "ABOVE_MA60":     "▲ ÜBER MA60",
            "BELOW_MA60":     "▼ UNTER MA60",
            "CROSSOVER_DOWN": "↓ MA60 BRUCH",
        }
        st = b.get(signal, "color:#888")
        lb = labels.get(signal, signal)
        return f'<span style="padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:0.5px;{st}">{lb}</span>'

    def indicator_pills(row):
        pills = []
        raw_d = row.get("details", {})
        try:
            d = json.loads(raw_d) if isinstance(raw_d, str) else (raw_d or {})
        except Exception:
            d = {}
        pill_map = [
            ("rsi_rising_above_50",    f"RSI↑{row['rsi']:.0f}"),
            ("rsi_ideal_zone",         f"RSI✓{row['rsi']:.0f}"),
            ("macd_fresh_crossover",   "MACD✗"),
            ("macd_hist_accelerating", "MACD↑↑"),
            ("breakout_holding",       "BRK-HOLD"),
            ("breakout_continuing",    "BRK-CONT"),
            ("squeeze_with_breakout",  "SQZ🚀"),
            ("squeeze_building",       "SQZ⏳"),
            ("obv_new_high",           "OBV-HIGH"),
            ("adx_trending",           f"ADX{row['adx']:.0f}"),
            ("volume_on_up_day",       f"VOL↑×{row['vol_ratio']:.1f}"),
            ("volume_dry_up_before",   "VOL-EXP"),
            ("ema200_above",           "EMA200✓"),
            ("near_52w_high",          f"52W{row['pct_52w']}%"),
            ("higher_highs_lows",      "HH+HL"),
            ("close_near_day_high",    "CLO-HIGH"),
            ("breakout_of_range",      "RNG-BRK"),
        ]
        colors = {
            "rsi_rising_above_50": "#ff9944", "rsi_ideal_zone": "#ff9944",
            "macd_fresh_crossover": "#ff44aa", "macd_hist_accelerating": "#ff44aa",
            "breakout_holding": "#00ff88", "breakout_continuing": "#00ffaa",
            "squeeze_with_breakout": "#00ffdd", "squeeze_building": "#44ffdd",
            "obv_new_high": "#4488ff",
            "adx_trending": "#ffdd44",
            "volume_on_up_day": "#ff6644", "volume_dry_up_before": "#ff8844",
            "ema200_above": "#88ff44",
            "near_52w_high": "#aa88ff",
            "higher_highs_lows": "#88ccff", "close_near_day_high": "#44ff88",
            "breakout_of_range": "#ffaa00",
        }
        for key, label in pill_map:
            if d.get(key, {}).get("active"):
                c = colors.get(key, "#888")
                pills.append(
                    f'<span style="background:rgba(255,255,255,0.05);border:1px solid {c}44;'
                    f'color:{c};padding:2px 7px;border-radius:3px;font-size:9px;margin:1px;'
                    f'display:inline-block">{label}</span>'
                )
        # Combo-Badge
        combos = row.get("active_combos", "")
        if combos:
            for c_label in combos.split(", "):
                if c_label.strip():
                    pills.append(
                        f'<span style="background:rgba(255,215,0,0.1);border:1px solid gold;'
                        f'color:gold;padding:2px 7px;border-radius:3px;font-size:9px;margin:1px;'
                        f'display:inline-block;font-weight:700">{c_label.strip()}</span>'
                    )
        return "".join(pills)

    def spark(prices_json):
        try:
            prices = json.loads(prices_json) if isinstance(prices_json, str) else prices_json
        except Exception:
            return ""
        if not prices or len(prices) < 2: return ""
        mn, mx = min(prices), max(prices)
        rng = mx - mn or 1
        pts = " ".join(
            f"{(i/(len(prices)-1))*100:.1f},{36-((p-mn)/rng)*32-2:.1f}"
            for i, p in enumerate(prices)
        )
        c = "#00ff88" if prices[-1] >= prices[0] else "#ff4466"
        return (f'<svg width="100" height="36" viewBox="0 0 100 36">'
                f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/></svg>')

    def prob_bar(prob):
        bar_c = "#00ff88" if prob >= 60 else "#ffaa00" if prob >= 45 else "#ff9944" if prob >= 30 else "#ff4466"
        return (f'<div style="display:flex;align-items:center;gap:8px;min-width:140px">'
                f'<div style="flex:1;height:5px;background:#1a1a2e;border-radius:3px;overflow:hidden">'
                f'<div style="width:{prob}%;height:100%;background:{bar_c};border-radius:3px"></div></div>'
                f'<span style="color:{bar_c};font-size:11px;font-weight:700;min-width:35px">{prob}%</span>'
                f'</div>')

    def pct(val, show_sign=True):
        if val is None: return "—"
        c = "#00ff88" if val >= 0 else "#ff4466"
        s = "+" if val >= 0 and show_sign else ""
        return f'<span style="color:{c}">{s}{val:.2f}%</span>'

    rows_html = ""
    for i, row in top.iterrows():
        has_border = row["signal"] in ("CROSSOVER_UP", "RECENT_CROSS")
        border_s   = ' style="border-left:3px solid #00ff88"' if has_border else ""
        rows_html += f"""
        <tr{border_s} class="data-row"
            data-signal="{row['signal']}"
            data-category="{row['category']}"
            data-score="{row['score']}">
          <td style="padding:10px 14px;font-weight:900;font-size:14px;font-family:'Courier New',monospace"><a href="https://finance.yahoo.com/quote/{row['ticker']}" target="_blank" style="color:#fff;text-decoration:none;border-bottom:1px solid #333;padding-bottom:1px" onmouseover="this.style.color='#00ff88';this.style.borderBottomColor='#00ff88'" onmouseout="this.style.color='#fff';this.style.borderBottomColor='#333'">{row['ticker']}</a></td>
          <td style="padding:10px 14px">{badge(row['signal'])}</td>
          <td style="padding:10px 14px">{prob_bar(row['score'])}</td>
          <td style="padding:10px 14px;color:#fff;font-weight:700">${row['price']:.2f}</td>
          <td style="padding:10px 14px;color:#aaa;font-size:11px">${row['ma60']:.2f}</td>
          <td style="padding:10px 14px">{pct(row['pct_from_ma60'])}</td>
          <td style="padding:10px 14px;color:{'#ff9944' if 55<=row['rsi']<=70 else '#e8e8f0'}">{row['rsi']:.0f}</td>
          <td style="padding:10px 14px">{pct(row['perf_30d'])}</td>
          <td style="padding:10px 14px">{spark(row['sparkline'])}</td>
          <td style="padding:10px 4px">{indicator_pills(row)}</td>
          <td style="padding:10px 14px;font-weight:800;font-size:15px;color:{'#00ff88' if row['score']>=60 else '#ffaa00' if row['score']>=45 else '#ff9944' if row['score']>=30 else '#ff4466'}">{row['score']:.1f}</td>
        </tr>"""

    crossovers = len(df[df["signal"].isin(["CROSSOVER_UP", "RECENT_CROSS"])])
    sehr_stark = len(df[df["category"] == "🔥 SEHR STARK"])
    stark      = len(df[df["category"] == "📈 STARK"])
    analyzed   = len(df)
    gen_time   = datetime.now().strftime("%d.%m.%Y %H:%M")

    legend_items = [
        ("RSI↑",      "RSI > 50 UND steigt – frisches Momentum",          "#ff9944"),
        ("MACD✗",     "MACD kreuzt Signal nach oben (max. 2T alt)",        "#ff44aa"),
        ("MACD↑↑",    "MACD-Histogramm steigt 3 Tage in Folge",           "#ff44aa"),
        ("BRK-HOLD",  "Nach Cross: Preis hält das Cross-Niveau (≥0%)",     "#00ff88"),
        ("BRK-CONT",  "Nach Cross: Preis steigt jeden Tag weiter",         "#00ffaa"),
        ("SQZ🚀",     "Bollinger Squeeze + Ausbruch über oberes Band",     "#00ffdd"),
        ("OBV-HIGH",  "OBV auf neuem 30-Tage-Hoch (Inst. kaufen)",        "#4488ff"),
        ("VOL↑×n",    "Volumen-Spike > 150% NUR an einem Up-Day",         "#ff6644"),
        ("VOL-EXP",   "Volumen vorher trocken, jetzt Explosion",           "#ff8844"),
        ("HH+HL",     "Höhere Hochs UND höhere Tiefs (echte Struktur)",    "#88ccff"),
        ("CLO-HIGH",  "Schlusskurs im oberen 30% der Tagesrange",          "#44ff88"),
        ("RNG-BRK",   "Ausbruch aus enger 5-Tage-Konsolidierung",         "#ffaa00"),
        ("EMA200✓",   "Preis über EMA200 (Langfristtrend intakt)",         "#88ff44"),
        ("52W%",      "Innerhalb 5% vom 52-Wochen-Hoch",                  "#aa88ff"),
    ]
    legend_html = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="background:rgba(255,255,255,0.05);border:1px solid {c}44;color:{c};'
        f'padding:2px 8px;border-radius:3px;font-size:10px;min-width:80px;text-align:center">'
        f'{name}</span><span style="font-size:11px;color:#aaa">{desc}</span></div>'
        for name, desc, c in legend_items
    )

    combo_legend = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="background:rgba(255,215,0,0.1);border:1px solid gold;color:gold;'
        f'padding:2px 8px;border-radius:3px;font-size:10px;min-width:140px;text-align:center;font-weight:700">'
        f'{label}</span><span style="font-size:11px;color:#aaa">+{bonus} Pkt</span></div>'
        for keys, bonus, label in COMBO_BONUSES
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout Screener v3.0 · S&P 500</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#09090f;color:#e8e8f0;font-family:'Space Mono',monospace;min-height:100vh}}
  body::before{{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(0,255,136,0.02) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(0,255,136,0.02) 1px,transparent 1px);
    background-size:44px 44px;pointer-events:none;z-index:0}}
  .wrap{{position:relative;z-index:1;max-width:1600px;margin:0 auto;padding:40px 24px}}
  header{{border-left:3px solid #00ff88;padding-left:20px;margin-bottom:40px}}
  .tag{{font-size:10px;letter-spacing:4px;color:#00ff88;text-transform:uppercase;margin-bottom:8px}}
  h1{{font-family:'Syne',sans-serif;font-size:clamp(28px,4vw,50px);font-weight:800;color:#fff;line-height:1.1}}
  h1 em{{color:#00ff88;font-style:normal}}
  .version{{display:inline-block;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);
            color:#00ff88;font-size:10px;padding:3px 10px;border-radius:4px;margin-left:12px;vertical-align:middle}}
  .meta{{margin-top:10px;font-size:11px;color:#999;letter-spacing:1px}}
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:36px}}
  .stat{{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:16px 20px}}
  .slabel{{font-size:9px;letter-spacing:2px;color:#999;text-transform:uppercase;margin-bottom:6px}}
  .sval{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800}}
  .green{{color:#00ff88}}.blue{{color:#4488ff}}.orange{{color:#ffaa00}}.white{{color:#fff}}
  .controls{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
  .search{{background:#0f0f18;border:1px solid #1a1a28;color:#e8e8f0;font-family:'Space Mono',monospace;
           font-size:12px;padding:10px 16px;border-radius:6px;width:220px;outline:none;letter-spacing:1px}}
  .search:focus{{border-color:#00ff88}}
  .filters{{display:flex;gap:6px;flex-wrap:wrap}}
  .fbtn{{background:#0f0f18;border:1px solid #1a1a28;color:#444;font-family:'Space Mono',monospace;
         font-size:10px;padding:8px 14px;border-radius:4px;cursor:pointer;letter-spacing:1px;transition:all .2s}}
  .fbtn.on,.fbtn:hover{{border-color:#00ff88;color:#00ff88}}
  .fbtn.on{{background:rgba(0,255,136,0.08)}}
  .table-wrap{{overflow-x:auto;border-radius:8px;border:1px solid #1a1a28}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead tr{{background:#0f0f18;border-bottom:1px solid #1a1a28}}
  th{{padding:12px 14px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;
      color:#999;cursor:pointer;white-space:nowrap;user-select:none}}
  th:hover{{color:#888}}
  tbody tr{{border-bottom:1px solid rgba(26,26,40,.6);transition:background .1s}}
  tbody tr:hover{{background:#0f0f18}}
  tbody tr:last-child{{border-bottom:none}}
  .legend-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px}}
  .legend{{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:24px}}
  .legend h3{{font-family:'Syne',sans-serif;font-size:12px;letter-spacing:2px;color:#888;
              text-transform:uppercase;margin-bottom:16px}}
  .legend-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:4px}}
  .footer{{margin-top:24px;font-size:10px;color:#666;letter-spacing:1px;text-align:center}}
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
  .data-row{{animation:fadeUp .3s ease both}}
  @media(max-width:768px){{.legend-wrap{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="tag">S&P 500 · NYSE/NASDAQ · Advanced Screener</div>
    <h1>Breakout <em>Intelligence</em> <span class="version">v3.0</span></h1>
    <div class="meta">// {analyzed} VON {total_tickers} AKTIEN NACH VORFILTER · 10+ INDIKATOREN · KOMBINATIONS-BONI · {gen_time}</div>
  </header>

  <div class="stats">
    <div class="stat"><div class="slabel">MA60 Breakouts</div><div class="sval green">{crossovers}</div></div>
    <div class="stat"><div class="slabel">Sehr stark ≥60</div><div class="sval orange">{sehr_stark}</div></div>
    <div class="stat"><div class="slabel">Stark ≥45</div><div class="sval blue">{stark}</div></div>
    <div class="stat"><div class="slabel">Score-Skala</div><div class="sval white">0–100</div></div>
    <div class="stat"><div class="slabel">Analysiert</div><div class="sval white">{analyzed} <span style="font-size:14px;color:#999">/ {total_tickers}</span></div></div>
  </div>

  {market_health_html}

  <div class="controls">
    <input class="search" type="text" id="searchBox" placeholder="TICKER SUCHEN..." oninput="filter()"/>
    <div class="filters">
      <button class="fbtn on" onclick="setF('all',this)">ALLE</button>
      <button class="fbtn" onclick="setF('crossover',this)">🔥 BREAKOUT</button>
      <button class="fbtn" onclick="setF('combo',this)">🚀 MIT BONUS</button>
      <button class="fbtn" onclick="setF('sehr_stark',this)">⚡ SEHR STARK</button>
      <button class="fbtn" onclick="setF('stark',this)">📈 STARK</button>
    </div>
  </div>

  <div class="table-wrap">
    <table id="mainTable">
      <thead><tr>
        <th onclick="sort(0)">TICKER</th>
        <th onclick="sort(1)">SIGNAL</th>
        <th onclick="sort(2)">SCORE-BALKEN</th>
        <th onclick="sort(3)">KURS</th>
        <th>MA60</th>
        <th onclick="sort(5)">ABST. MA60</th>
        <th onclick="sort(6)">RSI</th>
        <th onclick="sort(7)">30T PERF</th>
        <th>CHART <span style="font-size:8px;color:#888;font-weight:400">(30T)</span></th>
        <th>AKTIVE INDIKATOREN</th>
        <th onclick="sort(10)">SCORE (0–100)</th>
      </tr></thead>
      <tbody id="tbody">{rows_html}</tbody>
    </table>
  </div>

  <div class="legend-wrap">
    <div class="legend">
      <h3>// Indikator-Legende</h3>
      <div class="legend-grid">{legend_html}</div>
    </div>
    <div class="legend">
      <h3>// Kombinations-Boni (Gold-Badges)</h3>
      {combo_legend}
      <div style="margin-top:16px;font-size:10px;color:#777;line-height:1.8">
        Kombinations-Boni werden zusätzlich zum Basis-Score vergeben<br>
        wenn mehrere starke Signale gleichzeitig feuern.<br>
        Dies erhöht die Selektivität und filtert Fehlsignale heraus.
      </div>
    </div>
  </div>

  <div class="footer">
    // S&P 500 Advanced Breakout Screener v3.1 · Daten: Yahoo Finance via yfinance<br>
    // Parallel-Downloads · Tages-Cache · MultiIndex-Safe · Nicht als Anlageberatung zu verstehen
  </div>
</div>
<script>
let curF = 'all';
const dirs = {{}};

function setF(f, btn) {{
  curF = f;
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  filter();
}}

function filter() {{
  const q = (document.getElementById('searchBox').value || '').toUpperCase();
  document.querySelectorAll('#tbody .data-row').forEach(row => {{
    const ticker = row.cells[0]?.textContent || '';
    const sig = row.dataset.signal || '';
    const cat = row.dataset.category || '';
    const indicators = row.cells[9]?.textContent || '';

    let mf = true;
    if (curF === 'crossover') mf = sig.includes('CROSSOVER_UP') || sig.includes('RECENT_CROSS');
    if (curF === 'combo')     mf = indicators.includes('🚀') || indicators.includes('⚡') || indicators.includes('💥') || indicators.includes('📊') || indicators.includes('📈') || indicators.includes('📦');
    if (curF === 'sehr_stark') mf = cat.includes('SEHR STARK');
    if (curF === 'stark') mf = cat.includes('STARK') || cat.includes('SEHR STARK');

    const mq = !q || ticker.includes(q);
    row.style.display = (mf && mq) ? '' : 'none';
  }});
}}

function sort(col) {{
  dirs[col] = -(dirs[col] || 1);
  const tbody = document.getElementById('tbody');
  const rows = Array.from(tbody.querySelectorAll('.data-row'));
  rows.sort((a, b) => {{
    const av = a.cells[col]?.textContent.replace(/[^0-9.\\-]/g,'') || '';
    const bv = b.cells[col]?.textContent.replace(/[^0-9.\\-]/g,'') || '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dirs[col];
    return av.localeCompare(bv) * dirs[col];
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("╔" + "═"*60 + "╗")
    print("║  ADVANCED BREAKOUT SCREENER v3.0 · S&P 500" + " "*17 + "║")
    print("║  + S&P 500 Marktgesundheits-Analyse (NEU)" + " "*18 + "║")
    print("╚" + "═"*60 + "╝")

    # ── SCHRITT 0: S&P 500 MARKTANALYSE (NEU) ──────────────────────
    fgi_data = get_fear_greed_index()
    spx_data = analyze_sp500_health()
    print_market_health(spx_data, fgi_data)

    # Kaufempfehlung in Konsole
    verdict = spx_data.get("verdict", "ABWARTEN")
    if verdict == "VORSICHT":
        print("  ⚠  MARKT IN KORREKTUR – Breakout-Kandidaten mit erhöhter Vorsicht bewerten!")
        print("  ⚠  Nur Top-Scores in Betracht ziehen und enge Stop-Loss setzen.\n")
    elif verdict == "ABWARTEN":
        print("  💡 Markt unentschieden – Nur die stärksten Breakout-Signale verfolgen.\n")
    else:
        print("  ✅ Marktbedingungen günstig – Breakout-Kandidaten haben gute Chancen.\n")

    # ── SCHRITT 1-3: AKTIEN-SCREENER ───────────────────────────────
    tickers    = get_tickers()
    price_data = download_prices(tickers)
    df_results = analyze_all(price_data)

    if len(df_results) == 0:
        print("\n⚠  Keine Aktien nach Vorfilter übrig. Versuche MAX_DIST_BELOW_MA60 zu lockern.")
        return

    # CSV
    csv_cols = ["ticker", "signal", "category", "score", "combo_score", "active_combos",
                "price", "ma60", "pct_from_ma60", "rsi", "macd_hist",
                "adx", "vol_ratio", "bb_squeeze", "obv_new_high", "near_52w_high",
                "consolidation", "perf_30d", "last_date"]
    df_results[csv_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV: {OUTPUT_CSV}")

    # HTML (jetzt mit Marktgesundheitsblock)
    html = build_html(df_results, spx=spx_data, fgi=fgi_data, total_tickers=len(price_data))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML: {OUTPUT_HTML}")
    print(f"   → file://{os.path.abspath(OUTPUT_HTML)}")

    # Konsolen-Ausgabe
    print(f"\n{'═'*75}")
    print(f"  🏆 TOP {TOP_N_CONSOLE} BREAKOUT-KANDIDATEN  (Score 0–100)")
    print(f"{'═'*75}")
    print(f"  {'TICKER':<8} {'SIGNAL':<16} {'SCORE':>5}  {'KURS':>7}  "
          f"{'RSI':>5}  {'VOL×':>5}  KOMBOS")
    print(f"  {'─'*8} {'─'*16} {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*25}")

    for _, r in df_results.head(TOP_N_CONSOLE).iterrows():
        sig_s = (r['signal']
                 .replace("CROSSOVER_UP", "CROSS↑")
                 .replace("RECENT_CROSS", "RECENT↑")
                 .replace("ABOVE_MA60", "ABOVE")
                 .replace("BELOW_MA60", "BELOW")
                 .replace("CROSSOVER_DOWN", "CROSS↓"))
        combos   = r.get("active_combos", "") or "—"
        cat_icon = r['category'].split()[0]
        print(f"  {r['ticker']:<8} {sig_s:<16} {r['score']:>4.1f}  "
              f"${r['price']:>6.2f}  {r['rsi']:>5.0f}  "
              f"{r['vol_ratio']:>4.1f}×  {cat_icon} {combos}")

    print(f"\n  ℹ  Score 0–100 · ≥60 = Sehr stark · ≥45 = Stark · ≥30 = Mittel")
    print(f"  ℹ  Failed-Breakout-Filter: Cross vor ≤5T und >{abs(FAILED_BREAKOUT_THRESH)*100:.0f}% gefallen → ignoriert")
    print(f"  ℹ  5T-Trend-Filter: >{abs(MIN_PRICE_TREND_5D)*100:.0f}% Rückgang in 5T → ignoriert\n")


if __name__ == "__main__":
    main()
