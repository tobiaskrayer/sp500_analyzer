"""
╔══════════════════════════════════════════════════════════════════╗
║   S&P 500 · ADVANCED BREAKOUT SCREENER  v3.1                    ║
║   + S&P 500 MARKTGESUNDHEITS-ANALYSE (NEU)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  NEU in v3.1: S&P500 INDEX-ANALYSE                              ║
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
# MODUL: S&P 500 MARKTGESUNDHEITS-ANALYSE  (NEU in v3.1)
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
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp = None
        else:
            resp = None
        if resp and resp.status_code == 200:
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

        # ── SIGNAL-SAMMLUNG (mit Gewichtung) ──────────────────────────────
        bull = []   # (gewicht, beschreibung)
        bear = []

        if result["above_ma50"]:
            bull.append((2, f"Kurs über MA50 (+{result['pct_from_ma50']:.1f}%)"))
        else:
            bear.append((2, f"Kurs unter MA50 ({result['pct_from_ma50']:.1f}%)"))

        if result["above_ma200"] is not None:
            if result["above_ma200"]:
                bull.append((3, f"Kurs über MA200 (+{result['pct_from_ma200']:.1f}%)"))
            else:
                bear.append((3, f"Kurs unter MA200 ({result['pct_from_ma200']:.1f}%)"))

        if result["golden_cross"] is not None:
            if result["golden_cross"]:
                bull.append((3, "Golden Cross aktiv (MA50 > MA200)"))
            else:
                bear.append((3, "Death Cross aktiv (MA50 < MA200)"))

        if rsi_val > 50 and result["rsi_trend"] == "steigend":
            bull.append((2, f"RSI {rsi_val:.0f} steigend (bullisches Momentum)"))
        elif rsi_val < 45:
            bear.append((1, f"RSI {rsi_val:.0f} – schwaches Momentum"))
        elif rsi_val > 70:
            bear.append((2, f"RSI {rsi_val:.0f} – überkauft, Korrekturrisiko"))

        if result["macd_bullish"]:
            bull.append((2, "MACD-Histogramm positiv & steigend"))
        else:
            bear.append((1, "MACD-Histogramm negativ oder fallend"))

        if bb_pos > 0.7:
            bear.append((1, f"Kurs im oberen BB-Bereich ({bb_pos*100:.0f}%) – überdehnt"))
        elif bb_pos < 0.3:
            bull.append((1, f"Kurs im unteren BB-Bereich ({bb_pos*100:.0f}%) – mögliche Erholung"))

        if result["vix"] is not None:
            if result["vix"] < 18:
                bull.append((2, f"VIX {result['vix']:.1f} – geringe Volatilität (ruhiger Markt)"))
            elif result["vix"] > 25:
                bear.append((3, f"VIX {result['vix']:.1f} – erhöhte Angst im Markt"))

        if result["change_5d_pct"] and result["change_5d_pct"] > 1.5:
            bull.append((1, f"5-Tages-Momentum +{result['change_5d_pct']:.1f}%"))
        elif result["change_5d_pct"] and result["change_5d_pct"] < -1.5:
            bear.append((1, f"5-Tages-Momentum {result['change_5d_pct']:.1f}%"))

        # Für Anzeige: nur die Beschreibung (ohne Gewicht)
        result["signals_bull"] = [desc for _, desc in bull]
        result["signals_bear"] = [desc for _, desc in bear]

        # ── GESAMT-SCORE: kontinuierliche Berechnung (0–100) ─────────────
        # Jeder Indikator liefert einen Teilwert 0–100, dann gewichteter Mittelwert.
        # Deutlich feingranularer als die frühere binäre bull/bear-Ratio-Formel.

        # 1) Trendstruktur (35 %) — MA50, MA200, Golden/Death Cross
        pct50_v  = result["pct_from_ma50"]  or 0.0
        pct200_v = result["pct_from_ma200"] or 0.0
        if result["above_ma50"] and result["above_ma200"] and result.get("golden_cross"):
            # Voll bullisch: Score wächst mit dem Abstand über die MAs
            t_pts = min(100.0, 70.0 + min(pct50_v, pct200_v) * 1.5)
        elif result["above_ma50"] and result["above_ma200"]:
            t_pts = min(68.0, 52.0 + pct50_v * 2.0)
        elif result.get("golden_cross") is False:
            # Death Cross aktiv
            t_pts = max(0.0, 28.0 + pct50_v * 1.5)
        elif result.get("above_ma200"):
            t_pts = 35.0
        elif result["above_ma50"]:
            t_pts = 32.0
        else:
            t_pts = max(0.0, 20.0 + pct50_v * 1.5)

        # 2) RSI-Score (18 %) — ideale Zone 52–68, Richtung zählt mit
        _rsi_map = [
            (20, 5.0), (30, 15.0), (40, 30.0), (48, 43.0), (52, 55.0),
            (60, 72.0), (68, 82.0), (75, 68.0), (83, 45.0), (100, 20.0)
        ]
        r_pts = 50.0
        for _thresh, _pts in _rsi_map:
            if rsi_val <= _thresh:
                r_pts = _pts
                break
        if result["rsi_trend"] == "steigend":
            r_pts = min(100.0, r_pts + 8.0)

        # 3) MACD-Score (17 %) — Histogramm normiert auf Verlauf der letzten 60 Tage
        _ref = float(macd_h.iloc[-60:].abs().quantile(0.80)) if len(macd_h) >= 20 else 5.0
        _ref = _ref if _ref > 0 else 1.0
        _hn  = max(-1.0, min(1.0, hist_val / (_ref * 1.5)))
        m_pts = 50.0 + _hn * 50.0
        if result["macd_bullish"]:
            m_pts = max(m_pts, 58.0)

        # 4) VIX-Score (16 %) — vollständig kontinuierlich (niedriger VIX = besser)
        _vix  = result["vix"] or 20.0
        v_pts = max(0.0, min(100.0, (45.0 - _vix) / 35.0 * 100.0))

        # 5) Preis-Momentum (9 %) — Kombination aus 5T- und 20T-Rendite
        _chg5  = result["change_5d_pct"]  or 0.0
        _chg20 = result["change_20d_pct"] or 0.0
        _mom   = _chg5 * 0.6 + _chg20 * 0.4
        p_pts  = max(0.0, min(100.0, (_mom + 8.0) / 16.0 * 100.0))

        # 6) Bollinger-Position (5 %) — Mitte-oben ideal, Extreme schlechter
        _bp = result["bb_position"] or 0.5
        if _bp <= 0.5:
            b_pts = 50.0 + _bp * 40.0           # 50–70 (Kurs steigt in die Bandmitte)
        else:
            b_pts = max(35.0, 90.0 - _bp * 60.0) # 90 bei 0.5, 30 bei 1.0 (überdehnt)

        # Gewichteter Mittelwert → 0–100
        score = round(max(0.0, min(100.0,
            (35 * t_pts + 18 * r_pts + 17 * m_pts +
             16 * v_pts +  9 * p_pts +  5 * b_pts) / 100.0
        )), 1)
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

    def _pad(text: str) -> str:
        """Füllt Text auf W Zeichen auf, robust gegen jede Textlänge."""
        # Emoji/Unicode kann breiter sein, aber für Konsole reicht len()
        pad = max(0, W - len(text))
        return text + " " * pad

    print(f"\n╔{sep}╗")
    print(f"║{'S&P 500 MARKTGESUNDHEITS-ANALYSE · v3.1':^{W}}║")
    print(f"╠{sep}╣")

    # Preisinfo
    if spx["price"]:
        chg1 = spx["change_1d_pct"] or 0
        chg5 = spx["change_5d_pct"] or 0
        chg20= spx["change_20d_pct"]or 0
        arrow1 = "▲" if chg1 >= 0 else "▼"
        arrow5 = "▲" if chg5 >= 0 else "▼"
        arrow20= "▲" if chg20>= 0 else "▼"
        line = (f"  S&P 500 (^GSPC): ${spx['price']:,.2f}  "
                f"1T: {arrow1}{abs(chg1):.2f}%  "
                f"5T: {arrow5}{abs(chg5):.2f}%  "
                f"20T: {arrow20}{abs(chg20):.2f}%")
        print(f"║{_pad(line)}║")
    else:
        print(f"║{_pad('  S&P 500 Daten nicht verfügbar')}║")

    # MA-Info
    if spx["ma50"]:
        ma50_sym  = "✓" if spx["above_ma50"]  else "✗"
        ma200_sym = "✓" if spx.get("above_ma200") else "✗" if spx.get("above_ma200") is False else "–"
        gc = "Golden Cross ✓" if spx.get("golden_cross") else ("Death Cross ✗" if spx.get("golden_cross") is False else "–")
        line = (f"  MA50: ${spx['ma50']:,.0f} [{ma50_sym}]  "
                f"MA200: ${spx['ma200']:,.0f} [{ma200_sym}]  {gc}")
        print(f"║{_pad(line)}║")

    # RSI / MACD / VIX
    rsi_info  = f"RSI: {spx['rsi']:.0f} ({spx['rsi_trend']})" if spx["rsi"] else "RSI: N/A"
    macd_info = f"MACD-Hist: {'↑ Bullish' if spx['macd_bullish'] else '↓ Bearish'}" if spx["macd_hist"] is not None else ""
    vix_info  = f"VIX: {spx['vix']:.1f}" if spx["vix"] else ""
    line = f"  {rsi_info}   {macd_info}   {vix_info}"
    print(f"║{_pad(line)}║")

    print(f"╠{sep}╣")

    # Fear & Greed Index
    fgi_val   = fgi["value"]
    fgi_label = fgi["label"]
    fgi_bar   = _make_fgi_bar(fgi_val, width=30) if fgi_val is not None else "[N/A]"
    line = f"  FEAR & GREED INDEX: {fgi_bar} {fgi_val or 'N/A'}  ← {fgi_label}"
    print(f"║{_pad(line)}║")
    desc_line = f"  {fgi['description'][:W-4]}"
    print(f"║{_pad(desc_line)}║")

    print(f"╠{sep}╣")
    print(f"║{'TECHNISCHE SIGNALE':^{W}}║")

    # Bullische Signale
    for s in spx["signals_bull"]:
        line = f"  ✅ {s}"
        print(f"║{_pad(line)}║")
    # Bärische Signale
    for s in spx["signals_bear"]:
        line = f"  ⚠  {s}"
        print(f"║{_pad(line)}║")

    print(f"╠{sep}╣")

    # Trend-Label
    tl = spx.get("trend_label", "")
    print(f"║{_pad(f'  20T-Trend: {tl}')}║")

    # Verdict
    v_emoji = spx.get("verdict_emoji", "")
    verdict = spx.get("verdict", "N/A")
    score   = spx.get("score", 50)
    score_bar = _make_score_bar(score, width=20)
    v_line = f"  {v_emoji} MARKT-URTEIL: {verdict}  [{score_bar}] {score:.0f}/100"
    print(f"║{_pad(v_line)}║")

    # Kaufempfehlung
    if verdict == "KAUFEN":
        tip = "  → Technisch intakter Markt: Breakout-Kandidaten sind chancenreich"
    elif verdict == "ABWARTEN":
        tip = "  → Unklare Lage: Nur Top-Setups mit starker Bestätigung kaufen"
    else:
        tip = "  → Vorsicht empfohlen: Positionen klein halten, Stops setzen!"
    print(f"║{_pad(tip)}║")

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
        return f'<div class="mh-section"><p style="color:var(--text-dim)">Marktanalyse nicht verfügbar: {spx["error"]}</p></div>'

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
        c = "var(--green)" if v >= 0 else "var(--red)"
        arrow = "▲" if v >= 0 else "▼"
        return f'<span style="color:{c}">{arrow}{abs(v):.2f}%</span>'

    return f"""
<style>
.mh-section {{
  background: var(--glass-bg);
  backdrop-filter: blur(var(--glass-blur));
  -webkit-backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 28px 32px;
  margin-bottom: 32px;
  font-family: 'Courier New', monospace;
  box-shadow: 0 4px 32px rgba(0,0,0,0.15);
}}
.mh-title {{
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 4px;
}}
.mh-header {{
  font-size: 22px;
  font-weight: 700;
  color: var(--text);
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
  background: var(--glass-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  border-radius: 8px;
  padding: 16px;
  transition: transform .2s;
}}
.mh-card:hover {{ transform: translateY(-1px); }}
.mh-card-label {{
  font-size: 9px;
  color: var(--text-dim);
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
  color: var(--text);
}}
.mh-card-sub {{
  font-size: 11px;
  color: #aaa;
  margin-top: 6px;
}}
.mh-fgi-wrap {{
  background: var(--glass-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
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
  color: var(--text-dim);
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
  border: 1px solid var(--glass-border);
  color: var(--text-muted);
  background: var(--glass-bg);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}}
.mh-chart-label {{
  font-size: 9px;
  color: #777;
  letter-spacing: 1px;
  margin-top: 4px;
  text-align: right;
}}
/* Marktgesundheit: Mobile */
@media (max-width: 640px) {{
  .mh-section  {{ padding: 16px 14px }}
  .mh-header   {{ font-size: 16px; margin-bottom: 16px }}
  .mh-grid     {{ grid-template-columns: 1fr !important }}
  .mh-signals  {{ grid-template-columns: 1fr !important }}
  .mh-top-row  {{ grid-template-columns: 1fr !important }}
  .mh-top-mid  {{ padding: 0; display: flex; flex-wrap: wrap; gap: 6px }}
  .mh-top-gauge{{ border-left: none; padding-left: 0; border-top: 1px solid var(--glass-border); padding-top: 12px }}
  .mh-verdict-wrap {{ flex-direction: column; gap: 12px }}
  .mh-verdict-score-bar-bg {{ width: 100% }}
  .mh-card-val {{ font-size: 20px }}
  .mh-ma-pill  {{ font-size: 10px; padding: 4px 9px }}
}}
</style>

<div class="mh-section">
  <div class="mh-title">// S&P 500 Index-Analyse</div>
  <div class="mh-header">Marktgesundheits-Check <span style="font-size:14px;color:var(--text-muted);font-weight:400">— Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')}</span></div>

  <!-- Zeile 1: Kurs + MAs + Urteil -->
  <div class="mh-top-row" style="display:grid;grid-template-columns:auto 1fr auto;gap:20px;align-items:center;margin-bottom:20px;background:var(--glass-bg);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--glass-border);border-radius:10px;padding:16px 20px">
    <!-- Kurs -->
    <div>
      <div style="font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">S&amp;P 500</div>
      <div style="font-size:28px;font-weight:700;color:var(--text)">${spx.get('price', 'N/A'):,.2f}</div>
      <div style="font-size:11px;color:var(--text-muted);margin-top:2px">1T: {color_pct(chg1)} &nbsp;5T: {color_pct(chg5)} &nbsp;20T: {color_pct(chg20)}</div>
    </div>
    <!-- MA-Pills + FGI -->
    <div class="mh-top-mid" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:0 16px">
      <div class="mh-ma-pill">MA50: ${spx.get('ma50','N/A'):,.0f} [{ma50_ok}] ({spx.get('pct_from_ma50',0):+.1f}%)</div>
      <div class="mh-ma-pill">MA200: ${spx.get('ma200','N/A'):,.0f} [{ma200_ok}] ({spx.get('pct_from_ma200',0):+.1f}%)</div>
      <div class="mh-ma-pill">{gc_label}</div>
      <div class="mh-ma-pill">{spx.get('trend_label','—')}</div>
      <div class="mh-ma-pill" style="color:{fgi_color};border-color:{fgi_color}44">F&amp;G: {fgi_val} — {fgi_label}</div>
    </div>
    <!-- Urteil Mini-Gauge -->
    <div class="mh-top-gauge" style="text-align:center;border-left:1px solid var(--glass-border);padding-left:20px">
      <svg width="80" height="52" viewBox="0 0 80 52">
        <path d="M 10 45 A 30 30 0 0 1 70 45" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6" stroke-linecap="round"/>
        <path d="M 10 45 A 30 30 0 0 1 70 45" fill="none" stroke="{verdict_color}" stroke-width="6" stroke-linecap="round"
              stroke-dasharray="{score / 100 * 94.2:.1f} 94.2"
              style="filter:drop-shadow(0 0 4px {verdict_color}80)"/>
        <text x="40" y="40" text-anchor="middle" font-family="Syne,sans-serif" font-size="16" font-weight="800" fill="{verdict_color}">{score:.0f}</text>
      </svg>
      <div style="font-size:9px;color:{verdict_color};letter-spacing:1px;font-weight:700;margin-top:-2px">{verdict}</div>
    </div>
  </div>

  <!-- Zeile 2: RSI + MACD + VIX in einer Zeile -->
  <div class="mh-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">

    <div class="mh-card">
      <div class="mh-card-label">RSI (14) — 60 Tage</div>
      <div class="mh-card-top">
        <span class="mh-card-val" style="color:{rsi_color}">{rsi_cur}</span>
        <span style="font-size:11px;color:var(--text-muted)">RSI {spx.get('rsi_trend','')}</span>
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
        <span class="mh-card-val" style="color:{'var(--green)' if spx.get('macd_bullish') else 'var(--red)'}">{macd_cur}</span>
        <span style="font-size:11px;color:var(--text-muted)">{macd_dir}</span>
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
        <span style="font-size:11px;color:var(--text-muted)">{spx.get('vix_label', '')}</span>
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
      <div style="font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase">Fear &amp; Greed Index (CNN)</div>
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

  <!-- Verdict with Radial Gauge (Upgrade 3) -->
  <div class="mh-verdict-wrap" style="display:flex;align-items:center;gap:32px;background:var(--glass-bg);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:2px solid {verdict_color};border-radius:12px;padding:24px 28px">
    <div style="flex-shrink:0">
      <svg width="130" height="80" viewBox="0 0 130 80">
        <!-- Gauge background arc -->
        <path d="M 15 70 A 50 50 0 0 1 115 70" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="10" stroke-linecap="round"/>
        <!-- Gauge colored arc -->
        <path d="M 15 70 A 50 50 0 0 1 115 70" fill="none" stroke="{verdict_color}" stroke-width="10" stroke-linecap="round"
              stroke-dasharray="{score / 100 * 157:.1f} 157"
              style="filter:drop-shadow(0 0 6px {verdict_color}80);transition:stroke-dasharray 1.5s ease"/>
        <!-- Score text -->
        <text x="65" y="62" text-anchor="middle" font-family="Syne,sans-serif" font-size="26" font-weight="800" fill="{verdict_color}">{score:.0f}</text>
        <text x="65" y="76" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8" fill="currentColor" opacity="0.5">/ 100</text>
      </svg>
    </div>
    <div>
      <div style="font-size:10px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">Markt-Urteil</div>
      <div style="font-size:28px;font-weight:700;color:{verdict_color};letter-spacing:2px">{verdict_emoji} {verdict}</div>
      <div style="font-size:11px;color:var(--text-dim);margin-top:8px;max-width:420px;line-height:1.6">
        {"Technisch intakter Markt: Die Bedingungen für Breakout-Investitionen sind günstig. Starke Setups aus dem Screener haben erhöhte Erfolgswahrscheinlichkeit." if verdict == "KAUFEN" else
         "Unklare Marktlage: Nur hochwertige Setups mit mehrfacher Bestätigung in Betracht ziehen. Engere Stops setzen und Positionsgrößen reduzieren." if verdict == "ABWARTEN" else
         "Markt zeigt Schwächesignale oder befindet sich in einer Korrektur. Vorsicht bei Neukäufen! Bestehende Positionen schützen und auf klare Trendumkehr warten."}
      </div>
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
# NACHRICHTEN-ANALYSE
# ──────────────────────────────────────────────
NEWS_TOP_N = 30    # Für wie viele Top-Aktien Nachrichten laden

# ──────────────────────────────────────────────
# PERFORMANCE: PARALLELE DOWNLOADS + CACHE
# ──────────────────────────────────────────────
PARALLEL_WORKERS = 4          # Parallele Download-Threads
CACHE_DIR        = Path(".screener_cache")
CACHE_MAX_AGE_H  = 6          # Cache nach X Stunden ungültig
SCORE_CACHE_FILE = Path(".score_history.json")  # Score-Verlauf für Glättung

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
    "ma60_crossover_today":    45,   # Crossover genau HEUTE → Primärsignal (reduziert für weniger Klippe)
    "ma60_crossover_recent":   35,   # Crossover in letzten 1-5 Tagen (erhöht für Stabilität)
    "price_above_ma60_rising": 15,   # Über MA60 + MA60 selbst steigt (nachhaltiger Trend)

    # ── BREAKOUT-GESUNDHEIT ──────────────────────────────────────
    # Entscheidet ob ein Recent Cross wirklich gut ist
    "breakout_holding":        40,   # Nach Cross: Preis heute >= Cross-Hoch (hält die Bewegung)
    "breakout_continuing":     20,   # Nach Cross: Preis steigt jeden Tag weiter (Continuation)

    # ── PERSISTENTE MOMENTUM-SIGNALE (NEU) ───────────────────────
    # Diese feuern über mehrere Tage → stabilisiert den Score
    "consecutive_rise_3d":     20,   # Preis 3 aufeinanderfolgende Tage gestiegen
    "macd_positive_trend":     15,   # MACD-Histogramm 3 Tage in Folge positiv (nachhaltiger Aufwärtstrend)

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
        "BA","BCR","BMY","AVGO","BR","BRO","BF-B","BLDR","BXP","CHRW","CDNS",
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
    # Hash über die tatsächlichen Ticker – verhindert Kollisionen bei
    # gleicher Anzahl aber unterschiedlicher Zusammensetzung
    import hashlib
    ticker_hash = hashlib.md5("".join(sorted(tickers)).encode()).hexdigest()[:12]
    cache_key   = f"daily_v3_{date.today().isoformat()}_{len(tickers)}_{ticker_hash}"
    data        = _cache_load(cache_key)

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

def get_market_status() -> dict:
    """
    Prüft ob der US-Aktienmarkt (NYSE/NASDAQ) gerade geöffnet ist.
    Gibt Marktzeit (ET), Fortschritt des Handelstages und
    Volumen-Hochrechnungsfaktor zurück.
    Benötigt keine externe Bibliothek (kein pytz).
    """
    from datetime import timezone, timedelta
    utc_now = datetime.now(timezone.utc)

    # US-Sommerzeit (EDT): 2. Sonntag März → 1. Sonntag November
    year  = utc_now.year
    mar1  = datetime(year, 3,  1, tzinfo=timezone.utc)
    nov1  = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday() + 7) % 7 + 7)
    dst_end   = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    offset    = timedelta(hours=-4 if dst_start <= utc_now < dst_end else -5)
    et_now    = utc_now + offset

    weekday   = et_now.weekday()          # 0=Mo … 6=So
    et_min    = et_now.hour * 60 + et_now.minute
    OPEN, CLOSE = 9 * 60 + 30, 16 * 60   # 9:30–16:00 ET

    is_open   = (weekday < 5) and (OPEN <= et_min < CLOSE)
    pct       = min(1.0, max(0.0, (et_min - OPEN) / (CLOSE - OPEN))) if is_open else 0.0
    # Volumen-Skalierung: hochrechnen auf vollen Handelstag
    # Erst ab 5 % des Tages (≈ ~20 Min nach Öffnung) sinnvoll
    vol_scale = (1.0 / pct) if (is_open and pct > 0.05) else 1.0
    vol_scale = min(vol_scale, 8.0)       # max. 8× (Frühhandel-Schutz)

    time_str  = f"{et_now.hour:02d}:{et_now.minute:02d} ET"
    if is_open:
        pct_int = int(pct * 100)
        label   = f"LIVE · {time_str} · {pct_int}% des Handelstages"
    else:
        label   = f"MARKT GESCHLOSSEN · Schlusskurse · {time_str}"

    return {
        "is_open":   is_open,
        "time_str":  time_str,
        "pct":       pct,
        "vol_scale": vol_scale,
        "label":     label,
    }


# Markt-Status wird einmalig in analyze_all() gesetzt und von analyze() gelesen
MARKET_STATUS: dict = {"is_open": False, "vol_scale": 1.0, "label": "", "pct": 0.0}


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

        # ── VORFILTER (bereits in analyze_all durchgeführt, hier nur Sicherheits-Check) ──
        # Kein erneuter pre_filter()-Aufruf nötig, da analyze_all() nur gefilterte Ticker übergibt.

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

        # ── ADX + ATR ─────────────────────────────────────────────
        try:
            adx_s, plus_di, minus_di = calc_adx(high, low, close)
            adx_val    = float(adx_s.iloc[-1])
            plus_di_v  = float(plus_di.iloc[-1])
            minus_di_v = float(minus_di.iloc[-1])
        except Exception:
            adx_val    = 0.0
            plus_di_v  = 0.0
            minus_di_v = 0.0

        # ATR(14) – Average True Range, Basis für Swing-Ziel und Stop
        try:
            _tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr14     = float(_tr.rolling(14).mean().iloc[-1])
            atr_pct   = round(atr14 / today_c * 100, 2) if today_c > 0 else 0.0
        except Exception:
            atr14, atr_pct = 0.0, 0.0

        # ── VOLUMEN-ANALYSE ──────────────────────────────────────
        vol_ratio     = 0.0
        vol_spike_up  = False   # Spike NUR an Up-Day
        vol_dry_before = False   # Volumen vorher trocken, jetzt hoch

        if vol is not None and len(vol) >= 20 and vol.sum() > 0:
            avg_vol    = float(vol.iloc[-21:-1].mean())
            today_v_raw = float(vol.iloc[-1])
            # Während offenem Markt: Intraday-Volumen auf Tagesende hochrechnen
            vol_scale  = MARKET_STATUS.get("vol_scale", 1.0)
            today_v    = today_v_raw * vol_scale
            vol_ratio  = today_v / avg_vol if avg_vol > 0 else 0.0
            is_up_day  = today_c > yest_c  # Preis ist heute gestiegen
            vol_spike_up = vol_ratio > 1.5 and is_up_day
            # Trocken = durchschnittliches Volumen der letzten 5 Tage war unter Durchschnitt
            avg_5d_prev = float(vol.iloc[-6:-1].mean())
            vol_dry_before = (avg_5d_prev < avg_vol * 0.8) and (today_v > avg_vol * 1.5)

        # ── INTRADAY-ÄNDERUNG (nur bei offenem Markt aussagekräftig) ──
        intraday_chg_pct = None
        if MARKET_STATUS.get("is_open") and MARKET_STATUS.get("pct", 0) > 0.05:
            intraday_chg_pct = round((today_c - yest_c) / yest_c * 100, 2) if yest_c else None

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

        # Crossover in letzten 1–5 Tagen (ohne heute) — erweitert für Robustheit
        recent_cross = any(
            float(close.iloc[-j-1]) <= float(ma60.iloc[-j-1]) and
            float(close.iloc[-j])   > float(ma60.iloc[-j])
            for j in range(2, min(6, n-1))
        )

        # ── BREAKOUT-GESUNDHEIT ───────────────────────────────────
        # Für recent_cross: Wie hat sich die Aktie NACH dem Cross entwickelt?
        breakout_holding    = False   # Preis heute >= Preis am Cross-Tag
        breakout_continuing = False   # Preis steigt jeden Tag seit Cross

        if recent_cross and not cross_up:
            # Finde den genauen Cross-Tag
            cross_day_idx = None
            for j in range(2, min(6, n-1)):
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

        # ── SIGNAL-ALTER (Tage seit letztem MA60-Crossover) ──────
        # 0 = heute, 1 = gestern, ... 99 = kein frischer Cross
        signal_age = 99
        if cross_up:
            signal_age = 0
        elif recent_cross:
            for j in range(2, min(6, n-1)):
                c_p = float(close.iloc[-j-1]); c_c = float(close.iloc[-j])
                m_p = float(ma60.iloc[-j-1]);  m_c = float(ma60.iloc[-j])
                if c_p <= m_p and c_c > m_c:
                    signal_age = j - 1   # j=2 → 1T ago, j=3 → 2T ago …
                    break

        # Relative Stärke: Heute-Performance vs. 5-Tage-Durchschnitt der Tagesrenditen
        returns_5d    = close.pct_change().iloc[-6:-1]
        today_ret     = (today_c - yest_c) / yest_c
        outperforming = today_ret > float(returns_5d.mean()) * 1.2

        # Über EMA200
        above_ema200  = today_c > today_ema and not np.isnan(today_ema)

        # ── PERSISTENTE MULTI-TAGES-SIGNALE ──────────────────────
        # Preis steigt 3 aufeinanderfolgende Tage (feuert mehrere Tage)
        consecutive_rise_3d = (
            n >= 4 and
            float(close.iloc[-1]) > float(close.iloc[-2]) and
            float(close.iloc[-2]) > float(close.iloc[-3]) and
            float(close.iloc[-3]) > float(close.iloc[-4])
        )

        # MACD-Histogramm 3 Tage hintereinander positiv (nachhaltiger Aufwärtstrend)
        macd_positive_trend = (today_hist > 0) and (yest_hist > 0) and (prev_hist > 0)

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

        # Persistente Multi-Tages-Signale
        add("consecutive_rise_3d",     consecutive_rise_3d)
        add("macd_positive_trend",     macd_positive_trend)

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
        # Dynamisch: Summe aller Gewichte minus sich ausschließende Signale
        # (kann nicht gleichzeitig recent_cross UND cross_up haben,
        #  ebenso nicht breakout_holding ohne recent_cross)
        mutually_exclusive = max(
            WEIGHTS["ma60_crossover_today"],
            WEIGHTS["ma60_crossover_recent"]
                + WEIGHTS["breakout_holding"]
                + WEIGHTS["breakout_continuing"],
            WEIGHTS["price_above_ma60_rising"],
        )
        # Squeeze: entweder with_breakout oder building
        squeeze_max = max(WEIGHTS["squeeze_with_breakout"],
                          WEIGHTS["squeeze_building"])
        # Alle nicht-trend, nicht-squeeze Gewichte
        other_keys = [k for k in WEIGHTS
                      if k not in ("ma60_crossover_today", "ma60_crossover_recent",
                                   "price_above_ma60_rising", "breakout_holding",
                                   "breakout_continuing", "squeeze_with_breakout",
                                   "squeeze_building")]
        other_sum = sum(WEIGHTS[k] for k in other_keys)

        # Top-3 Combo-Boni
        top_combos = sorted([b for _, b, _ in COMBO_BONUSES], reverse=True)[:3]

        realistic_max = mutually_exclusive + squeeze_max + other_sum + sum(top_combos)
        score_100  = min(score / realistic_max * 100, 100.0)

        # ── KONTINUIERLICHE FEINKORREKTUR (±5 Punkte) ────────────────────
        # Alle binären Gewichte sind Vielfache von 5 → viele Aktien landen
        # exakt auf demselben Rohwert. Die Feinkorrektur nutzt die tatsächlichen
        # Float-Werte der Indikatoren, um Gleichstände aufzulösen.
        fine = 0.0

        # RSI-Qualität: Idealzone 55-65, kontinuierlich bewertet (±1.5 Pkt)
        fine += max(0.0, 1.5 - abs(today_rsi - 60.0) / 16.0)

        # MACD-Histogramm-Stärke relativ zu den letzten 20 Tagen (±2 Pkt)
        _mref = max(float(macd_hist.iloc[-21:-1].abs().mean()), 0.0001)
        fine += max(-2.0, min(2.0, today_hist / _mref * 2.0))

        # ADX-Stärke: stärkerer Trend = besser (0..1 Pkt)
        fine += min(1.0, max(0.0, (adx_val - 20.0) / 25.0))

        # Schlusskurs-Qualität innerhalb der Tagesrange (-0.5..+0.5 Pkt)
        fine += (close_vs_range - 0.5)

        # Volumen-Stärke über 1.5×-Schwelle hinaus (0..0.5 Pkt)
        fine += min(0.5, max(0.0, (vol_ratio - 1.5) * 0.2))

        # Signal-Frische: frische Crossover bekommen Bonus, alte einen Abzug
        # Wichtig für Swing-Trading: je frischer, desto mehr Upside noch möglich
        _freshness = {0: +3.0, 1: +1.5, 2: +0.5, 3: -0.5, 4: -1.5}
        fine += _freshness.get(signal_age, -2.5)   # 99 (kein Cross) → -2.5

        score_100 = round(min(max(score_100 + fine, 0.0), 100.0), 1)

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
            "atr14":             round(atr14, 2),
            "atr_pct":           atr_pct,
            "signal_age":        signal_age,
            "swing_target":      round(today_c + 2.0 * atr14, 2),
            "swing_stop":        round(today_c - 1.5 * atr14, 2),
            "intraday_chg_pct":  intraday_chg_pct,
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
    global MARKET_STATUS
    MARKET_STATUS = get_market_status()
    ms = MARKET_STATUS
    icon = "🟢" if ms["is_open"] else "⚫"
    print(f"\n{icon} Markt: {ms['label']}")
    if ms["is_open"]:
        print(f"   ℹ  Volumen wird mit Faktor {ms['vol_scale']:.1f}× auf Tagesende hochgerechnet")

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
    df = pd.DataFrame(results).reset_index(drop=True)

    # ── SCORE-GLÄTTUNG: Mische heutigen Score mit Vortag (3-Tage-EMA) ──
    # Verhindert starke Tagesschwankungen durch Einzel-Tages-Signale
    today_str = date.today().isoformat()
    prev_scores: dict[str, list[float]] = {}
    if SCORE_CACHE_FILE.exists():
        try:
            cached = json.loads(SCORE_CACHE_FILE.read_text(encoding="utf-8"))
            # Format: {ticker: [score_gestern, score_vorgestern], date: "..."}
            cache_date = cached.get("date", "")
            if cache_date != today_str:  # nur verwenden wenn nicht selber Tag
                prev_scores = {k: v for k, v in cached.items() if k != "date"}
        except Exception:
            pass

    smoothed_list = []
    for _, row in df.iterrows():
        t = row["ticker"]
        s_today = float(row["score"])
        history = prev_scores.get(t, [])
        if len(history) >= 2:
            # 3-Tage-EMA: 50% heute, 35% gestern, 15% vorgestern
            s_smooth = round(0.50 * s_today + 0.35 * history[0] + 0.15 * history[1], 1)
        elif len(history) == 1:
            s_smooth = round(0.65 * s_today + 0.35 * history[0], 1)
        else:
            s_smooth = s_today
        smoothed_list.append(s_smooth)

    df["score_smoothed"] = smoothed_list

    # Heutigen Score für nächsten Tag speichern
    try:
        new_cache: dict = {"date": today_str}
        for _, row in df.iterrows():
            t = row["ticker"]
            s = float(row["score"])
            old = prev_scores.get(t, [])
            new_cache[t] = [s] + old[:1]  # max. 2 Einträge aufbewahren
        SCORE_CACHE_FILE.write_text(json.dumps(new_cache), encoding="utf-8")
    except Exception:
        pass

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"   ✓ {len(df)} Aktien im Ergebnis (sortiert nach aktuellem Score)")
    return df


# ══════════════════════════════════════════════
# SCHRITT 4: HTML-REPORT
# ══════════════════════════════════════════════

def build_html(df: pd.DataFrame, spx: dict = None, fgi: dict = None,
               total_tickers: int = 0, news_html: str = "",
               news_results: dict = None) -> str:
    top     = df.head(150)
    # Marktgesundheits-Block (leer wenn nicht vorhanden)
    market_health_html = ""
    if spx is not None and fgi is not None:
        market_health_html = build_market_health_html(spx, fgi)

    def badge(signal):
        b = {
            "CROSSOVER_UP":   "background:rgba(0,200,83,0.12);color:var(--green);border:1px solid rgba(0,200,83,0.4)",
            "RECENT_CROSS":   "background:rgba(0,200,83,0.06);color:var(--green);border:1px solid rgba(0,200,83,0.2)",
            "ABOVE_MA60":     "background:rgba(68,136,255,0.1);color:var(--blue);border:1px solid rgba(68,136,255,0.3)",
            "BELOW_MA60":     "background:rgba(255,68,102,0.06);color:var(--red);border:1px solid rgba(255,68,102,0.2)",
            "CROSSOVER_DOWN": "background:rgba(255,68,102,0.12);color:var(--red);border:1px solid rgba(255,68,102,0.4)",
        }
        labels = {
            "CROSSOVER_UP":   "🔥 MA60 BREAKOUT ↑",
            "RECENT_CROSS":   "⚡ KÜRZL. BREAKOUT",
            "ABOVE_MA60":     "▲ ÜBER MA60",
            "BELOW_MA60":     "▼ UNTER MA60",
            "CROSSOVER_DOWN": "↓ MA60 BRUCH",
        }
        st = b.get(signal, "color:var(--text-muted)")
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
        c = "var(--green)" if prices[-1] >= prices[0] else "var(--red)"
        # Fill polygon for area under line
        first_x = 0.0
        last_x  = 100.0
        # Calculate total path length estimate for animation
        total_len = 0
        prev_x, prev_y = 0, 36 - ((prices[0]-mn)/rng)*32 - 2
        for i, p in enumerate(prices):
            cx = (i/(len(prices)-1))*100
            cy = 36 - ((p-mn)/rng)*32 - 2
            total_len += ((cx-prev_x)**2 + (cy-prev_y)**2)**0.5
            prev_x, prev_y = cx, cy
        total_len = int(total_len) + 10
        last_pt_x = 100
        last_pt_y = 36 - ((prices[-1]-mn)/rng)*32 - 2
        return (
            f'<svg width="100" height="36" viewBox="0 0 100 36" style="overflow:visible">'
            f'<defs><linearGradient id="sg{id(prices)}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{c}" stop-opacity="0.15"/>'
            f'<stop offset="100%" stop-color="{c}" stop-opacity="0.01"/>'
            f'</linearGradient></defs>'
            f'<polygon points="{first_x:.0f},36 {pts} {last_x:.0f},36" '
            f'fill="url(#sg{id(prices)})" opacity="0.8"/>'
            f'<polyline class="spark-line" points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.9" '
            f'style="--line-len:{total_len};stroke-dasharray:{total_len};stroke-dashoffset:{total_len}"/>'
            f'<circle class="spark-dot" cx="{last_pt_x:.1f}" cy="{last_pt_y:.1f}" r="2.5" fill="{c}" opacity="0"/>'
            f'<animate attributeName="opacity" from="0" to="1" begin="1s" dur="0.3s" fill="freeze" '
            f'xlink:href=".spark-dot"/>'
            f'</svg>'
        )

    def prob_bar(prob):
        bar_c = "var(--green)" if prob >= 60 else "var(--orange)" if prob >= 45 else "#ff9944" if prob >= 30 else "var(--red)"
        return (f'<div style="display:flex;align-items:center;gap:8px;min-width:140px">'
                f'<div style="flex:1;height:5px;background:var(--bg-card-solid);border-radius:3px;overflow:hidden">'
                f'<div style="width:{prob}%;height:100%;background:{bar_c};border-radius:3px"></div></div>'
                f'<span style="color:{bar_c};font-size:11px;font-weight:700;min-width:35px">{prob}%</span>'
                f'</div>')

    def pct(val, show_sign=True):
        if val is None: return "—"
        c = "var(--green)" if val >= 0 else "var(--red)"
        s = "+" if val >= 0 and show_sign else ""
        return f'<span style="color:{c}">{s}{val:.2f}%</span>'

    rows_html = ""
    col_count = 11  # Anzahl Spalten für colspan
    for i, row in top.iterrows():
        has_border = row["signal"] in ("CROSSOVER_UP", "RECENT_CROSS")
        border_s   = ' style="border-left:3px solid var(--green)"' if has_border else ""
        ticker     = row["ticker"]

        # News-Daten für diese Zeile aufbereiten
        nr = (news_results or {}).get(ticker, {})
        has_news = bool(nr and "error" not in nr)

        if has_news:
            rec       = nr.get("recommendation", "ABWARTEN")
            rec_color = nr.get("rec_color", "#ffd600")
            sentiment = nr.get("sentiment_label", "neutral")
            sent_color= "#00c853" if sentiment=="positiv" else "#ff1744" if sentiment=="negativ" else "#ffd600"
            reason    = nr.get("reason", "—").replace('"', '&quot;')
            n_art     = nr.get("n_articles", 0)
            headlines = nr.get("top_headlines", [])
            sec_files = nr.get("sec_filings", [])

            hl_html = ""
            for h in headlines[:4]:
                s    = h.get("score", 0)
                c    = "#00c853" if s > 0 else "#ff4466" if s < 0 else "#888"
                dot  = "▲" if s > 0 else "▼" if s < 0 else "·"
                url  = h.get("url", "")
                title_part = (
                    f'<a href="{url}" target="_blank" '
                    f'style="color:var(--text-muted);text-decoration:none;border-bottom:1px solid var(--border)" '
                    f'onmouseover="this.style.color=\'var(--accent)\';this.style.borderBottomColor=\'var(--accent)\'" '
                    f'onmouseout="this.style.color=\'var(--text-muted)\';this.style.borderBottomColor=\'var(--border)\'">'
                    f'{h["title"]}</a>'
                    if url else h["title"]
                )
                hl_html += (f'<div style="display:flex;gap:8px;margin-bottom:6px;align-items:baseline">'
                            f'<span style="color:{c};flex-shrink:0;font-size:12px">{dot}</span>'
                            f'<span style="font-size:12px;line-height:1.5">{title_part}'
                            f'<span style="color:var(--text-dim);margin-left:8px;font-size:10px">'
                            f'{h["date"]} · {h["source"]}</span></span></div>')

            # SEC EDGAR Block
            sec_html = ""
            if sec_files:
                SEC_COLORS = {
                    "8-K": "#ff9944", "8-K/A": "#ff9944",
                    "10-Q": "#4488ff", "10-K": "#4488ff",
                    "4": "#aa88ff", "SC 13G": "#ffdd44", "SC 13D": "#ffdd44",
                    "DEF 14A": "#88ccff",
                }
                sec_items = ""
                for f in sec_files[:5]:
                    form  = f.get("form", "")
                    fc    = SEC_COLORS.get(form, "#888")
                    furl  = f.get("url", "")
                    ftitle= f.get("title", form)
                    fdate = f.get("date", "")
                    link  = (f'<a href="{furl}" target="_blank" '
                             f'style="color:var(--text-muted);text-decoration:none;border-bottom:1px solid var(--border)" '
                             f'onmouseover="this.style.color=\'var(--accent)\';this.style.borderBottomColor=\'var(--accent)\'" '
                             f'onmouseout="this.style.color=\'var(--text-muted)\';this.style.borderBottomColor=\'var(--border)\'">'
                             f'{ftitle}</a>' if furl else ftitle)
                    sec_items += (f'<div style="display:flex;gap:8px;margin-bottom:5px;align-items:baseline">'
                                  f'<span style="font-size:9px;font-weight:700;color:{fc};'
                                  f'border:1px solid {fc}55;padding:1px 5px;border-radius:3px;'
                                  f'flex-shrink:0;letter-spacing:0.5px">{form}</span>'
                                  f'<span style="font-size:11px;line-height:1.5">{link}'
                                  f'<span style="color:var(--text-dim);margin-left:8px;font-size:10px">{fdate}</span>'
                                  f'</span></div>')
                sec_html = f"""
              <div style="border-top:1px solid var(--border);padding-top:10px;margin-top:4px;margin-bottom:10px">
                <div style="font-size:9px;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                  📋 SEC EDGAR · Offizielle Meldungen (letzte 30 Tage)
                </div>
                {sec_items}
              </div>"""

            news_expand = f"""
        <tr id="news-{ticker}" style="display:none;background:var(--bg-card-solid)">
          <td colspan="{col_count}" style="padding:0">
            <div style="padding:14px 14px 14px 16px;border-left:3px solid {rec_color};
                        border-bottom:1px solid var(--border);font-family:'Courier New',monospace">
              <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px">
                <div style="font-size:10px;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase">
                  📰 Nachrichten · {n_art} Artikel
                </div>
                <div style="display:flex;gap:8px;align-items:center">
                  <span style="font-size:9px;color:{sent_color};border:1px solid {sent_color}44;
                               padding:2px 8px;border-radius:3px;text-transform:uppercase;
                               letter-spacing:1px">{sentiment}</span>
                  <span style="font-size:12px;font-weight:700;color:{rec_color};
                               border:1px solid {rec_color}66;padding:4px 14px;
                               border-radius:4px;letter-spacing:1px">{rec}</span>
                </div>
              </div>
              <div style="margin-bottom:10px">{hl_html}</div>
              {sec_html}
              <div style="font-size:11px;color:var(--text-dim);border-top:1px solid var(--border);padding-top:10px">
                💡 {reason}
              </div>
            </div>
          </td>
        </tr>"""
            row_cursor = "cursor:pointer"
            row_onclick = f' onclick="toggleNews(\'{ticker}\')"'
            row_title   = ' title="Klicken für Nachrichten"'
        else:
            news_expand = ""
            row_cursor  = ""
            row_onclick = ""
            row_title   = ""

        # News data attributes for hover card
        news_data_attrs = ""
        if has_news:
            # Escape reason for HTML attribute (already has &quot; from earlier)
            top_hl = headlines[0]["title"].replace('"', '&quot;')[:80] if headlines else ""
            news_data_attrs = (
                f' data-news-rec="{rec}"'
                f' data-news-rec-color="{rec_color}"'
                f' data-news-sentiment="{sentiment}"'
                f' data-news-sent-color="{sent_color}"'
                f' data-news-reason="{reason}"'
                f' data-news-count="{n_art}"'
                f' data-news-headline="{top_hl}"'
            )

        # Heatmap class based on smoothed score (Upgrade 4)
        s_disp = row.get('score_smoothed', row['score'])
        heat_class = ""
        if s_disp >= 60:   heat_class = " heat-fire"
        elif s_disp >= 45: heat_class = " heat-strong"
        elif s_disp >= 30: heat_class = " heat-mid"

        rows_html += f"""
        <tr{border_s} class="data-row{heat_class}"
            data-signal="{row['signal']}"
            data-category="{row['category']}"
            data-score="{row['score']}"
            data-ticker="{ticker}"
            data-price="{row['price']}"
            data-rsi="{row['rsi']}"
            data-perf="{row.get('perf_30d','')}"
            data-vol="{row['vol_ratio']}"{news_data_attrs}
            style="{'border-left:3px solid var(--green);' if has_border else ''}{row_cursor}"{row_onclick}{row_title}>
          <td style="padding:10px 14px;font-weight:900;font-size:14px;font-family:'Courier New',monospace">
            {'<span class="news-arrow">▶</span>' if has_news else ''}
            <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank"
               style="color:var(--text);text-decoration:none;border-bottom:1px solid var(--border);padding-bottom:1px"
               onmouseover="this.style.color='var(--accent)';this.style.borderBottomColor='var(--accent)'"
               onmouseout="this.style.color='var(--text)';this.style.borderBottomColor='var(--border)'"
               onclick="event.stopPropagation()">{ticker}</a>
          </td>
          <td style="padding:10px 14px">
            {badge(row['signal'])}
            {(lambda age: f'<div style="margin-top:4px;font-size:9px;font-weight:700;letter-spacing:0.5px;color:{"var(--green)" if age==0 else "var(--orange)" if age<=2 else "var(--text-dim)"}">'
              + ("⚡ HEUTE" if age==0 else f"{'🕐' if age<=2 else '🕓'} {age}T ALT") + '</div>'
             )(row.get('signal_age', 99)) if row.get('signal_age', 99) < 99 else ''}
            {(lambda chg: (
                f'<div style="margin-top:3px;font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;display:inline-block;'
                + ('background:rgba(255,68,102,0.15);color:#ff4466;border:1px solid rgba(255,68,102,0.3)">'
                   f'⚠ HEUTE {chg:+.1f}%</div>' if chg < -1.5 else
                   'background:rgba(255,170,0,0.12);color:#ffaa00;border:1px solid rgba(255,170,0,0.25)">'
                   f'↘ HEUTE {chg:+.1f}%</div>' if chg < 0 else
                   'background:rgba(0,255,136,0.1);color:var(--green);border:1px solid rgba(0,255,136,0.2)">'
                   f'↗ HEUTE {chg:+.1f}%</div>')
            ) if chg is not None else '')(row.get('intraday_chg_pct'))}
          </td>
          <td style="padding:10px 14px">{prob_bar(row['score'])}</td>
          <td style="padding:10px 14px;color:var(--text);font-weight:700">${row['price']:.2f}</td>
          <td style="padding:10px 14px;color:var(--text-muted);font-size:11px">${row['ma60']:.2f}</td>
          <td style="padding:10px 14px">{pct(row['pct_from_ma60'])}</td>
          <td style="padding:10px 14px;color:{'var(--orange)' if 55<=row['rsi']<=70 else 'var(--text)'}">{row['rsi']:.0f}</td>
          <td style="padding:10px 14px;font-size:11px;line-height:1.6">
            {(lambda a, t, s: f'<span style="color:var(--text-muted)">{a:.1f}%</span>'
              + f'<br><span style="color:var(--green);font-size:10px" title="Swing-Ziel +2×ATR">▲ ${t:.2f}</span>'
              + f' <span style="color:var(--red);font-size:10px" title="Stop -1.5×ATR">▼ ${s:.2f}</span>'
             )(row.get('atr_pct', 0), row.get('swing_target', row['price']), row.get('swing_stop', row['price']))}
          </td>
          <td style="padding:10px 14px">{spark(row['sparkline'])}</td>
          <td style="padding:10px 4px">{indicator_pills(row)}</td>
          <td style="padding:10px 14px;font-weight:800;font-size:15px" class="{'score-fire' if row['score']>=60 else 'score-strong' if row['score']>=45 else 'score-mid' if row['score']>=30 else 'score-weak'}"
              title="Geglättet (3T-Ø): {row.get('score_smoothed', row['score']):.1f}">
            {row['score']:.1f}
          </td>
        </tr>{news_expand}"""

    crossovers = len(df[df["signal"].isin(["CROSSOVER_UP", "RECENT_CROSS"])])
    sehr_stark = len(df[df["category"] == "🔥 SEHR STARK"])
    stark      = len(df[df["category"] == "📈 STARK"])
    analyzed   = len(df)
    gen_time   = datetime.now().strftime("%d.%m.%Y %H:%M")
    _ms        = MARKET_STATUS
    _ms_color  = "#00ff88" if _ms.get("is_open") else "#888"
    _ms_badge  = (f'<span style="background:rgba(0,255,136,0.12);border:1px solid rgba(0,255,136,0.4);'
                  f'color:#00ff88;font-size:10px;padding:2px 10px;border-radius:4px;'
                  f'margin-left:12px;letter-spacing:1px">🟢 LIVE · {_ms.get("time_str","")}</span>'
                  if _ms.get("is_open") else
                  f'<span style="background:rgba(128,128,128,0.1);border:1px solid rgba(128,128,128,0.3);'
                  f'color:#888;font-size:10px;padding:2px 10px;border-radius:4px;'
                  f'margin-left:12px;letter-spacing:1px">⚫ SCHLUSSKURSE</span>')

    # ── Upgrade 7: Score-Verteilung als Mini-Histogramm ──────────
    dist_bins   = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,101)]
    dist_colors = ["#ff2244","#ff4466","#ff6644","#ff9944","#ffaa00","#ffdd44","#88dd44","#44dd88","#00ff88","#00ffcc"]
    dist_counts = []
    for lo, hi in dist_bins:
        cnt = len(df[(df["score"] >= lo) & (df["score"] < hi)])
        dist_counts.append(cnt)
    dist_max = max(dist_counts) if dist_counts else 1
    dist_bars_html = ""
    dist_labels_html = ""
    for idx, ((lo, hi), cnt) in enumerate(zip(dist_bins, dist_counts)):
        h_pct = max(2, int(cnt / dist_max * 100)) if cnt > 0 else 2
        c = dist_colors[idx]
        dist_bars_html += (f'<div class="dist-bar" style="height:{h_pct}%;background:{c};'
                           f'opacity:{0.4 if cnt==0 else 0.85}" '
                           f'title="{lo}–{hi-1}: {cnt} Aktien"></div>')
        dist_labels_html += f'<div class="dist-label">{lo}</div>'

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
        f'{name}</span><span style="font-size:11px;color:var(--text-muted)">{desc}</span></div>'
        for name, desc, c in legend_items
    )

    combo_legend = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="background:rgba(255,215,0,0.1);border:1px solid gold;color:gold;'
        f'padding:2px 8px;border-radius:3px;font-size:10px;min-width:140px;text-align:center;font-weight:700">'
        f'{label}</span><span style="font-size:11px;color:var(--text-muted)">+{bonus} Pkt</span></div>'
        for keys, bonus, label in COMBO_BONUSES
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout Screener v3.1 · S&P 500</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-body: #07070d;
    --bg-card: rgba(15,15,24,0.7);
    --bg-card-solid: #0f0f18;
    --bg-hover: #0d0d1a;
    --border: rgba(255,255,255,0.06);
    --border-accent: rgba(0,255,136,0.3);
    --text: #e8e8f0;
    --text-muted: #888;
    --text-dim: #555;
    --accent: #00ff88;
    --green: #00ff88;
    --blue: #4488ff;
    --orange: #ffaa00;
    --red: #ff4466;
    --glass-blur: 16px;
    --glass-bg: rgba(15,15,24,0.55);
    --glass-border: rgba(255,255,255,0.08);
  }}
  [data-theme="light"] {{
    --bg-body: #f0f0f4;
    --bg-card: rgba(255,255,255,0.7);
    --bg-card-solid: #ffffff;
    --bg-hover: #f5f5ff;
    --border: rgba(0,0,0,0.08);
    --border-accent: rgba(0,180,90,0.4);
    --text: #1a1a2e;
    --text-muted: #666;
    --text-dim: #999;
    --accent: #00aa55;
    --green: #00aa55;
    --blue: #2266cc;
    --orange: #cc8800;
    --red: #cc2244;
    --glass-blur: 20px;
    --glass-bg: rgba(255,255,255,0.6);
    --glass-border: rgba(0,0,0,0.06);
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg-body);color:var(--text);font-family:'Space Mono',monospace;min-height:100vh;transition:background .4s,color .4s}}
  body::before{{content:'';position:fixed;inset:0;
    background-image:
      radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,255,136,0.04), transparent),
      radial-gradient(ellipse 60% 50% at 80% 80%, rgba(68,136,255,0.03), transparent),
      linear-gradient(rgba(0,255,136,0.015) 1px,transparent 1px),
      linear-gradient(90deg,rgba(0,255,136,0.015) 1px,transparent 1px);
    background-size:100% 100%,100% 100%,44px 44px,44px 44px;pointer-events:none;z-index:0}}
  [data-theme="light"] body::before{{
    background-image:
      radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,180,90,0.05), transparent),
      radial-gradient(ellipse 60% 50% at 80% 80%, rgba(68,136,255,0.04), transparent);
    background-size:100% 100%,100% 100%;
  }}
  .wrap{{position:relative;z-index:1;max-width:1600px;margin:0 auto;padding:40px 24px}}
  header{{border-left:3px solid var(--accent);padding-left:20px;margin-bottom:40px;position:relative}}
  .tag{{font-size:10px;letter-spacing:4px;color:var(--accent);text-transform:uppercase;margin-bottom:8px}}
  h1{{font-family:'Syne',sans-serif;font-size:clamp(28px,4vw,50px);font-weight:800;color:var(--text);line-height:1.1}}
  h1 em{{color:var(--accent);font-style:normal}}
  .version{{display:inline-block;background:rgba(0,255,136,0.1);border:1px solid var(--border-accent);
            color:var(--accent);font-size:10px;padding:3px 10px;border-radius:4px;margin-left:12px;vertical-align:middle}}
  .meta{{margin-top:10px;font-size:11px;color:var(--text-muted);letter-spacing:1px}}

  /* Theme toggle */
  .theme-toggle{{position:absolute;top:0;right:0;background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));
    -webkit-backdrop-filter:blur(var(--glass-blur));border:1px solid var(--glass-border);
    color:var(--text-muted);font-family:'Space Mono',monospace;font-size:11px;padding:8px 14px;
    border-radius:8px;cursor:pointer;letter-spacing:1px;transition:all .3s}}
  .theme-toggle:hover{{border-color:var(--accent);color:var(--accent)}}

  /* Stats cards – glasmorphism */
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:36px}}
  .stat{{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-border);border-radius:12px;padding:16px 20px;
    box-shadow:0 4px 24px rgba(0,0,0,0.15);transition:transform .2s,box-shadow .2s}}
  .stat:hover{{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,0.25)}}
  .slabel{{font-size:9px;letter-spacing:2px;color:var(--text-muted);text-transform:uppercase;margin-bottom:6px}}
  .sval{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800}}
  .green{{color:var(--green)}}.blue{{color:var(--blue)}}.orange{{color:var(--orange)}}.white{{color:var(--text)}}

  /* Score distribution mini-histogram (Upgrade 7) */
  .score-dist{{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-border);border-radius:12px;padding:16px 20px;
    box-shadow:0 4px 24px rgba(0,0,0,0.15)}}
  .dist-bars{{display:flex;align-items:flex-end;gap:3px;height:48px;margin-top:8px}}
  .dist-bar{{flex:1;border-radius:2px 2px 0 0;min-width:0;transition:height .6s ease}}
  .dist-labels{{display:flex;gap:3px;margin-top:4px}}
  .dist-label{{flex:1;font-size:7px;color:var(--text-dim);text-align:center;letter-spacing:0.5px}}

  /* Controls */
  .controls{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
  .search{{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-border);color:var(--text);font-family:'Space Mono',monospace;
    font-size:12px;padding:10px 16px;border-radius:8px;width:220px;outline:none;letter-spacing:1px;transition:border .2s}}
  .search:focus{{border-color:var(--accent)}}
  .filters{{display:flex;gap:6px;flex-wrap:wrap}}
  .fbtn{{background:var(--glass-bg);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
    border:1px solid var(--glass-border);color:var(--text-dim);font-family:'Space Mono',monospace;
    font-size:10px;padding:8px 14px;border-radius:6px;cursor:pointer;letter-spacing:1px;transition:all .25s}}
  .fbtn.on,.fbtn:hover{{border-color:var(--accent);color:var(--accent)}}
  .fbtn.on{{background:rgba(0,255,136,0.08)}}
  [data-theme="light"] .fbtn.on{{background:rgba(0,180,90,0.08)}}

  /* Table – glasmorphism wrapper + sticky header (Upgrade 5) */
  .table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--glass-border);
    background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    box-shadow:0 4px 32px rgba(0,0,0,0.2);max-height:85vh;overflow-y:auto}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead{{position:sticky;top:0;z-index:10}}
  thead tr{{background:var(--bg-card-solid);border-bottom:2px solid var(--border)}}
  th{{padding:12px 14px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;
      color:var(--text-muted);cursor:pointer;white-space:nowrap;user-select:none;position:relative;transition:color .2s}}
  th:hover{{color:var(--accent)}}
  /* Sort indicator (Upgrade 9) */
  th .sort-arrow{{font-size:8px;margin-left:3px;opacity:0;transition:opacity .2s}}
  th.sorted .sort-arrow{{opacity:1;color:var(--accent)}}
  tbody tr{{border-bottom:1px solid var(--border);transition:background .15s}}
  tbody tr:hover{{background:var(--bg-hover)}}
  tbody tr:last-child{{border-bottom:none}}

  /* Heatmap glow for score rows (Upgrade 4) */
  .data-row{{animation:fadeUp .3s ease both;position:relative}}
  .data-row::after{{content:'';position:absolute;right:0;top:0;bottom:0;width:0;
    pointer-events:none;border-radius:0 0 0 0;opacity:0.06;transition:width .3s}}
  .data-row.heat-fire::after{{width:100%;background:linear-gradient(90deg,transparent 50%,var(--green))}}
  .data-row.heat-strong::after{{width:80%;background:linear-gradient(90deg,transparent 40%,var(--orange))}}
  .data-row.heat-mid::after{{width:60%;background:linear-gradient(90deg,transparent 50%,#ff9944)}}

  .data-row[onclick]:hover{{background:var(--bg-hover)}}
  .news-arrow{{font-size:10px;color:var(--text-dim);margin-right:4px;display:inline-block;transition:transform .2s}}

  /* Hover detail card (Upgrade 6) */
  .hover-card{{display:none;position:fixed;z-index:100;background:var(--bg-card-solid);border:1px solid var(--glass-border);
    border-radius:12px;padding:16px 20px;width:380px;max-width:calc(100vw - 24px);box-shadow:0 12px 48px rgba(0,0,0,0.5);
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);pointer-events:none;
    font-family:'Space Mono',monospace;font-size:11px}}
  .hover-card.visible{{display:block;animation:cardIn .2s ease}}
  @keyframes cardIn{{from{{opacity:0;transform:translateY(8px) scale(0.97)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}

  /* Legend – glass */
  .legend-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px}}
  .news-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .news-card{{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-border);border-radius:12px;padding:18px 20px;transition:border .2s,transform .2s}}
  .news-card:hover{{border-color:rgba(255,255,255,0.15);transform:translateY(-1px)}}
  .news-ticker{{font-size:18px;font-weight:900;color:var(--text);font-family:'Courier New',monospace}}
  /* news-grid mobile: handled in the responsive block below */
  .legend{{background:var(--glass-bg);backdrop-filter:blur(var(--glass-blur));-webkit-backdrop-filter:blur(var(--glass-blur));
    border:1px solid var(--glass-border);border-radius:12px;padding:24px}}
  .legend h3{{font-family:'Syne',sans-serif;font-size:12px;letter-spacing:2px;color:var(--text-muted);
              text-transform:uppercase;margin-bottom:16px}}
  .legend-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:4px}}
  .footer{{margin-top:24px;font-size:10px;color:var(--text-dim);letter-spacing:1px;text-align:center}}

  /* Animations */
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
  @keyframes drawLine{{from{{stroke-dashoffset:var(--line-len)}}to{{stroke-dashoffset:0}}}}
  @keyframes pulseGlow{{0%,100%{{filter:drop-shadow(0 0 2px var(--accent))}}50%{{filter:drop-shadow(0 0 8px var(--accent))}}}}
  .spark-line{{animation:drawLine 1.2s ease forwards}}
  .spark-dot{{animation:pulseGlow 2s ease infinite 1.2s}}

  /* Score column colors */
  .score-fire{{ color:var(--green) }}
  .score-strong{{ color:var(--orange) }}
  .score-mid{{ color:#dd7733 }}
  .score-weak{{ color:var(--red) }}

  /* Light mode: darken chart indicator colors for readability */
  [data-theme="light"] .mh-card .mh-chart-label span {{ opacity:0.85 }}
  [data-theme="light"] .mh-chart-label span[style*="#00ff88"] {{ color:#00994d !important }}
  [data-theme="light"] .mh-chart-label span[style*="#ffaa00"] {{ color:#aa7700 !important }}
  [data-theme="light"] .mh-card-val[style*="#00ff88"] {{ color:#00994d !important }}

  /* Light mode: heatmap row glow less intense */
  [data-theme="light"] .data-row::after {{ opacity:0.04 }}

  /* Smooth scroll for filter (Upgrade 10) */
  html{{scroll-behavior:smooth}}


  /* ── TABLET (≤ 900px): Chart + Indikatoren ausblenden ────── */
  @media (max-width:900px) {{
    #mainTable th:nth-child(9),  #mainTable td:nth-child(9),
    #mainTable th:nth-child(10), #mainTable td:nth-child(10) {{ display:none }}
    .legend-wrap {{ grid-template-columns:1fr }}
    .news-grid   {{ grid-template-columns:1fr }}
  }}

  /* ── MOBILE (≤ 640px): Kompakte Ansicht ────────────────────── */
  @media (max-width:640px) {{
    /* Layout */
    .wrap {{ padding:14px 10px }}
    header {{ padding-left:12px; margin-bottom:18px }}
    h1 {{ font-size:20px !important }}
    .meta {{ font-size:9px; letter-spacing:0 }}
    .version {{ font-size:9px; padding:2px 7px; margin-left:6px }}
    /* Theme-Toggle: nicht mehr absolut, sondern unter dem Titel */
    .theme-toggle {{ position:static; display:block; margin-top:10px; font-size:11px; padding:7px 12px }}

    /* Stats-Karten: 2 Spalten */
    .stats {{ grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px }}
    .sval  {{ font-size:20px }}
    .slabel{{ font-size:8px }}

    /* Controls: gestapelt */
    .controls {{ flex-direction:column; align-items:stretch; gap:8px }}
    .search   {{ width:100%; font-size:12px }}
    .filters  {{ gap:4px; flex-wrap:wrap }}
    .fbtn     {{ font-size:9px; padding:7px 9px }}

    /* Tabellen-Spalten ausblenden:
       3=Score-Balken  5=MA60  6=Abst.MA60  8=30T-Perf  9=Chart  10=Indikatoren */
    #mainTable th:nth-child(3),  #mainTable td:nth-child(3),
    #mainTable th:nth-child(5),  #mainTable td:nth-child(5),
    #mainTable th:nth-child(6),  #mainTable td:nth-child(6),
    #mainTable th:nth-child(8),  #mainTable td:nth-child(8),
    #mainTable th:nth-child(9),  #mainTable td:nth-child(9),
    #mainTable th:nth-child(10), #mainTable td:nth-child(10) {{ display:none }}
    table    {{ font-size:12px }}
    th, td   {{ padding:9px 8px !important }}
    .table-wrap {{ max-height:none }}

    /* Hover-Karte: volle Breite unten */
    .hover-card {{ width:calc(100vw - 20px); left:10px !important; right:10px; top:auto !important; bottom:10px; position:fixed }}

    /* Signal-Badge: kleinere Schrift */
    .filters span, thead th {{ font-size:8px }}

    /* Legend */
    .legend-wrap  {{ grid-template-columns:1fr; gap:12px; margin-top:20px }}
    .legend-grid  {{ grid-template-columns:1fr }}
    .legend       {{ padding:16px }}
    .news-grid    {{ grid-template-columns:1fr }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="tag">S&P 500 · NYSE/NASDAQ · Advanced Screener</div>
    <h1>Breakout <em>Intelligence</em> <span class="version">v3.1</span></h1>
    <div class="meta">// {analyzed} VON {total_tickers} AKTIEN NACH VORFILTER · 10+ INDIKATOREN · KOMBINATIONS-BONI · {gen_time}{_ms_badge}</div>
    <button class="theme-toggle" onclick="toggleTheme()">☀ / ☾</button>
  </header>

  <div class="stats">
    <div class="stat"><div class="slabel">MA60 Breakouts</div><div class="sval green">{crossovers}</div></div>
    <div class="stat"><div class="slabel">Sehr stark ≥60</div><div class="sval orange">{sehr_stark}</div></div>
    <div class="stat"><div class="slabel">Stark ≥45</div><div class="sval blue">{stark}</div></div>
    <div class="stat"><div class="slabel">Analysiert</div><div class="sval white">{analyzed} <span style="font-size:14px;color:var(--text-muted)">/ {total_tickers}</span></div></div>
    <div class="score-dist">
      <div class="slabel">Score-Verteilung</div>
      <div class="dist-bars">{dist_bars_html}</div>
      <div class="dist-labels">{dist_labels_html}</div>
    </div>
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
        <th onclick="sort(0)">TICKER<span class="sort-arrow">▲</span></th>
        <th onclick="sort(1)">SIGNAL<span class="sort-arrow">▲</span></th>
        <th onclick="sort(2)">SCORE-BALKEN<span class="sort-arrow">▲</span></th>
        <th onclick="sort(3)">KURS<span class="sort-arrow">▲</span></th>
        <th>MA60</th>
        <th onclick="sort(5)">ABST. MA60<span class="sort-arrow">▲</span></th>
        <th onclick="sort(6)">RSI<span class="sort-arrow">▲</span></th>
        <th onclick="sort(7)">ATR / ZIEL<span class="sort-arrow">▲</span></th>
        <th>CHART <span style="font-size:8px;color:var(--text-dim);font-weight:400">(30T)</span></th>
        <th>AKTIVE INDIKATOREN</th>
        <th onclick="sort(10)">SCORE (0–100)<span class="sort-arrow">▲</span></th>
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
      <div style="margin-top:16px;font-size:10px;color:var(--text-dim);line-height:1.8">
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
let sortedCol = null;

// ── Upgrade 8: Dark/Light Toggle ──────────────────────────
function toggleTheme() {{
  const html = document.documentElement;
  const cur = html.getAttribute('data-theme');
  html.setAttribute('data-theme', cur === 'light' ? 'dark' : 'light');
  localStorage?.setItem?.('bk-theme', html.getAttribute('data-theme'));
}}
// Restore saved theme
try {{
  const saved = localStorage?.getItem?.('bk-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
}} catch(e) {{}}

function toggleNews(ticker) {{
  const row = document.getElementById('news-' + ticker);
  if (!row) return;
  const isOpen = row.style.display !== 'none';
  row.style.display = isOpen ? 'none' : 'table-row';
  const arrow = row.previousElementSibling?.querySelector('.news-arrow');
  if (arrow) arrow.textContent = isOpen ? '▶' : '▼';
}}

function setF(f, btn) {{
  curF = f;
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  filter();
  // Upgrade 10: Smooth scroll to table after filter
  document.querySelector('.table-wrap')?.scrollIntoView({{behavior:'smooth',block:'nearest'}});
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

// ── Upgrade 9: Sort with arrow indicators ─────────────────
function sort(col) {{
  dirs[col] = -(dirs[col] || 1);
  sortedCol = col;
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

  // Update sort arrows
  document.querySelectorAll('th').forEach(th => th.classList.remove('sorted'));
  const ths = document.querySelectorAll('thead th');
  if (ths[col]) {{
    ths[col].classList.add('sorted');
    const arrow = ths[col].querySelector('.sort-arrow');
    if (arrow) arrow.textContent = dirs[col] === 1 ? '▲' : '▼';
  }}
}}

// ── Upgrade 6: Hover Detail Card (fixed) ──────────────────
const hoverCard = document.createElement('div');
hoverCard.className = 'hover-card';
hoverCard.innerHTML = '<div id="hc-inner"></div>';
document.body.appendChild(hoverCard);

let hoverTimer = null;
let activeHoverRow = null;

// Use event delegation on tbody for reliable enter/leave
const tbody_el = document.getElementById('tbody');
if (tbody_el) {{
  tbody_el.addEventListener('mouseover', e => {{
    const row = e.target.closest('.data-row');
    if (!row || row === activeHoverRow) return;
    // If row has news toggle (onclick attr), still show hover card
    activeHoverRow = row;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {{
      if (activeHoverRow !== row) return; // Mouse already left
      const ticker = row.dataset.ticker || row.cells[0]?.textContent?.trim() || '';
      const score  = row.dataset.score || '';
      const price  = row.dataset.price || '';
      const rsi    = row.dataset.rsi || '';
      const perf   = row.dataset.perf || '';
      const vol    = row.dataset.vol || '';
      const sig    = row.dataset.signal || '';
      const cat    = row.dataset.category || '';
      const sparkCell = row.cells[8];
      const sparkSvg  = sparkCell ? sparkCell.innerHTML : '';

      const sigColor = sig.includes('CROSS') ? 'var(--green)' :
                       sig.includes('ABOVE') ? 'var(--blue)' : 'var(--red)';

      // News data (optional)
      const newsRec      = row.dataset.newsRec || '';
      const newsRecColor = row.dataset.newsRecColor || '';
      const newsSent     = row.dataset.newsSentiment || '';
      const newsSentColor= row.dataset.newsSentColor || '';
      const newsReason   = row.dataset.newsReason || '';
      const newsCount    = row.dataset.newsCount || '';
      const newsHeadline = row.dataset.newsHeadline || '';

      let newsHtml = '';
      if (newsRec) {{
        const sentDot = newsSent === 'positiv' ? '▲' : newsSent === 'negativ' ? '▼' : '●';
        newsHtml =
          `<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border)">` +
          `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">` +
          `<span style="font-size:9px;color:var(--text-dim);letter-spacing:1px;text-transform:uppercase">📰 News · ${{newsCount}} Artikel</span>` +
          `<div style="display:flex;gap:6px;align-items:center">` +
          `<span style="font-size:9px;color:${{newsSentColor}};border:1px solid ${{newsSentColor}}44;padding:1px 6px;border-radius:3px;text-transform:uppercase;letter-spacing:0.5px">${{sentDot}} ${{newsSent}}</span>` +
          `<span style="font-size:10px;font-weight:700;color:${{newsRecColor}};border:1px solid ${{newsRecColor}}66;padding:2px 8px;border-radius:3px;letter-spacing:0.5px">${{newsRec}}</span>` +
          `</div></div>` +
          (newsHeadline ? `<div style="font-size:10px;color:var(--text-muted);line-height:1.5;margin-top:4px">„${{newsHeadline}}"</div>` : '') +
          (newsReason ? `<div style="font-size:9px;color:var(--text-dim);margin-top:4px;line-height:1.4">💡 ${{newsReason}}</div>` : '') +
          `</div>`;
      }}

      document.getElementById('hc-inner').innerHTML =
        `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">` +
        `<span style="font-size:18px;font-weight:900;font-family:'Courier New',monospace;color:var(--text)">${{ticker}}</span>` +
        `<span style="font-size:22px;font-weight:800;color:${{parseFloat(score)>=60?'var(--green)':parseFloat(score)>=45?'var(--orange)':'var(--red)'}}">${{score}}</span>` +
        `</div>` +
        `<div style="margin:8px 0">${{sparkSvg}}</div>` +
        `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:10px;color:var(--text-muted)">` +
        `<span>Kurs: <b style="color:var(--text)">$${{price}}</b></span>` +
        `<span>RSI: <b style="color:var(--text)">${{rsi}}</b></span>` +
        `<span>30T: <b style="color:${{parseFloat(perf)>=0?'var(--green)':'var(--red)'}}">${{perf?perf+'%':'—'}}</b></span>` +
        `<span>Vol×: <b style="color:var(--text)">${{vol}}</b></span>` +
        `</div>` +
        `<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:10px">` +
        `<span style="color:${{sigColor}}">${{sig.replace('_',' ')}}</span>` +
        ` · <span style="color:var(--text-dim)">${{cat}}</span></div>` +
        newsHtml;

      // Position: prefer below row, flip above if no space
      const rect = row.getBoundingClientRect();
      const cardH = newsRec ? 300 : 200; // taller when news section present
      let top = rect.bottom + 8;
      if (top + cardH > window.innerHeight) {{
        top = rect.top - cardH - 8;
      }}
      let left = Math.max(8, Math.min(rect.left, window.innerWidth - 400));
      hoverCard.style.left = left + 'px';
      hoverCard.style.top  = top + 'px';
      hoverCard.classList.add('visible');
    }}, 400);
  }});

  tbody_el.addEventListener('mouseleave', () => {{
    clearTimeout(hoverTimer);
    activeHoverRow = null;
    hoverCard.classList.remove('visible');
  }});

  // Also hide when mouse enters a different row quickly
  tbody_el.addEventListener('mouseout', e => {{
    const row = e.target.closest('.data-row');
    const related = e.relatedTarget?.closest?.('.data-row');
    if (row && row !== related) {{
      // Moving to a different row or leaving rows entirely
      clearTimeout(hoverTimer);
      hoverCard.classList.remove('visible');
      if (!related) activeHoverRow = null;
    }}
  }});
}}

// ── Sparkline draw animation fix ──────────────────────────
// CSS animation needs correct dasharray/offset per line
document.querySelectorAll('.spark-line').forEach(line => {{
  const len = line.getTotalLength ? line.getTotalLength() : 300;
  line.style.strokeDasharray = len;
  line.style.strokeDashoffset = len;
  line.style.setProperty('--line-len', len);
}});
</script>
</body>
</html>"""





# ══════════════════════════════════════════════
# NACHRICHTEN-ANALYSE (Yahoo Finance + Sentiment)
# ══════════════════════════════════════════════

try:
    from news_module import analyze_news_top_n, build_news_html
except ImportError:
    def analyze_news_top_n(df, n=10):
        print("\n⚠  news_module.py nicht gefunden – Nachrichten-Analyse übersprungen.")
        return {}
    def build_news_html(news_results, df):
        return ""

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("╔" + "═"*60 + "╗")
    print("║  ADVANCED BREAKOUT SCREENER v3.1 · S&P 500" + " "*17 + "║")
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
    csv_cols = ["ticker", "signal", "signal_age", "category", "score", "score_smoothed",
                "combo_score", "active_combos", "price", "atr_pct", "swing_target",
                "swing_stop", "ma60", "pct_from_ma60", "rsi", "macd_hist",
                "adx", "vol_ratio", "bb_squeeze", "obv_new_high", "near_52w_high",
                "consolidation", "perf_30d", "last_date"]
    df_results[csv_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV: {OUTPUT_CSV}")

    # HTML (jetzt mit Marktgesundheitsblock + Nachrichten)
    news_data = analyze_news_top_n(df_results, n=NEWS_TOP_N)
    html = build_html(df_results, spx=spx_data, fgi=fgi_data,
                      total_tickers=len(price_data), news_results=news_data)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML: {OUTPUT_HTML}")
    print(f"   → file://{os.path.abspath(OUTPUT_HTML)}")

    # Konsolen-Ausgabe
    print(f"\n{'═'*90}")
    print(f"  🏆 TOP {TOP_N_CONSOLE} SWING-KANDIDATEN  (Sortiert nach aktuellem Score · Haltedauer: 1–14 Tage)")
    print(f"{'═'*90}")
    print(f"  {'TICKER':<8} {'SIGNAL':<14} {'ALT':>4}  {'SCORE':>5}  {'KURS':>7}  "
          f"{'ATR%':>5}  {'ZIEL':>7}  {'STOP':>7}  {'RSI':>4}  KOMBOS")
    print(f"  {'─'*8} {'─'*14} {'─'*4}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*22}")

    for _, r in df_results.head(TOP_N_CONSOLE).iterrows():
        sig_s = (r['signal']
                 .replace("CROSSOVER_UP", "CROSS↑ HEUTE")
                 .replace("RECENT_CROSS", "CROSS↑ KÜRZL")
                 .replace("ABOVE_MA60",   "ÜBER MA60")
                 .replace("BELOW_MA60",   "UNTER MA60")
                 .replace("CROSSOVER_DOWN","CROSS↓"))
        age_s  = ("HEUTE" if r.get('signal_age')==0 else
                  f"{r.get('signal_age','?')}T alt" if r.get('signal_age',99)<99 else "—")
        combos = r.get("active_combos", "") or "—"
        cat_icon = r['category'].split()[0]
        print(f"  {r['ticker']:<8} {sig_s:<14} {age_s:>5}  {r['score']:>5.1f}  "
              f"${r['price']:>6.2f}  {r.get('atr_pct',0):>4.1f}%  "
              f"${r.get('swing_target',0):>6.2f}  ${r.get('swing_stop',0):>6.2f}  "
              f"{r['rsi']:>4.0f}  {cat_icon} {combos}")

    print(f"\n  ℹ  ALT = Tage seit MA60-Crossover  ·  ATR% = tägliche Volatilität")
    print(f"  ℹ  ZIEL = Kurs + 2×ATR(14)  ·  STOP = Kurs - 1.5×ATR(14)")
    print(f"  ℹ  Score 0–100 · ≥60 = Sehr stark · ≥45 = Stark · ≥30 = Mittel")
    print(f"  ℹ  Frische Signale (HEUTE/1T) = höher gewichtet für kurze Haltedauer\n")


if __name__ == "__main__":
    main()
