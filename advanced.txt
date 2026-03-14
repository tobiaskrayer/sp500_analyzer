"""
╔══════════════════════════════════════════════════════════════════╗
║   RUSSELL 2000 · ADVANCED BREAKOUT SCREENER                     ║
║   Multi-Indikator Scoring System für maximale Trefferquote      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INDIKATOREN:                                                    ║
║  1. MA60 Crossover          (Haupttrend-Signal)                  ║
║  2. RSI (14)                (Momentum – ideal: 50–70)            ║
║  3. MACD                    (Trend-Momentum-Kreuzung)            ║
║  4. Bollinger Band Squeeze  (Volatilitätskompression vor Ausbruch)║
║  5. OBV Trend               (Institutionelles Volumen)           ║
║  6. ADX (14)                (Trendstärke – ideal: >20)           ║
║  7. EMA 200                 (Langfristiger Trend)                ║
║  8. 52W Hochpunkt           (Nähe zu neuem Jahreshoch)           ║
║  9. Volumen-Spike           (Bestätigung durch Handelsvolumen)   ║
║ 10. Konsolidierungsmuster   (Enge Range vor Ausbruch)            ║
║                                                                  ║
║  SCORING: Max 300 Punkte → Breakout-Wahrscheinlichkeit          ║
║                                                                  ║
║  Installation:                                                   ║
║    pip install yfinance pandas requests beautifulsoup4 tqdm      ║
║                                                                  ║
║  Ausführen:                                                      ║
║    python advanced_breakout_screener.py                          ║
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
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════

LOOKBACK_DAYS  = 300   # Mehr Daten für EMA200 und 52W
MA_PERIOD      = 60
MAX_TICKERS    = None  # None = alle, z.B. 100 für Schnelltest
BATCH_SIZE     = 40
SLEEP_BETWEEN  = 1.2
OUTPUT_CSV     = "breakout_results.csv"
OUTPUT_HTML    = "breakout_report.html"
TOP_N_CONSOLE  = 15    # Wie viele in der Konsole anzeigen

# ──────────────────────────────────────────────
# SCORING-GEWICHTE (gesamt max ~300 Punkte)
# ──────────────────────────────────────────────
WEIGHTS = {
    # TREND
    "ma60_crossover_today":   50,  # MA60 heute durchbrochen ← Kernsignal
    "ma60_crossover_recent":  30,  # Crossover in letzten 3 Tagen
    "ma60_above":             15,  # Über MA60 (kein frischer Crossover)
    "ema200_above":           20,  # Preis > EMA200 (Langfrist-Aufwärtstrend)
    "ma20_above_ma60":        10,  # Goldenes Kreuz MA20 > MA60

    # MOMENTUM
    "rsi_ideal":              25,  # RSI zwischen 50–65 (stark aber nicht überkauft)
    "rsi_bullish":            10,  # RSI > 50 (generell bullisch)
    "macd_crossover":         25,  # MACD-Linie kreuzt Signal-Linie nach oben
    "macd_above_zero":        10,  # MACD-Histogramm positiv
    "macd_momentum_rising":   10,  # MACD-Histogramm steigt

    # VOLATILITÄT / SQUEEZE
    "bb_squeeze":             25,  # Bollinger Band Squeeze → Energie aufgebaut
    "bb_breakout_up":         20,  # Preis bricht über oberes BB
    "consolidation":          20,  # Enge Konsolidierung (Range < 5% in 10 Tagen)

    # VOLUMEN / INSTITUTIONELL
    "volume_spike":           25,  # Tagesvolumen > 150% des 20T-Durchschnitts
    "volume_trend":           10,  # Steigendes Volumen in letzten 5 Tagen
    "obv_rising":             20,  # OBV-Trend positiv (Institutionen kaufen)

    # STRUKTUR
    "near_52w_high":          15,  # Innerhalb 10% vom 52W-Hoch (Ausbruchs-Setup)
    "adx_strong":             15,  # ADX > 20 (Trend vorhanden, nicht seitwärts)
    "higher_highs":           10,  # Höhere Hochs in letzten 10 Tagen
}


# ══════════════════════════════════════════════
# SCHRITT 1: TICKER LADEN
# ══════════════════════════════════════════════

def get_tickers() -> list:
    print("\n📋 Lade Russell 2000 / Small Cap Ticker-Liste...")
    tickers = _try_ishares() or _try_finviz() or _fallback_list()
    print(f"   ✓ {len(tickers)} Ticker geladen")
    return tickers


def _try_ishares() -> list:
    """iShares IWM ETF Holdings = echter Russell 2000."""
    url = ("https://www.ishares.com/us/products/239710/ISHARES-RUSSELL-2000-ETF/"
           "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        lines = resp.text.split('\n')
        start = next((i for i, l in enumerate(lines) if l.startswith('Ticker') or l.startswith('"Ticker"')), None)
        if start is None: return []
        tickers = []
        for line in lines[start+1:]:
            t = line.split(',')[0].strip().strip('"')
            if t and t.isalpha() and 1 <= len(t) <= 5 and t != 'Ticker':
                tickers.append(t)
        return tickers if len(tickers) > 100 else []
    except Exception:
        return []


def _try_finviz() -> list:
    """Finviz Small Cap Screener als Fallback."""
    from bs4 import BeautifulSoup
    tickers = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for page in range(1, 51):
        try:
            url = f"https://finviz.com/screener.ashx?v=111&f=cap_small,exch_nasd|nyse&r={((page-1)*20)+1}"
            soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, 'html.parser')
            links = soup.find_all('a', {'class': 'screener-link-primary'})
            if not links: break
            tickers.extend([l.text.strip() for l in links])
            time.sleep(0.4)
        except Exception:
            break
    return list(dict.fromkeys(tickers))


def _fallback_list() -> list:
    print("   → Verwende eingebettete Small-Cap-Liste...")
    return [
        "AACG","ACMR","ACLS","ADMA","AEHR","AEIS","AGEN","AGIO","ALKS","ALRM",
        "AMKR","AMNB","AMPH","AMSC","ANIP","APOG","APPF","APPN","ARKO","ARLO",
        "ARQT","ARWR","ASND","ASPS","ASTE","ATEC","ATNI","ATRO","ATSG","AVAV",
        "AVNS","AXGN","AXNX","AXSM","AXTI","BAND","BANF","BANR","BCBP","BCRX",
        "BECN","BELFA","BGFV","BLBD","BLKB","BLMN","BMNM","BOOT","BORR","BOSC",
        "BRDG","BRKL","BRNS","BSIG","BUSE","CACC","CAKE","CALX","CAMP","CAMT",
        "CARE","CASH","CASS","CATO","CBFV","CBRL","CCNE","CCRN","CDMO","CDNA",
        "CDRE","CEVA","CFBK","CFFN","CHUY","CIVB","CLBK","CLMB","CLSK","CMCO",
        "CMTL","CNMD","CNXN","COCP","CODX","COFS","CORT","CRAI","CRDF","CRMT",
        "CRUS","CRVL","CRWS","CSWI","CTBI","CTOS","CVBF","CVCO","CVLY","CVLG",
        "CWCO","DAIO","DAKT","DFIN","DGII","DLHC","DLTH","DOOR","DORM","DXLG",
        "DXPE","EASTW","ECPG","EDUC","EGIO","EIGR","ENSG","EPAC","ERII","ESAB",
        "ESEA","ESNT","ESTA","ESXB","ETWO","EWBC","EXFI","EXPI","EXTR","EZPW",
        "FATH","FBMS","FBNC","FCAP","FCEL","FDBC","FEIM","FELE","FFBC","FFIN",
        "FFNW","FFWM","FIBK","FISI","FIVN","FLGC","FLGT","FLNC","FLNX","FLOW",
        "FLWS","FMAO","FMBH","FNKO","FNLC","FOLD","FONR","FORR","FRGE","FRME",
        "FRPH","FRPT","FRST","FSBC","FSFG","FTCI","FULT","FUSB","GCMG","GEOS",
        "GEVO","GIGM","GIII","GLDD","GLNG","GLPI","GNLX","GNTX","GPOR","GPRK",
        "GPRO","GRBK","GSHD","GSIT","HAYN","HCKT","HCSG","HEAR","HEES","HELE",
        "HFFG","HFWA","HIBB","HIMX","HLTH","HOFT","HOLI","HOOK","HSKA","HTBK",
        "HTGC","HTLD","HTLF","HUBG","HURN","HWKN","IART","IBCP","IBIO","IBOC",
        "IBTX","ICAD","ICHR","ICPT","IDCC","IDEX","IDYA","IFRX","IGIC","IINN",
        "IMKTA","IMMR","INBK","INFU","INGN","INMD","INNV","INSE","INSG","INSW",
        "INTG","INTT","INVA","IRBT","IRDM","IRMD","IRWD","ISIG","ISPR","ITGR",
        "ITRI","IVAC","JACK","JAKK","JAMF","JBSS","JKHY","JOUT","JRVR","JYNT",
        "KALU","KAMN","KBAL","KFRC","KLIC","KNSA","KNSL","KOSS","KNDI","KPTI",
        "KRNT","KRYS","KSCP","KVHI","LAKE","LANC","LAND","LASR","LAUR","LAWS",
        "LCII","LCNB","LCUT","LECO","LEGH","LESL","LFUS","LGIH","LGND","LHCG",
        "LITE","LJPC","LKFN","LLAP","LMAT","LMND","LNKB","LNTH","LOCO","LOGI",
        "LOPE","LSCC","LSTR","LTBR","LUMO","MACK","MATW","MBCN","MBIN","MBUU",
        "MBWM","MCBC","MCHX","MDGL","MDXG","MEIP","MEOH","MERC","MGEE","MGIC",
        "MGRC","MIND","MITK","MLKN","MMSI","MNKD","MNRO","MNSB","MNTX","MOFG",
        "MORF","MRTN","MSBF","MSFG","MTEM","MTRX","MVBF","MXCT","MYFW","MYRG",
        "NATR","NBTB","NCBS","NDLS","NDSN","NEON","NESR","NETI","NEWT","NGVT",
        "NICK","NMIH","NMRK","NNBR","NOMD","NOVA","NOVN","NOVT","NSIT","NSSC",
        "NTGR","NTUS","NWFL","NWPX","OFIX","OFLX","OMCL","OMER","ONBK","ORGO",
        "ORRF","OSBC","OSIS","OSMT","OTTR","OVBC","PACW","PALI","PANL","PARR",
        "PATI","PBCP","PBHC","PBIP","PDCO","PEAK","PECK","PEGA","PETQ","PFBC",
        "PFHD","PFIS","PHAT","PKBK","PKOH","PLAY","PLBC","PLBY","PLCE","PLMR",
        "PLPC","PLSE","PLUG","PLUS","PLXS","PNFP","PNTG","PODD","POOL","POWI",
        "POWL","PRAA","PRAX","PRCH","PRDO","PRFT","PRGS","PROS","PRPH","PRPL",
        "PRTH","PRTK","PRTS","PSEC","PSIX","PTCT","PTLO","PTMN","PTSI","QUAD",
        "QUIK","QLYS","RADI","RAPT","RCKY","RCMT","RDNT","RDVT","REAL","RECO",
        "REED","RGEN","RFIL","RICK","RIGL","RILY","RKLY","RLGT","RLMD","RMCF",
        "RMNI","RNST","ROAD","ROIC","ROLL","ROSE","RPAY","RPID","RPTX","RRBI",
        "RRGB","RSLS","RUSHA","RYAM","SAFE","SAGA","SAGE","SAIA","SASR","SATS",
        "SBCF","SBFG","SBSI","SCHL","SCHN","SCSC","SEED","SEER","SELB","SFNC",
        "SFST","SGMO","SHBI","SHCR","SHLS","SHOS","SIBN","SIGA","SIGI","SILK",
        "SIRE","SIXS","SKGR","SKWD","SKYW","SLCA","SLGN","SMAR","SMBC","SMFL",
        "SMID","SMMF","SMPL","SMSI","SMTC","SNBR","SNCY","SNEX","SNGX","SNPO",
        "SNSC","SNTI","SOBR","SOHO","SOLV","SONO","SOTK","SPFI","SPGX","SPNS",
        "SPOK","SPPI","SPSC","SPTN","SSBI","SSIC","SSKN","SSNC","SSRM","STAA",
        "STBA","STGW","STIM","STNG","STON","STRL","STRN","STRS","STSA","STXS",
        "SWBI","SXTP","SYBT","SYBX","TACT","TBPH","TCBK","TCBS","TCCO","TCFC",
        "TCMD","TCPC","TELA","TELL","TERN","TFII","TGNA","THFF","TIGO","TISI",
        "TITN","TLIS","TMBR","TMCI","TMDX","TORC","TOWN","TPVG","TRDX","TREE",
        "TRIN","TRIP","TRMK","TRMT","TRNS","TROX","TRTX","TRVS","TRWH","TTEC",
        "TTGT","TTMI","TUYA","TVTX","TWST","TXMD","TXRH","TYRA","TZOO","UAMY",
        "UBCP","UBFO","UBSI","UCBI","UCTT","UEIC","UEPS","UFAB","UFPI","UFPT",
        "UGRO","UMBF","UNFI","UNIT","UNTY","UPBD","UPLD","URGN","USAP","USAS",
        "USBI","USEG","USFD","USIO","USLM","USPH","UTMD","UVSP","VBFC","VBIV",
        "VCTR","VCYT","VECO","VICR","VLGEA","VNDA","VOXX","VRCA","VRDN","VREX",
        "VRIG","VRSN","VRTS","VSCO","VSEC","VSTO","VTGN","VTOL","VTSI","WABC",
        "WAFD","WASH","WBHC","WDFC","WEBR","WEYS","WFRD","WIRE","WKHS","WLDN",
        "WLFC","WMGI","WNEB","WOOF","WSBC","WSBF","WSFS","WSTL","WTRG","WTRH",
        "XAIR","XBIO","XBIT","XCUR","XNCR","XOMA","XPER","XPON","XPRO","YELL",
        "YMTX","YORW","ZAPP","ZDGE","ZEAL","ZELK","ZETA","ZGLS","ZIOP","ZIXI",
        "ZLAB","ZUMZ","ZVRA","ZYME","ZYXI",
    ]


# ══════════════════════════════════════════════
# SCHRITT 2: KURSDATEN LADEN
# ══════════════════════════════════════════════

def download_prices(tickers: list) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    data  = {}

    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]

    print(f"\n📥 Lade Kursdaten für {len(tickers)} Ticker ({start.date()} → {end.date()})...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Downloading"):
        try:
            raw = yf.download(
                batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                group_by="ticker", auto_adjust=True, progress=False, threads=True,
            )
            for t in batch:
                try:
                    df = raw[t].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 80:  # Mindestens 80 Handelstage
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            pass
        time.sleep(SLEEP_BETWEEN)

    print(f"   ✓ {len(data)} Ticker mit ausreichend Daten ({LOOKBACK_DAYS} Tage)")
    return data


# ══════════════════════════════════════════════
# SCHRITT 3: INDIKATOREN BERECHNEN
# ══════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series: pd.Series, period=20, std_dev=2.0):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bw    = (upper - lower) / sma  # Bandwidth
    return upper, lower, sma, bw


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def calc_adx(high, low, close, period=14) -> pd.Series:
    """Average Directional Index."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def analyze(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df    = df.sort_index()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df.get("Volume", pd.Series(dtype=float))

        n = len(close)
        if n < 80: return None

        # ── MOVING AVERAGES ──────────────────────────────
        ma60  = close.rolling(60).mean()
        ma20  = close.rolling(20).mean()
        ema200= close.ewm(span=200, adjust=False).mean()

        today_c   = float(close.iloc[-1])
        yest_c    = float(close.iloc[-2])
        today_ma60= float(ma60.iloc[-1])
        yest_ma60 = float(ma60.iloc[-2])
        today_ema = float(ema200.iloc[-1])

        if any(np.isnan([today_c, yest_c, today_ma60, yest_ma60])): return None

        # ── RSI ──────────────────────────────────────────
        rsi      = calc_rsi(close)
        today_rsi= float(rsi.iloc[-1])

        # ── MACD ─────────────────────────────────────────
        macd_l, macd_sig, macd_hist = calc_macd(close)
        today_macd      = float(macd_l.iloc[-1])
        today_sig       = float(macd_sig.iloc[-1])
        today_hist      = float(macd_hist.iloc[-1])
        yest_hist       = float(macd_hist.iloc[-2])
        yest_macd_l     = float(macd_l.iloc[-2])
        yest_sig_l      = float(macd_sig.iloc[-2])

        # ── BOLLINGER BANDS ──────────────────────────────
        bb_up, bb_lo, bb_mid, bb_bw = calc_bollinger(close)
        today_bw      = float(bb_bw.iloc[-1])
        hist_bw       = bb_bw.iloc[-60:-1]
        bw_percentile = (hist_bw < today_bw).mean() if len(hist_bw) > 10 else 0.5
        # Squeeze = aktuelle BW im unteren 20. Percentil der letzten 60T
        bb_squeeze    = bw_percentile < 0.20
        bb_above_up   = today_c > float(bb_up.iloc[-1])

        # ── OBV ──────────────────────────────────────────
        obv_trend = False
        if vol is not None and len(vol) > 10 and vol.sum() > 0:
            obv       = calc_obv(close, vol)
            obv_ma10  = obv.rolling(10).mean()
            obv_trend = float(obv.iloc[-1]) > float(obv_ma10.iloc[-1])

        # ── ADX ──────────────────────────────────────────
        adx_s = None
        plus_di_v = None
        try:
            adx_s, plus_di, minus_di = calc_adx(high, low, close)
            adx_val    = float(adx_s.iloc[-1])
            plus_di_v  = float(plus_di.iloc[-1])
            minus_di_v = float(minus_di.iloc[-1])
        except Exception:
            adx_val = 0.0
            plus_di_v = 0.0
            minus_di_v = 0.0

        # ── VOLUMEN ──────────────────────────────────────
        vol_spike = False
        vol_trend = False
        vol_ratio = 0.0
        if vol is not None and len(vol) >= 20 and vol.sum() > 0:
            avg_vol  = float(vol.iloc[-21:-1].mean())
            today_v  = float(vol.iloc[-1])
            vol_ratio= today_v / avg_vol if avg_vol > 0 else 0
            vol_spike= vol_ratio > 1.5
            # Volumen-Trend: Durchschnitt letzter 5 Tage > vorherige 5 Tage
            recent5  = float(vol.iloc[-5:].mean())
            prev5    = float(vol.iloc[-10:-5].mean())
            vol_trend= recent5 > prev5 * 1.1 if prev5 > 0 else False

        # ── 52W-HOCH ─────────────────────────────────────
        high_52w   = float(high.rolling(min(252, n)).max().iloc[-1])
        pct_52w    = (today_c / high_52w) if high_52w > 0 else 0
        near_52w   = pct_52w >= 0.90  # Innerhalb 10% vom Jahreshoch

        # ── KONSOLIDIERUNG ────────────────────────────────
        range_10d  = float(high.iloc[-10:].max()) - float(low.iloc[-10:].min())
        mid_10d    = float(close.iloc[-10:].mean())
        consol_pct = range_10d / mid_10d if mid_10d > 0 else 999
        consolidation = consol_pct < 0.05  # Range < 5% → Kompression

        # ── HÖHERE HOCHS ─────────────────────────────────
        highs_5    = [float(high.iloc[-i]) for i in range(1, 6)]
        higher_highs = highs_5[0] > highs_5[2] and highs_5[2] > highs_5[4]

        # ── MA KREUZUNGEN ────────────────────────────────
        cross_up   = (yest_c <= yest_ma60) and (today_c > today_ma60)
        cross_down = (yest_c >= yest_ma60) and (today_c < today_ma60)
        above_ma60 = today_c > today_ma60
        above_ema200   = today_c > today_ema and not np.isnan(today_ema)
        ma20_above_ma60= float(ma20.iloc[-1]) > today_ma60

        # Crossover in letzten 3 Tagen?
        recent_cross = any(
            close.iloc[-j-1] <= ma60.iloc[-j-1] and close.iloc[-j] > ma60.iloc[-j]
            for j in range(2, min(4, n-1))
        )

        # ── MACD CROSSOVER ────────────────────────────────
        macd_crossover = (yest_macd_l <= yest_sig_l) and (today_macd > today_sig)
        macd_above_zero = today_hist > 0
        macd_hist_rising = today_hist > yest_hist

        # ── RSI BEWERTUNG ────────────────────────────────
        rsi_ideal   = 50 <= today_rsi <= 65   # Stark aber nicht überkauft
        rsi_bullish = today_rsi > 50

        # ── SCORING ──────────────────────────────────────
        score     = 0
        details   = {}

        def add(key, condition):
            nonlocal score
            pts = WEIGHTS.get(key, 0) if condition else 0
            score += pts
            details[key] = {"active": bool(condition), "pts": pts}

        add("ma60_crossover_today", cross_up)
        add("ma60_crossover_recent", recent_cross and not cross_up)
        add("ma60_above", above_ma60 and not cross_up and not recent_cross)
        add("ema200_above", above_ema200)
        add("ma20_above_ma60", ma20_above_ma60)
        add("rsi_ideal", rsi_ideal)
        add("rsi_bullish", rsi_bullish and not rsi_ideal)
        add("macd_crossover", macd_crossover)
        add("macd_above_zero", macd_above_zero)
        add("macd_momentum_rising", macd_hist_rising)
        add("bb_squeeze", bb_squeeze)
        add("bb_breakout_up", bb_above_up)
        add("consolidation", consolidation)
        add("volume_spike", vol_spike)
        add("volume_trend", vol_trend)
        add("obv_rising", obv_trend)
        add("near_52w_high", near_52w)
        add("adx_strong", adx_val > 20)
        add("higher_highs", higher_highs)

        # ── SIGNAL-LABEL ─────────────────────────────────
        if cross_up:            signal = "CROSSOVER_UP"
        elif recent_cross:      signal = "RECENT_CROSS"
        elif cross_down:        signal = "CROSSOVER_DOWN"
        elif above_ma60:        signal = "ABOVE_MA60"
        else:                   signal = "BELOW_MA60"

        # Breakout-Wahrscheinlichkeit (normiert 0–100%)
        max_score = sum(WEIGHTS.values())
        breakout_prob = round(score / max_score * 100, 1)

        # Kategorie
        if breakout_prob >= 60:   category = "🔥 SEHR STARK"
        elif breakout_prob >= 45: category = "📈 STARK"
        elif breakout_prob >= 30: category = "🟡 MITTEL"
        else:                     category = "⬇ SCHWACH"

        pct_from_ma60 = (today_c - today_ma60) / today_ma60 * 100
        perf_30d = ((today_c - float(close.iloc[-30])) / float(close.iloc[-30]) * 100) if n >= 30 else None

        return {
            "ticker":         ticker,
            "signal":         signal,
            "category":       category,
            "score":          score,
            "max_score":      max_score,
            "breakout_prob":  breakout_prob,
            "price":          round(today_c, 2),
            "ma60":           round(today_ma60, 2),
            "ema200":         round(today_ema, 2) if not np.isnan(today_ema) else None,
            "pct_from_ma60":  round(pct_from_ma60, 2),
            "rsi":            round(today_rsi, 1),
            "macd_hist":      round(today_hist, 4),
            "adx":            round(adx_val, 1),
            "vol_ratio":      round(vol_ratio, 2),
            "bb_squeeze":     bb_squeeze,
            "obv_rising":     obv_trend,
            "perf_30d":       round(perf_30d, 2) if perf_30d else None,
            "near_52w_high":  near_52w,
            "pct_52w":        round(pct_52w * 100, 1),
            "consolidation":  consolidation,
            "last_date":      df.index[-1].strftime("%Y-%m-%d"),
            "sparkline":      [round(float(c), 2) for c in close.iloc[-30:]],
            "details":        details,
        }

    except Exception as e:
        return None


def analyze_all(price_data: dict) -> pd.DataFrame:
    print(f"\n🔬 Analysiere {len(price_data)} Ticker mit 10 Indikatoren...")
    results = [r for r in (analyze(t, df) for t, df in tqdm(price_data.items(), desc="Analysiere")) if r]
    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"   ✓ {len(df)} Aktien erfolgreich ausgewertet")
    return df


# ══════════════════════════════════════════════
# SCHRITT 4: HTML-REPORT
# ══════════════════════════════════════════════

def build_html(df: pd.DataFrame) -> str:
    top = df.head(150)

    MAX_TOTAL = sum(WEIGHTS.values())

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
        d = row.get("details", {})
        pill_map = [
            ("rsi_ideal",         f"RSI {row['rsi']:.0f}✓"),
            ("rsi_bullish",       f"RSI {row['rsi']:.0f}"),
            ("macd_crossover",    "MACD✗"),
            ("macd_above_zero",   "MACD+"),
            ("bb_squeeze",        "BB-SQZ"),
            ("bb_breakout_up",    "BB↑"),
            ("obv_rising",        "OBV↑"),
            ("adx_strong",        f"ADX {row['adx']:.0f}"),
            ("volume_spike",      f"VOL×{row['vol_ratio']:.1f}"),
            ("volume_trend",      "VOL↑"),
            ("ema200_above",      "EMA200✓"),
            ("near_52w_high",     f"52W {row['pct_52w']}%"),
            ("consolidation",     "CONSOL"),
            ("higher_highs",      "HH"),
        ]
        colors = {
            "rsi_ideal": "#ff9944", "rsi_bullish": "#ff9944",
            "macd_crossover": "#ff44aa", "macd_above_zero": "#ff44aa",
            "bb_squeeze": "#44ffdd", "bb_breakout_up": "#44ffdd",
            "obv_rising": "#4488ff",
            "adx_strong": "#ffdd44",
            "volume_spike": "#ff6644", "volume_trend": "#ff6644",
            "ema200_above": "#88ff44",
            "near_52w_high": "#aa88ff",
            "consolidation": "#44ffaa", "higher_highs": "#88ccff",
        }
        for key, label in pill_map:
            if d.get(key, {}).get("active"):
                c = colors.get(key, "#888")
                pills.append(f'<span style="background:rgba(255,255,255,0.05);border:1px solid {c}44;color:{c};padding:2px 7px;border-radius:3px;font-size:9px;margin:1px;display:inline-block">{label}</span>')
        return "".join(pills)

    def spark(prices):
        if not prices or len(prices) < 2: return ""
        mn, mx = min(prices), max(prices)
        rng = mx - mn or 1
        pts = " ".join(f"{(i/(len(prices)-1))*100:.1f},{36 - ((p-mn)/rng)*32 - 2:.1f}" for i, p in enumerate(prices))
        c = "#00ff88" if prices[-1] >= prices[0] else "#ff4466"
        return f'<svg width="100" height="36" viewBox="0 0 100 36"><polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/></svg>'

    def prob_bar(prob):
        if prob >= 60:   bar_c = "#00ff88"
        elif prob >= 45: bar_c = "#ffaa00"
        elif prob >= 30: bar_c = "#ff9944"
        else:            bar_c = "#ff4466"
        return f'''<div style="display:flex;align-items:center;gap:8px;min-width:140px">
          <div style="flex:1;height:5px;background:#1a1a2e;border-radius:3px;overflow:hidden">
            <div style="width:{prob}%;height:100%;background:{bar_c};border-radius:3px;transition:width 0.5s"></div>
          </div>
          <span style="color:{bar_c};font-size:11px;font-weight:700;min-width:35px">{prob}%</span>
        </div>'''

    def pct(val, show_sign=True):
        if val is None: return "—"
        c = "#00ff88" if val >= 0 else "#ff4466"
        s = "+" if val >= 0 and show_sign else ""
        return f'<span style="color:{c}">{s}{val:.2f}%</span>'

    rows_html = ""
    for i, row in top.iterrows():
        border_l = ' style="border-left:3px solid #00ff88"' if row["signal"] in ("CROSSOVER_UP","RECENT_CROSS") else ""
        rows_html += f"""
        <tr{border_l} class="data-row"
            data-signal="{row['signal']}"
            data-category="{row['category']}"
            data-score="{row['score']}">
          <td style="padding:10px 14px;font-weight:900;font-size:14px;color:#fff;font-family:'Courier New',monospace">{row['ticker']}</td>
          <td style="padding:10px 14px">{badge(row['signal'])}</td>
          <td style="padding:10px 14px">{prob_bar(row['breakout_prob'])}</td>
          <td style="padding:10px 14px;color:#fff;font-weight:700">${row['price']:.2f}</td>
          <td style="padding:10px 14px;color:#666;font-size:11px">${row['ma60']:.2f}</td>
          <td style="padding:10px 14px">{pct(row['pct_from_ma60'])}</td>
          <td style="padding:10px 14px;color:{'#ff9944' if 50 <= row['rsi'] <= 65 else '#e8e8f0'}">{row['rsi']:.0f}</td>
          <td style="padding:10px 14px">{pct(row['perf_30d'])}</td>
          <td style="padding:10px 14px">{spark(row['sparkline'])}</td>
          <td style="padding:10px 4px">{indicator_pills(row)}</td>
          <td style="padding:10px 14px;font-weight:800;color:#fff;font-size:13px">{int(row['score'])}<span style="color:#333;font-size:10px">/{MAX_TOTAL}</span></td>
        </tr>"""

    # Stats
    crossovers   = len(df[df["signal"].isin(["CROSSOVER_UP","RECENT_CROSS"])])
    sehr_stark   = len(df[df["category"] == "🔥 SEHR STARK"])
    stark        = len(df[df["category"] == "📈 STARK"])
    analyzed     = len(df)
    gen_time     = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Indikator-Legende
    legend_items = [
        ("RSI 50–65", "Momentum ideal – stark ohne überkauft", "#ff9944"),
        ("MACD✗", "MACD-Linie kreuzt Signal nach oben", "#ff44aa"),
        ("BB-SQZ", "Bollinger Band Squeeze → Energie aufgebaut", "#44ffdd"),
        ("OBV↑", "On-Balance Volume steigt (Inst. kaufen)", "#4488ff"),
        ("ADX >20", "Starker Trend vorhanden (kein Seitwärts)", "#ffdd44"),
        ("VOL×n", "Volumen-Spike > 150% Durchschnitt", "#ff6644"),
        ("EMA200✓", "Preis über 200T-EMA (Langfristtrend ok)", "#88ff44"),
        ("52W%", "Nähe zum 52-Wochen-Hoch", "#aa88ff"),
        ("CONSOL", "Enge Konsolidierung <5% Range in 10T", "#44ffaa"),
        ("HH", "Höhere Hochs – Aufwärtsstruktur", "#88ccff"),
    ]
    legend_html = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="background:rgba(255,255,255,0.05);border:1px solid {c}44;color:{c};padding:2px 8px;border-radius:3px;font-size:10px;min-width:70px;text-align:center">{name}</span>'
        f'<span style="font-size:11px;color:#555">{desc}</span></div>'
        for name, desc, c in legend_items
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Advanced Breakout Screener · Russell 2000</title>
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
  .meta{{margin-top:10px;font-size:11px;color:#333;letter-spacing:1px}}
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:36px}}
  .stat{{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:16px 20px}}
  .slabel{{font-size:9px;letter-spacing:2px;color:#333;text-transform:uppercase;margin-bottom:6px}}
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
  th{{padding:12px 14px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#333;cursor:pointer;white-space:nowrap;user-select:none}}
  th:hover{{color:#888}}
  tbody tr{{border-bottom:1px solid rgba(26,26,40,.6);transition:background .1s}}
  tbody tr:hover{{background:#0f0f18}}
  tbody tr:last-child{{border-bottom:none}}
  .legend{{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:24px;margin-top:32px}}
  .legend h3{{font-family:'Syne',sans-serif;font-size:12px;letter-spacing:2px;color:#444;text-transform:uppercase;margin-bottom:16px}}
  .legend-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:4px}}
  .footer{{margin-top:24px;font-size:10px;color:#222;letter-spacing:1px;text-align:center}}
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
  .data-row{{animation:fadeUp .3s ease both}}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="tag">Russell 2000 · NYSE/NASDAQ · Advanced Screener</div>
    <h1>Breakout<br><em>Intelligence</em></h1>
    <div class="meta">// {analyzed} AKTIEN · 10 INDIKATOREN · GENERIERT {gen_time}</div>
  </header>

  <div class="stats">
    <div class="stat"><div class="slabel">MA60 Breakouts</div><div class="sval green">{crossovers}</div></div>
    <div class="stat"><div class="slabel">Sehr stark ≥60%</div><div class="sval orange">{sehr_stark}</div></div>
    <div class="stat"><div class="slabel">Stark ≥45%</div><div class="sval blue">{stark}</div></div>
    <div class="stat"><div class="slabel">Max Score</div><div class="sval white">{MAX_TOTAL}</div></div>
    <div class="stat"><div class="slabel">Analysiert</div><div class="sval white">{analyzed}</div></div>
  </div>

  <div class="controls">
    <input class="search" type="text" id="searchBox" placeholder="TICKER SUCHEN..." oninput="filter()"/>
    <div class="filters">
      <button class="fbtn on" onclick="setF('all',this)">ALLE</button>
      <button class="fbtn" onclick="setF('crossover',this)">🔥 BREAKOUT</button>
      <button class="fbtn" onclick="setF('sehr_stark',this)">⚡ SEHR STARK</button>
      <button class="fbtn" onclick="setF('stark',this)">📈 STARK</button>
      <button class="fbtn" onclick="setF('above',this)">ÜBER MA60</button>
    </div>
  </div>

  <div class="table-wrap">
    <table id="mainTable">
      <thead><tr>
        <th onclick="sort(0)">TICKER</th>
        <th onclick="sort(1)">SIGNAL</th>
        <th onclick="sort(2)">BREAKOUT-WAHRSCH.</th>
        <th onclick="sort(3)">KURS</th>
        <th>MA60</th>
        <th onclick="sort(5)">ABST. MA60</th>
        <th onclick="sort(6)">RSI</th>
        <th onclick="sort(7)">30T PERF</th>
        <th>CHART</th>
        <th>AKTIVE INDIKATOREN</th>
        <th onclick="sort(10)">SCORE</th>
      </tr></thead>
      <tbody id="tbody">{rows_html}</tbody>
    </table>
  </div>

  <div class="legend">
    <h3>// Indikator-Legende</h3>
    <div class="legend-grid">{legend_html}</div>
  </div>

  <div class="footer">
    // Russell 2000 Advanced Breakout Screener · Daten: Yahoo Finance via yfinance<br>
    // Nicht als Anlageberatung zu verstehen · Past performance ≠ future results
  </div>

</div>
<script>
let curF = 'all';
let sortCol = 10, sortDir = -1;

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
    const sc = parseInt(row.dataset.score || 0);

    let mf = true;
    if (curF === 'crossover') mf = sig.includes('CROSSOVER_UP') || sig.includes('RECENT_CROSS');
    if (curF === 'sehr_stark') mf = cat.includes('SEHR STARK');
    if (curF === 'stark') mf = cat.includes('STARK') || cat.includes('SEHR STARK');
    if (curF === 'above') mf = !sig.includes('BELOW') && !sig.includes('DOWN');

    const mq = !q || ticker.includes(q);
    row.style.display = (mf && mq) ? '' : 'none';
  }});
}}

const dirs = {{}};
function sort(col) {{
  dirs[col] = -(dirs[col] || 1);
  const tbody = document.getElementById('tbody');
  const rows = Array.from(tbody.querySelectorAll('.data-row'));
  rows.sort((a, b) => {{
    const av = a.cells[col]?.textContent.replace(/[^0-9.\-]/g,'') || '';
    const bv = b.cells[col]?.textContent.replace(/[^0-9.\-]/g,'') || '';
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
    print("╔" + "═"*58 + "╗")
    print("║  ADVANCED BREAKOUT SCREENER · RUSSELL 2000" + " "*15 + "║")
    print("║  10 Indikatoren · Multi-Signal Scoring System" + " "*11 + "║")
    print("╚" + "═"*58 + "╝")

    tickers    = get_tickers()
    price_data = download_prices(tickers)
    df_results = analyze_all(price_data)

    # CSV
    csv_cols = ["ticker","signal","category","score","breakout_prob","price","ma60",
                "pct_from_ma60","rsi","macd_hist","adx","vol_ratio","bb_squeeze",
                "obv_rising","near_52w_high","consolidation","perf_30d","last_date"]
    df_results[csv_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV: {OUTPUT_CSV}")

    # HTML
    html = build_html(df_results)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML: {OUTPUT_HTML}")
    print(f"   → file://{os.path.abspath(OUTPUT_HTML)}")

    # ── KONSOLEN-AUSGABE ────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  🏆 TOP {TOP_N_CONSOLE} AKTIEN MIT HÖCHSTEM BREAKOUT-POTENZIAL")
    print(f"{'═'*72}")
    print(f"  {'TICKER':<8} {'SIGNAL':<16} {'PROB':>5} {'SCORE':>6}  {'KURS':>7}  "
          f"{'RSI':>5}  {'MA60%':>6}  AKTIVE SIGNALE")
    print(f"  {'─'*8} {'─'*16} {'─'*5} {'─'*6}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*25}")

    for _, r in df_results.head(TOP_N_CONSOLE).iterrows():
        d = r.get("details", {})
        active = [k for k, v in d.items() if v.get("active")]
        active_short = ", ".join(active[:5]) + ("…" if len(active) > 5 else "")
        sig_short = r['signal'].replace("CROSSOVER_UP","CROSS↑").replace("RECENT_CROSS","RECENT↑") \
                               .replace("ABOVE_MA60","ABOVE").replace("BELOW_MA60","BELOW") \
                               .replace("CROSSOVER_DOWN","CROSS↓")
        cat_icon = r['category'].split()[0]
        print(f"  {r['ticker']:<8} {sig_short:<16} {r['breakout_prob']:>4.0f}% "
              f"{int(r['score']):>5}/{r['max_score']}  "
              f"${r['price']:>6.2f}  {r['rsi']:>5.0f}  "
              f"{r['pct_from_ma60']:>+5.1f}%  {cat_icon} {active_short}")

    print(f"\n  ℹ  Breakout-Wahrscheinlichkeit = Score / Max-Score × 100")
    print(f"  ℹ  {sum(WEIGHTS.values())} Punkte maximal · {len(WEIGHTS)} Indikatoren\n")


if __name__ == "__main__":
    main()
