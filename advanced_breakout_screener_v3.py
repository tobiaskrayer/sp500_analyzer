"""
╔══════════════════════════════════════════════════════════════════╗
║   RUSSELL 2000 · ADVANCED BREAKOUT SCREENER  v2.1               ║
║   Fokus: Nur bestätigte, aktive Ausbrüche ohne Failed Breakouts ║
╠══════════════════════════════════════════════════════════════════╣
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
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════

LOOKBACK_DAYS  = 300
MA_PERIOD      = 60
MAX_TICKERS    = None   # None = alle
BATCH_SIZE     = 40
SLEEP_BETWEEN  = 1.2
OUTPUT_CSV     = "breakout_results.csv"
OUTPUT_HTML    = "breakout_report.html"
TOP_N_CONSOLE  = 15

# ──────────────────────────────────────────────
# VORFILTER-SCHWELLWERTE
# ──────────────────────────────────────────────
MIN_PRICE               = 1.0    # Mindestpreis (Penny Stocks raus)
MIN_AVG_VOLUME          = 50_000 # Mindest-Durchschnittsvolumen
MAX_DIST_BELOW_MA60     = -0.15  # Max. 15% unter MA60 → sonst ignorieren
MIN_PRICE_TREND_20D     = -0.03  # Preis muss in 20T mind. > -3% sein
MIN_PRICE_TREND_5D      = -0.02  # NEU: Preis muss in 5T > -2% sein (kurzfr. Momentum)
FAILED_BREAKOUT_THRESH  = -0.03  # NEU: Cross vor ≤5T aber seitdem > 3% gefallen → raus

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
    print("\n📋 Lade Russell 2000 / Small Cap Ticker-Liste...")
    tickers = _try_ishares() or _try_finviz() or _fallback_list()
    print(f"   ✓ {len(tickers)} Ticker geladen")
    return tickers


def _try_ishares() -> list:
    url = ("https://www.ishares.com/us/products/239710/ISHARES-RUSSELL-2000-ETF/"
           "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        lines = resp.text.split('\n')
        start = next((i for i, l in enumerate(lines)
                      if l.startswith('Ticker') or l.startswith('"Ticker"')), None)
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
    from bs4 import BeautifulSoup
    tickers = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for page in range(1, 51):
        try:
            url = (f"https://finviz.com/screener.ashx?v=111"
                   f"&f=cap_small,exch_nasd|nyse&r={((page-1)*20)+1}")
            soup = BeautifulSoup(
                requests.get(url, headers=headers, timeout=10).text, 'html.parser')
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
                    if len(df) >= 80:
                        data[t] = df
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN)
    print(f"   ✓ {len(data)} Ticker mit ausreichend Daten")
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
    close = df["Close"]
    vol   = df.get("Volume", pd.Series(dtype=float))
    n     = len(close)

    today_c = float(close.iloc[-1])

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
        df    = df.sort_index()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df.get("Volume", pd.Series(dtype=float))
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
        base_max   = sum(WEIGHTS.values())
        all_combos = sum(b for _, b, _ in COMBO_BONUSES)
        total_max  = base_max + all_combos
        # score_100 ist der primäre Wert – direkt 0–100, eine Nachkommastelle
        score_100  = round(min(score / total_max * 100, 100.0), 1)

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
            "sparkline":      [round(float(c), 2) for c in close.iloc[-30:]],
            "details":        details,
        }

    except Exception:
        return None


def analyze_all(price_data: dict) -> pd.DataFrame:
    print(f"\n🔬 Analysiere {len(price_data)} Ticker (mit Vorfilter)...")
    results = [
        r for r in (analyze(t, df) for t, df in tqdm(price_data.items(), desc="Analysiere"))
        if r
    ]
    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"   ✓ {len(df)} Aktien nach Vorfilter ausgewertet")
    return df


# ══════════════════════════════════════════════
# SCHRITT 4: HTML-REPORT
# ══════════════════════════════════════════════

def build_html(df: pd.DataFrame) -> str:
    top     = df.head(150)

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

    def spark(prices):
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
          <td style="padding:10px 14px;font-weight:900;font-size:14px;color:#fff;font-family:'Courier New',monospace">{row['ticker']}</td>
          <td style="padding:10px 14px">{badge(row['signal'])}</td>
          <td style="padding:10px 14px">{prob_bar(row['score'])}</td>
          <td style="padding:10px 14px;color:#fff;font-weight:700">${row['price']:.2f}</td>
          <td style="padding:10px 14px;color:#666;font-size:11px">${row['ma60']:.2f}</td>
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
        f'{name}</span><span style="font-size:11px;color:#555">{desc}</span></div>'
        for name, desc, c in legend_items
    )

    combo_legend = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="background:rgba(255,215,0,0.1);border:1px solid gold;color:gold;'
        f'padding:2px 8px;border-radius:3px;font-size:10px;min-width:140px;text-align:center;font-weight:700">'
        f'{label}</span><span style="font-size:11px;color:#555">+{bonus} Pkt</span></div>'
        for keys, bonus, label in COMBO_BONUSES
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout Screener v2.1 · Russell 2000</title>
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
  th{{padding:12px 14px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;
      color:#333;cursor:pointer;white-space:nowrap;user-select:none}}
  th:hover{{color:#888}}
  tbody tr{{border-bottom:1px solid rgba(26,26,40,.6);transition:background .1s}}
  tbody tr:hover{{background:#0f0f18}}
  tbody tr:last-child{{border-bottom:none}}
  .legend-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px}}
  .legend{{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:24px}}
  .legend h3{{font-family:'Syne',sans-serif;font-size:12px;letter-spacing:2px;color:#444;
              text-transform:uppercase;margin-bottom:16px}}
  .legend-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:4px}}
  .footer{{margin-top:24px;font-size:10px;color:#222;letter-spacing:1px;text-align:center}}
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
  .data-row{{animation:fadeUp .3s ease both}}
  @media(max-width:768px){{.legend-wrap{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="tag">Russell 2000 · NYSE/NASDAQ · Advanced Screener</div>
    <h1>Breakout <em>Intelligence</em> <span class="version">v2.1</span></h1>
    <div class="meta">// {analyzed} AKTIEN NACH VORFILTER · 10+ INDIKATOREN · KOMBINATIONS-BONI · {gen_time}</div>
  </header>

  <div class="stats">
    <div class="stat"><div class="slabel">MA60 Breakouts</div><div class="sval green">{crossovers}</div></div>
    <div class="stat"><div class="slabel">Sehr stark ≥60</div><div class="sval orange">{sehr_stark}</div></div>
    <div class="stat"><div class="slabel">Stark ≥45</div><div class="sval blue">{stark}</div></div>
    <div class="stat"><div class="slabel">Score-Skala</div><div class="sval white">0–100</div></div>
    <div class="stat"><div class="slabel">Analysiert</div><div class="sval white">{analyzed}</div></div>
  </div>

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
        <th>CHART</th>
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
      <div style="margin-top:16px;font-size:10px;color:#333;line-height:1.8">
        Kombinations-Boni werden zusätzlich zum Basis-Score vergeben<br>
        wenn mehrere starke Signale gleichzeitig feuern.<br>
        Dies erhöht die Selektivität und filtert Fehlsignale heraus.
      </div>
    </div>
  </div>

  <div class="footer">
    // Russell 2000 Advanced Breakout Screener v2.1 · Daten: Yahoo Finance via yfinance<br>
    // Failed-Breakout-Filter aktiv · Kombinations-Boni · Nicht als Anlageberatung zu verstehen
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
    print("╔" + "═"*60 + "╗")
    print("║  ADVANCED BREAKOUT SCREENER v2.1 · RUSSELL 2000" + " "*11 + "║")
    print("║  Failed-Breakout-Filter + Breakout-Health-Scoring" + " "*9 + "║")
    print("╚" + "═"*60 + "╝")

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

    # HTML
    html = build_html(df_results)
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
