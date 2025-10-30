"""
STOCKTWITS-FIRST PIPELINE (resilient discovery, no preloaded tickers)

Flow
----
1) Discover crypto symbols from Stocktwits:
   - Seed with /trending/symbols.json
   - Expand via throttled /search/symbols.json sweeps over A..Z and 0..9
   - Keep only crypto-like results (type contains 'crypto' or symbol ends with '.X')
2) Write discovery inventory to Google Sheet: Active-Investing → Crypto-Scrape
3) For each discovered symbol (slug):
   - Fetch messages from Stocktwits (search-first to confirm slug → stream)
   - If messages > 0, compute VADER averages
   - Map slug to Kraken base (strip '.X'; XBT/DOGE normalization) and fetch OHLC(60m)
   - Compute RSI(14), EMA(20), SMA(50), simple momentum
   - If any step fails → EXCLUDE the symbol
4) Write only fully successful rows to Active-Investing → Crypto-Sentiment

Env
---
GOOGLE_CREDS_JSON
GOOGLE_SHEET_NAME   (default "Active-Investing")
SHEET_TICKERS_TAB   (default "Crypto-Scrape")
SHEET_SENTIMENT_TAB (default "Crypto-Sentiment")
KRAKEN_BASE_PAIR    (default "USD")
HTTP_TIMEOUT        (default 20)
MAX_STW_SYMBOLS     (default 600)   # cap for discovery list to avoid rate limits
MESSAGES_PER_SYMBOL (default 30)
BASE_SLEEP_SEC      (default 0.35)  # base pacing between calls
MAX_RETRIES         (default 3)     # retries for 429/5xx
MAX_TICKERS         (optional, if you want to hard-cap post-discovery processing)
"""

from __future__ import annotations

import os, json, re, time, math, logging
from typing import Dict, List, Optional, Set

import requests
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# --- Sentiment (VADER) ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

# --------------------------
# Config
# --------------------------
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))
DEFAULT_SHEET = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
TICKERS_TAB = os.getenv("SHEET_TICKERS_TAB", "Crypto-Scrape")
SENTIMENT_TAB = os.getenv("SHEET_SENTIMENT_TAB", "Crypto-Sentiment")
BASE_PAIR = os.getenv("KRAKEN_BASE_PAIR", "USD").upper()

MAX_STW_SYMBOLS = int(os.getenv("MAX_STW_SYMBOLS", "600"))
MESSAGES_PER_SYMBOL = int(os.getenv("MESSAGES_PER_SYMBOL", "30"))
BASE_SLEEP_SEC = float(os.getenv("BASE_SLEEP_SEC", "0.35"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "0"))  # 0 = no cap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
def log(msg: str): print(msg, flush=True)

# --------------------------
# Google Sheets
# --------------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

class SheetClient:
    def __init__(self, sheet_name: str):
        creds_json = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json: raise RuntimeError("GOOGLE_CREDS_JSON is missing.")
        sa_info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open(sheet_name)

    def clear_and_write(self, tab: str, rows: List[List]):
        try:
            ws = self.sheet.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = self.sheet.add_worksheet(title=tab, rows="5000", cols="26")
        ws.clear()
        if rows:
            ws.update(rows, "A1", value_input_option="RAW")  # values first (no deprecation)

# --------------------------
# Kraken API
# --------------------------
class KrakenClient:
    BASE = "https://api.kraken.com/0/public"

    def _get(self, path: str, params: Optional[dict] = None):
        r = requests.get(f"{self.BASE}/{path}", params=params or {}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("error"): raise RuntimeError(f"Kraken error: {data['error']}")
        return data["result"]

    def get_ohlc_for_symbol(self, base_symbol: str, quote: str = "USD", interval: int = 60) -> Optional[pd.DataFrame]:
        # Map common aliases
        kraken_base = {"BTC":"XBT","DOGE":"XDG"}.get(base_symbol.upper(), base_symbol.upper())
        pair = f"{kraken_base}{quote.upper()}"
        try:
            data = self._get("OHLC", params={"pair": pair, "interval": interval})
            if not data: return None
            keys = [k for k in data.keys() if k != "last"]
            if not keys: return None
            rows = data.get(keys[0], [])
            if not rows: return None
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
            df.set_index("time", inplace=True)
            for c in ["open","high","low","close","vwap","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            logging.info("Kraken OHLC fail for %s/%s: %s", base_symbol, quote, e)
            return None

# --------------------------
# Stocktwits (resilient discovery + fetch)
# --------------------------
SYMBOL_RE = re.compile(r"(?=.*[A-Z])[A-Z0-9]{2,10}$")
FIAT = {"USD","EUR","AUD","GBP","JPY","CAD","CHF","CNY","CNH","NZD","SEK","NOK","DKK","HKD","SGD","ZAR","MXN"}

def looks_crypto(sym_obj: dict) -> bool:
    sym = (sym_obj.get("symbol") or "").upper()
    typ = (sym_obj.get("type") or "").lower()
    if sym.endswith(".X"):             # common Stocktwits crypto suffix
        return True
    if "crypto" in typ:                # e.g., "cryptocurrency"
        return True
    return False

class StocktwitsClient:
    BASE = "https://api.stocktwits.com/api/2"
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0; +https://example.com)",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: Optional[dict]=None, *, allow_retry: bool=True) -> Optional[dict]:
        """GET with retry/backoff on 429/5xx."""
        url = f"{self.BASE}/{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = self.s.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
                code = r.status_code
                if code == 200:
                    return r.json()
                # retry on 429 and 5xx
                if allow_retry and (code == 429 or 500 <= code < 600):
                    sleep_s = BASE_SLEEP_SEC * (2 ** (attempt - 1))
                    logging.info("Stocktwits %s -> %s; retrying in %.2fs (attempt %d/%d)",
                                 path, code, sleep_s, attempt, MAX_RETRIES)
                    time.sleep(sleep_s)
                    continue
                logging.info("Stocktwits %s -> HTTP %s (no retry)", path, code)
                return None
            except Exception as e:
                if allow_retry and attempt < MAX_RETRIES:
                    sleep_s = BASE_SLEEP_SEC * (2 ** (attempt - 1))
                    logging.info("Stocktwits %s error %s; retrying in %.2fs (attempt %d/%d)",
                                 path, e, sleep_s, attempt, MAX_RETRIES)
                    time.sleep(sleep_s)
                    continue
                logging.info("Stocktwits %s error (final): %s", path, e)
                return None
        return None

    def trending_symbols(self) -> List[str]:
        data = self._get("trending/symbols.json")
        out: List[str] = []
        if not data: return out
        for s in data.get("symbols", []) or []:
            if looks_crypto(s):
                sym = (s.get("symbol") or "").upper()
                if SYMBOL_RE.fullmatch(sym) and sym not in FIAT:
                    out.append(sym)
        time.sleep(BASE_SLEEP_SEC)
        return out

    def search_symbols(self, q: str) -> List[str]:
        data = self._get("search/symbols.json", {"q": q})
        out: List[str] = []
        if not data: return out
        for s in data.get("symbols", []) or []:
            if looks_crypto(s):
                sym = (s.get("symbol") or "").upper()
                if SYMBOL_RE.fullmatch(sym) and sym not in FIAT:
                    out.append(sym)
        time.sleep(BASE_SLEEP_SEC)
        return out

    def discover_crypto_symbols(self, cap: int = 600) -> List[str]:
        """Trending seed + throttled alphanumeric search sweep until cap."""
        seen: Set[str] = set()
        out: List[str] = []

        # 1) seed with trending
        for sym in self.trending_symbols():
            if sym not in seen:
                seen.add(sym); out.append(sym)
                if len(out) >= cap: return out

        # 2) search sweep A..Z and 0..9 (throttled)
        queries = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for q in queries:
            for sym in self.search_symbols(q):
                if sym not in seen:
                    seen.add(sym); out.append(sym)
                    if len(out) >= cap: return out
        return out

    # per-symbol: confirm slug via search, then pull stream
    def find_slug_for_symbol(self, raw: str) -> Optional[str]:
        # Many crypto slugs on STW end with ".X"
        # Search raw, then try raw+".X" if exact isn't returned by search
        data = self._get("search/symbols.json", {"q": raw})
        if data:
            q_up = raw.upper()
            cands = [(s.get("symbol") or "").upper() for s in data.get("symbols", []) or [] if looks_crypto(s)]
            if not cands:
                return None
            # Priority: exact match with .X, then exact, then contains, else first
            if f"{q_up}.X" in cands: return f"{q_up}.X"
            if q_up in cands: return q_up
            for c in cands:
                if q_up in c: return c
            return cands[0]
        return None

    def fetch_messages(self, slug: str, limit: int = 30) -> List[str]:
        from urllib.parse import quote
        data = self._get(f"streams/symbol/{quote(slug)}.json")
        msgs: List[str] = []
        if not data: return msgs
        for m in data.get("messages", [])[:limit]:
            body = (m.get("body") or "").strip()
            if body: msgs.append(body)
        time.sleep(BASE_SLEEP_SEC)
        return msgs

# --------------------------
# TA (pure pandas)
# --------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()
def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame) -> dict:
    out = {"rsi14": None, "ema20": None, "sma50": None, "momentum_up": None}
    if df is None or df.empty: return out
    close = df["close"].astype(float)
    rsi14 = rsi_wilder(close, 14).dropna()
    ema20 = ema(close, 20).dropna()
    sma50 = sma(close, 50).dropna()
    out["rsi14"] = round(float(rsi14.iloc[-1]), 2) if not rsi14.empty else None
    out["ema20"] = round(float(ema20.iloc[-1]), 6) if not ema20.empty else None
    out["sma50"] = round(float(sma50.iloc[-1]), 6) if not sma50.empty else None
    out["momentum_up"] = bool(close.iloc[-1] > close.iloc[-21]) if len(close) >= 21 else None
    return out

# --------------------------
# Helpers
# --------------------------
def slug_to_base(slug: str) -> str:
    base = slug[:-2] if slug.endswith(".X") else slug
    return {"XBT":"BTC","XDG":"DOGE"}.get(base.upper(), base.upper())

# --------------------------
# Main
# --------------------------
def main():
    sheets = SheetClient(DEFAULT_SHEET)
    stw = StocktwitsClient()
    kraken = KrakenClient()
    vader = SentimentIntensityAnalyzer()

    # 1) Discover Stocktwits crypto slugs
    log("Discovering crypto symbols from Stocktwits (trending + throttled search)…")
    slugs = stw.discover_crypto_symbols(cap=MAX_STW_SYMBOLS)
    if not slugs:
        log("No Stocktwits crypto symbols discovered. Exiting.")
        return

    # Optional hard-cap for processing volume
    if MAX_TICKERS and len(slugs) > MAX_TICKERS:
        slugs = slugs[:MAX_TICKERS]

    log(f"Discovered {len(slugs)} slugs. Example: {slugs[:12]}")

    # Write discovery inventory to Crypto-Scrape
    sheets.clear_and_write(TICKERS_TAB, [["stw_symbol"]] + [[s] for s in sorted(slugs)])
    log(f"Wrote {len(slugs)} Stocktwits crypto symbols to '{TICKERS_TAB}'.")

    # 2) For each slug: messages -> sentiment -> Kraken -> TA
    results: List[List] = [[
        "symbol", "messages", "avg_compound", "avg_pos", "avg_neg", "avg_neu",
        "rsi14", "ema20", "sma50", "momentum_up"
    ]]

    for slug in slugs:
        # Messages
        messages = stw.fetch_messages(slug, limit=MESSAGES_PER_SYMBOL)
        if not messages:
            logging.info("[%s] No Stocktwits messages -> skip", slug)
            continue

        # Sentiment
        comps=poss=negs=neus=0.0
        for t in messages:
            s = vader.polarity_scores(t)
            comps += s["compound"]; poss += s["pos"]; negs += s["neg"]; neus += s["neu"]
        n = float(len(messages))
        agg = {
            "count": int(n),
            "compound": round(comps/n, 4),
            "pos": round(poss/n, 4),
            "neg": round(negs/n, 4),
            "neu": round(neus/n, 4),
        }

        # Kraken OHLC (map slug -> base)
        base = slug_to_base(slug)
        if base in FIAT:
            logging.info("[%s] Looks fiat -> skip", slug)
            continue

        ohlc = kraken.get_ohlc_for_symbol(base, quote=BASE_PAIR, interval=60)
        if ohlc is None or ohlc.empty:
            logging.info("[%s] No Kraken OHLC -> skip", slug)
            continue

        ind = compute_indicators(ohlc)
        results.append([
            base, agg["count"], agg["compound"], agg["pos"], agg["neg"], agg["neu"],
            ind["rsi14"], ind["ema20"], ind["sma50"], ind["momentum_up"]
        ])

    # 3) Write only successful rows
    sheets.clear_and_write(SENTIMENT_TAB, results)
    log(f"Wrote {len(results)-1} rows to '{SENTIMENT_TAB}'. Done.")

if __name__ == "__main__":
    main()
