"""
Pipeline:
  Stocktwits universe (search) -> messages -> VADER sentiment
  -> Kraken OHLC -> TA (RSI14, EMA20, SMA50, momentum_up)
  -> Write only fully-successful rows to Crypto-Sentiment

Env:
  GOOGLE_CREDS_JSON  (service account JSON)
  GOOGLE_SHEET_NAME  (default Active-Investing)
  SHEET_TICKERS_TAB  (default Crypto-Scrape)       # will list discovered Stocktwits crypto symbols (slugs)
  SHEET_SENTIMENT_TAB(default Crypto-Sentiment)
  KRAKEN_BASE_PAIR   (default USD)
  HTTP_TIMEOUT       (default 20)
  MAX_STW_SYMBOLS    (optional cap for discovered STW symbols, default 500)
  MESSAGES_PER_SYMBOL(default 30)
  RATE_SLEEP_SEC     (default 0.35)
"""

from __future__ import annotations

import os, json, re, time, logging
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

MAX_STW_SYMBOLS = int(os.getenv("MAX_STW_SYMBOLS", "500"))
MESSAGES_PER_SYMBOL = int(os.getenv("MESSAGES_PER_SYMBOL", "30"))
RATE_SLEEP_SEC = float(os.getenv("RATE_SLEEP_SEC", "0.35"))

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
            ws = self.sheet.add_worksheet(title=tab, rows="2000", cols="26")
        ws.clear()
        if rows:
            ws.update(rows, "A1", value_input_option="RAW")  # values first (fix deprecation)

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
        # Kraken uses e.g. XBT for BTC, XDG for DOGE
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
# Stocktwits client
# --------------------------
SYMBOL_RE = re.compile(r"(?=.*[A-Z])[A-Z0-9]{2,10}$")
FIAT = {"USD","EUR","AUD","GBP","JPY","CAD","CHF","CNY","CNH","NZD","SEK","NOK","DKK","HKD","SGD","ZAR","MXN"}

class StocktwitsClient:
    BASE = "https://api.stocktwits.com/api/2"
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0; +https://example.com)",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: Optional[dict]=None) -> Optional[dict]:
        try:
            r = self.s.get(f"{self.BASE}/{path}", params=params or {}, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                logging.info("Stocktwits GET %s -> %s", path, r.status_code)
                return None
            return r.json()
        except Exception as e:
            logging.info("Stocktwits GET error %s: %s", path, e)
            return None

    def discover_crypto_symbols(self, cap: int = 500) -> List[str]:
        """
        Heuristic discovery: sweep search queries to collect crypto symbols.
        No hardcoded tickers; uses alphanumeric sweeps.
        """
        queries = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        seen: Set[str] = set()
        out: List[str] = []
        for q in queries:
            data = self._get("search/symbols.json", {"q": q})
            time.sleep(RATE_SLEEP_SEC)
            if not data: continue
            for s in data.get("symbols", []) or []:
                sym = (s.get("symbol") or "").upper()
                # Prefer only crypto-like results; Stocktwits 'type' varies (e.g., "cryptocurrency", "crypto")
                typ = (s.get("type") or "").lower()
                if typ and "crypto" not in typ:
                    continue
                if not SYMBOL_RE.fullmatch(sym):  # basic hygiene
                    continue
                if sym in FIAT:  # skip fiat codes
                    continue
                if sym not in seen:
                    seen.add(sym)
                    out.append(sym)
                    if len(out) >= cap:
                        return out
        return out

    def fetch_messages(self, slug: str, limit: int = 30) -> List[str]:
        data = self._get(f"streams/symbol/{slug}.json")
        if not data: return []
        msgs = []
        for m in data.get("messages", [])[:limit]:
            t = (m.get("body") or "").strip()
            if t: msgs.append(t)
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
# Main
# --------------------------
def main():
    sheets = SheetClient(DEFAULT_SHEET)
    stw = StocktwitsClient()
    kraken = KrakenClient()
    vader = SentimentIntensityAnalyzer()

    # 1) Discover Stocktwits crypto symbols (slugs)
    log("Discovering crypto symbols from Stocktwits…")
    stw_syms = stw.discover_crypto_symbols(cap=MAX_STW_SYMBOLS)
    if not stw_syms:
        log("No Stocktwits crypto symbols discovered. Exiting.")
        return
    log(f"Discovered {len(stw_syms)} symbols. Example: {stw_syms[:12]}")

    # Write discovery inventory to Crypto-Scrape
    sheets.clear_and_write(TICKERS_TAB, [["stw_symbol"]] + [[s] for s in sorted(stw_syms)])
    log(f"Wrote {len(stw_syms)} Stocktwits crypto symbols to '{TICKERS_TAB}'.")

    # 2) For each symbol: fetch messages & sentiment
    results: List[List] = [[
        "symbol", "messages", "avg_compound", "avg_pos", "avg_neg", "avg_neu",
        "rsi14", "ema20", "sma50", "momentum_up"
    ]]

    for slug in stw_syms:
        # messages
        msgs = stw.fetch_messages(slug, limit=MESSAGES_PER_SYMBOL)
        time.sleep(RATE_SLEEP_SEC)
        if not msgs:
            logging.info("[%s] No Stocktwits messages -> skip", slug)
            continue

        # sentiment
        comps=poss=negs=neus=0.0
        for t in msgs:
            s = vader.polarity_scores(t)
            comps += s["compound"]; poss += s["pos"]; negs += s["neg"]; neus += s["neu"]
        n = float(len(msgs))
        agg = {
            "count": int(n),
            "compound": round(comps/n, 4),
            "pos": round(poss/n, 4),
            "neg": round(negs/n, 4),
            "neu": round(neus/n, 4),
        }

        # Map to Kraken base symbol: strip trailing ".X" if present
        base = slug[:-2] if slug.endswith(".X") else slug
        # Skip obvious non-crypto fiat leftovers
        if base.upper() in FIAT: 
            logging.info("[%s] Looks fiat -> skip", slug)
            continue

        # 3) Kraken OHLC + TA
        ohlc = kraken.get_ohlc_for_symbol(base, quote=BASE_PAIR, interval=60)
        if ohlc is None or ohlc.empty:
            logging.info("[%s] No Kraken OHLC -> skip", slug)
            continue

        ind = compute_indicators(ohlc)
        results.append([
            base, agg["count"], agg["compound"], agg["pos"], agg["neg"], agg["neu"],
            ind["rsi14"], ind["ema20"], ind["sma50"], ind["momentum_up"]
        ])

    # 4) Write only fully successful rows
    sheets.clear_and_write(SENTIMENT_TAB, results)
    log(f"Wrote {len(results)-1} rows to '{SENTIMENT_TAB}'. Done.")

if __name__ == "__main__":
    main()
