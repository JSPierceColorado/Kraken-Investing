"""
Kraken Crypto Data Bot (Railway-ready, single-file)

What it does
------------
1) Discover tradable Kraken crypto base assets with a {BASE}{QUOTE} pair (default quote: USD).
2) Clear & write the tickers to Google Sheet: Active-Investing → Crypto-Scrape.
3) For each ticker, fetch Stocktwits messages, compute VADER sentiment:
   - If no messages available, skip it (do not write to sentiment tab).
4) For tickers with sentiment, fetch Kraken OHLC and compute RSI(14), EMA(20), SMA(50), momentum flag.
5) Write results to Active-Investing → Crypto-Sentiment.

Environment Variables
---------------------
Required:
- GOOGLE_CREDS_JSON     (full JSON for a Google Service Account with edit access to your Sheet)
- GOOGLE_SHEET_NAME     (default: Active-Investing)
- SHEET_TICKERS_TAB     (default: Crypto-Scrape)
- SHEET_SENTIMENT_TAB   (default: Crypto-Sentiment)

Optional:
- KRAKEN_BASE_PAIR=USD
- SENTIMENT_SOURCE=stocktwits
- MAX_TICKERS=50
- HTTP_TIMEOUT=20

Notes
-----
- Share your Google Sheet with the service account email found in the GOOGLE_CREDS_JSON.
- Add a Railway Cron to schedule runs.
"""

from __future__ import annotations

import os
import json
import time
import math
from typing import Dict, List, Optional, Set

import requests
import pandas as pd
import numpy as np
import pandas_ta as ta

import gspread
from google.oauth2.service_account import Credentials

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


# --------------------------
# Config & helpers
# --------------------------
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "50"))
DEFAULT_SHEET = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
TICKERS_TAB = os.getenv("SHEET_TICKERS_TAB", "Crypto-Scrape")
SENTIMENT_TAB = os.getenv("SHEET_SENTIMENT_TAB", "Crypto-Sentiment")
BASE_PAIR = os.getenv("KRAKEN_BASE_PAIR", "USD").upper()
SENTIMENT_SOURCE = os.getenv("SENTIMENT_SOURCE", "stocktwits").lower()


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------
# Google Sheets client
# --------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class SheetClient:
    def __init__(self, sheet_name: str):
        creds_json = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json:
            raise RuntimeError("GOOGLE_CREDS_JSON is missing.")
        try:
            sa_info = json.loads(creds_json)
        except Exception as e:
            raise RuntimeError("GOOGLE_CREDS_JSON must be valid JSON.") from e

        creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open(sheet_name)

    def clear_and_write(self, tab_name: str, rows: List[List]):
        try:
            ws = self.sheet.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            ws = self.sheet.add_worksheet(title=tab_name, rows="1000", cols="26")
        ws.clear()
        if rows:
            ws.update("A1", rows, value_input_option="RAW")


# --------------------------
# Kraken API
# --------------------------
class KrakenClient:
    BASE = "https://api.kraken.com/0/public"

    def _get(self, path: str, params: Optional[dict] = None):
        url = f"{self.BASE}/{path}"
        r = requests.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            # Kraken returns list of error codes
            raise RuntimeError(f"Kraken error: {data['error']}")
        return data["result"]

    def list_base_assets_with_quote(self, base_pair: str = "USD") -> Set[str]:
        """Return normalized base symbols that have a pair with the given quote (e.g., USD)."""
        pairs = self._get("AssetPairs")  # mapping of pairname -> details
        result: Set[str] = set()
        quote = base_pair.upper()

        for _, info in pairs.items():
            wsname = info.get("wsname")  # e.g., "XBT/USD"
            if not wsname or "/" not in wsname:
                continue
            base, q = wsname.split("/")
            if q.upper() != quote:
                continue
            result.add(normalize_base_symbol(base))
        return result

    def get_ohlc_for_symbol(
        self, base_symbol: str, quote: str = "USD", interval: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC for base/quote (e.g., BTC/USD). Interval in minutes (1, 5, 15, 60, 240, 1440, ...).
        Returns DataFrame indexed by datetime with columns: open, high, low, close, vwap, volume, count.
        """
        kraken_base = denormalize_to_kraken_base(base_symbol)
        pair = f"{kraken_base}{quote.upper()}"  # Kraken accepts pair without slash e.g., XBTUSD
        try:
            data = self._get("OHLC", params={"pair": pair, "interval": interval})
        except Exception:
            return None
        if not data:
            return None
        # result dict key is canonical pair code; pull first key
        key = next(iter(data.keys()))
        rows = data[key]
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).tz_convert(None)
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "vwap", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


# Kraken symbol normalization (common mappings)
NORMALIZE_MAP = {"XBT": "BTC", "XDG": "DOGE"}
DENORMALIZE_MAP = {"BTC": "XBT", "DOGE": "XDG"}


def normalize_base_symbol(base: str) -> str:
    b = base.upper()
    return NORMALIZE_MAP.get(b, b)


def denormalize_to_kraken_base(sym: str) -> str:
    s = sym.upper()
    return DENORMALIZE_MAP.get(s, s)


# --------------------------
# Sentiment
# --------------------------
class SentimentClient:
    def __init__(self, source: str = "stocktwits"):
        self.source = (source or "stocktwits").lower()
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_messages_for_symbol(self, symbol: str, limit: int = 30) -> List[str]:
        if self.source == "stocktwits":
            return self._stocktwits_messages(symbol, limit=limit)
        return []

    def _stocktwits_messages(self, symbol: str, limit: int = 30) -> List[str]:
        msgs: List[str] = []
        for sym in (f"{symbol}.X", symbol):  # many crypto streams use BTC.X
            try:
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
                r = requests.get(url, timeout=HTTP_TIMEOUT)
                if r.status_code != 200:
                    continue
                data = r.json()
                for m in data.get("messages", [])[:limit]:
                    body = (m.get("body") or "").strip()
                    if body:
                        msgs.append(body)
                if msgs:
                    break
            except Exception:
                continue
        return msgs

    def aggregate_vader(self, messages: List[str]) -> Dict[str, float]:
        if not messages:
            return {"count": 0, "compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
        comps, poss, negs, neus = [], [], [], []
        for t in messages:
            s = self.analyzer.polarity_scores(t)
            comps.append(s["compound"])
            poss.append(s["pos"])
            negs.append(s["neg"])
            neus.append(s["neu"])
        n = len(messages)
        return {
            "count": n,
            "compound": round(sum(comps) / n, 4),
            "pos": round(sum(poss) / n, 4),
            "neg": round(sum(negs) / n, 4),
            "neu": round(sum(neus) / n, 4),
        }


# --------------------------
# Technical indicators
# --------------------------
def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Given OHLC dataframe, compute:
    - rsi14 (float)
    - ema20 (float)
    - sma50 (float)
    - momentum_up (bool): close_now > close_20 ago
    """
    out: Dict[str, Optional[float | bool]] = {"rsi14": None, "ema20": None, "sma50": None, "momentum_up": None}
    if df is None or df.empty:
        return out

    close = df["close"].copy()

    # RSI 14
    rsi = ta.rsi(close, length=14).dropna()
    out["rsi14"] = round(float(rsi.iloc[-1]), 2) if not rsi.empty else None

    # EMA 20
    ema20 = ta.ema(close, length=20).dropna()
    out["ema20"] = round(float(ema20.iloc[-1]), 6) if not ema20.empty else None

    # SMA 50
    sma50 = ta.sma(close, length=50).dropna()
    out["sma50"] = round(float(sma50.iloc[-1]), 6) if not sma50.empty else None

    # Simple momentum: last close vs 20 periods ago
    if len(close) >= 21:
        out["momentum_up"] = bool(close.iloc[-1] > close.iloc[-21])

    return out


# --------------------------
# Main orchestration
# --------------------------
def main():
    sheet_name = DEFAULT_SHEET
    tickers_tab = TICKERS_TAB
    sentiment_tab = SENTIMENT_TAB
    base_pair = BASE_PAIR

    sheets = SheetClient(sheet_name)
    kraken = KrakenClient()
    senti = SentimentClient(source=SENTIMENT_SOURCE)

    # 1) Discover symbols with the desired quote (e.g., USD)
    log(f"Fetching Kraken assets with {base_pair} pairs…")
    symbols = sorted(list(kraken.list_base_assets_with_quote(base_pair=base_pair)))
    if not symbols:
        log("No symbols discovered. Exiting.")
        return
    if MAX_TICKERS and len(symbols) > MAX_TICKERS:
        symbols = symbols[:MAX_TICKERS]

    log(f"Found {len(symbols)} symbols. Example: {symbols[:10]}")

    # 2) Write tickers to Crypto-Scrape
    rows = [["symbol"]] + [[s] for s in symbols]
    sheets.clear_and_write(tickers_tab, rows)
    log(f"Wrote {len(symbols)} symbols to '{tickers_tab}'.")

    # 3) Build sentiment + indicators rows
    out_rows: List[List] = [[
        "symbol", "messages", "avg_compound", "avg_pos", "avg_neg", "avg_neu",
        "rsi14", "ema20", "sma50", "momentum_up"
    ]]

    for sym in symbols:
        msgs = senti.fetch_messages_for_symbol(sym)
        if not msgs:
            log(f"[{sym}] No Stocktwits messages → skipping sentiment row.")
            continue

        agg = senti.aggregate_vader(msgs)

        # 4) Technicals via Kraken OHLC (60-minute candles)
        ohlc = kraken.get_ohlc_for_symbol(sym, quote=base_pair, interval=60)
        if ohlc is None or ohlc.empty:
            log(f"[{sym}] No OHLC data; technicals will be blank.")
            ind = {"rsi14": None, "ema20": None, "sma50": None, "momentum_up": None}
        else:
            ind = compute_indicators(ohlc)

        out_rows.append([
            sym, agg["count"], agg["compound"], agg["pos"], agg["neg"], agg["neu"],
            ind["rsi14"], ind["ema20"], ind["sma50"], ind["momentum_up"]
        ])

    # 5) Write sentiment table (only those with messages)
    sheets.clear_and_write(sentiment_tab, out_rows)
    log(f"Wrote {len(out_rows)-1} rows to '{sentiment_tab}'. Done.")


if __name__ == "__main__":
    main()
