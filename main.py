"""
Kraken Tracker & Seller (Perpetual loop, trailing take profit)

Sheet (tab: Kraken Integration)
-------------------------------
A: Symbol
B: P/L %
C: Quantity
D: Avg Cost (quote)
E: Last Price (quote)
F: Armed (TRUE/FALSE)
G: High P/L % (since armed)
H: Status (e.g., HOLD / ARMED / SOLD / STOP_LOSS)
I: Last Update (UTC ISO)
J: Notes (last action / errors)

Trading rules
-------------
- Stop loss: if P/L% <= LOSS_THRESHOLD (default -3), SELL 100%
- Trailing take profit:
  - If P/L% >= ARM_THRESHOLD (default +5), set Armed=TRUE and High=P/L%
  - While Armed, update High on new highs; if P/L% <= High - TRAIL_GIVEBACK (default 2), SELL 100%

Env Vars
--------
KRAKEN_API_KEY, KRAKEN_API_SECRET
GOOGLE_CREDS_JSON
GOOGLE_SHEET_NAME      (default: Active-Investing)
SHEET_KRAKEN_TAB       (default: Kraken Integration)
QUOTE_CCY              (default: USD)
LOSS_THRESHOLD         (default: -3)      # percent
ARM_THRESHOLD          (default: 5)       # percent
TRAIL_GIVEBACK         (default: 2)       # percent
INTERVAL_SEC           (default: 300)     # sleep between loops
LIVE_TRADING           (default: false)   # true to place real orders
"""

from __future__ import annotations

import os
import json
import time
import hmac
import hashlib
import base64
import logging
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import krakenex

# ---------- Config ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
SHEET_KRAKEN_TAB = os.getenv("SHEET_KRAKEN_TAB", "Kraken Integration")
QUOTE_CCY = os.getenv("QUOTE_CCY", "USD").upper()

LOSS_THRESHOLD = float(os.getenv("LOSS_THRESHOLD", "-3"))
ARM_THRESHOLD = float(os.getenv("ARM_THRESHOLD", "5"))
TRAIL_GIVEBACK = float(os.getenv("TRAIL_GIVEBACK", "2"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "300"))
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

KRAKEN_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Map for Kraken base asset codes <-> common symbols
KRKN_BASE_TO_COMMON = {"XBT": "BTC", "XDG": "DOGE"}
COMMON_TO_KRKN_BASE = {"BTC": "XBT", "DOGE": "XDG"}

# ---------- Helpers ----------
def now_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

# ---------- Google Sheets ----------
class SheetClient:
    def __init__(self, sheet_name: str):
        creds_json = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json:
            raise RuntimeError("GOOGLE_CREDS_JSON missing")
        sa_info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open(sheet_name)

    def read_tab(self, tab: str) -> pd.DataFrame:
        try:
            ws = self.sheet.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = self.sheet.add_worksheet(title=tab, rows="2000", cols="20")
        data = ws.get_all_values()
        if not data:
            return pd.DataFrame()
        header, *rows = data
        if not header:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=header)
        return df

    def write_tab(self, tab: str, rows: List[List]):
        try:
            ws = self.sheet.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = self.sheet.add_worksheet(title=tab, rows="2000", cols="20")
        ws.clear()
        if rows:
            # values first, then range (deprecation-safe)
            ws.update(rows, "A1", value_input_option="RAW")

# ---------- Kraken ----------
class KrakenClient:
    PUBLIC = "https://api.kraken.com/0/public"
    PRIVATE = "https://api.kraken.com/0/private"

    def __init__(self, key: str, secret: str):
        self.api = krakenex.API(key, secret)
        self.session = requests.Session()

    # --------- Public ---------
    def get_asset_pairs(self) -> dict:
        r = self.session.get(f"{self.PUBLIC}/AssetPairs", timeout=20)
        r.raise_for_status()
        res = r.json()
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]

    def ticker(self, pairs: List[str]) -> dict:
        # pairs must be comma-separated altnames like XBTUSD
        params = {"pair": ",".join(pairs)}
        r = self.session.get(f"{self.PUBLIC}/Ticker", params=params, timeout=20)
        r.raise_for_status()
        res = r.json()
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]

    # --------- Private (via krakenex) ---------
    def balance(self) -> dict:
        res = self.api.query_private("Balance")
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]

    def trades_history(self) -> dict:
        # you can paginate if needed; this fetches recent history by default
        res = self.api.query_private("TradesHistory", {"trades": True})
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]["trades"]

    def add_order_market_sell(self, pair_altname: str, volume: float) -> dict:
        payload = {
            "pair": pair_altname,   # e.g., XBTUSD
            "type": "sell",
            "ordertype": "market",
            "volume": str(volume)
        }
        res = self.api.query_private("AddOrder", payload)
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res["result"]

# ---------- Pair mapping & pricing ----------
def build_pair_maps(asset_pairs: dict, quote_ccy: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      base_to_altpair: map common base symbol (BTC) -> pair altname (XBTUSD)
      altpair_to_base: reverse map altname -> common base symbol
    Only pairs quoted in `quote_ccy` are considered.
    """
    quote_ccy = quote_ccy.upper()
    base_to_altpair: Dict[str, str] = {}
    altpair_to_base: Dict[str, str] = {}
    for key, info in asset_pairs.items():
        altname = info.get("altname")      # e.g., XBTUSD
        wsname = info.get("wsname")        # e.g., XBT/USD
        if not altname or not wsname or "/" not in wsname:
            continue
        base, quote = wsname.split("/")
        if quote.upper() != quote_ccy:
            continue
        common = KRKN_BASE_TO_COMMON.get(base.upper(), base.upper())  # XBT->BTC, XDG->DOGE
        base_to_altpair[common] = altname
        altpair_to_base[altname] = common
    return base_to_altpair, altpair_to_base

def current_prices(kraken: KrakenClient, base_to_pair: Dict[str, str]) -> Dict[str, float]:
    if not base_to_pair:
        return {}
    pairs = list(base_to_pair.values())
    # Ticker allows batching many pairs
    result = {}
    # Kraken supports comma-separated list; batch to stay safe
    BATCH = 20
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i+BATCH]
        t = kraken.ticker(batch)
        for pair_alt, data in t.items():
            # 'c' last trade [price, lot]
            price = to_float(data.get("c", [None])[0])
            if price is not None:
                result[pair_alt] = price
    # Map back to base symbol
    out = {}
    for base, pair in base_to_pair.items():
        if pair in result:
            out[base] = result[pair]
    return out

# ---------- Cost basis from TradesHistory (spot USD) ----------
def compute_avg_cost_for_position(base: str, qty: float, trades: dict, base_to_pair: Dict[str, str]) -> Optional[float]:
    """
    Compute the weighted average cost of the CURRENT held quantity (qty) using FIFO against spot USD trades.
    Returns None if no trade data found.
    """
    pair = base_to_pair.get(base)
    if not pair:
        return None

    # Extract relevant trades in this pair
    rows = []
    for txid, tr in trades.items():
        if tr.get("pair") != pair:
            continue
        ttype = tr.get("type")           # buy/sell
        vol = to_float(tr.get("vol"))
        price = to_float(tr.get("price"))
        cost = to_float(tr.get("cost"))  # total cost (price*vol)
        fee  = to_float(tr.get("fee") or 0.0)
        time_ = float(tr.get("time", 0.0))
        if not vol or not price:
            continue
        rows.append((time_, ttype, vol, price, cost, fee))
    if not rows:
        return None

    # Sort by time ascending (FIFO)
    rows.sort(key=lambda r: r[0])

    # Build inventory: add buys, remove sells
    # After processing all, take the lots that make up the final qty and compute cost basis.
    lots: List[Tuple[float, float]] = []  # list of (vol, unit_cost)
    for _, ttype, vol, price, cost, fee in rows:
        unit_cost = (cost + fee) / vol if vol else price
        if ttype == "buy":
            lots.append((vol, unit_cost))
        elif ttype == "sell":
            # remove from lots FIFO
            remaining = vol
            new_lots = []
            for lvol, lcost in lots:
                if remaining <= 0:
                    new_lots.append((lvol, lcost))
                    continue
                take = min(lvol, remaining)
                lvol_after = lvol - take
                remaining -= take
                if lvol_after > 0:
                    new_lots.append((lvol_after, lcost))
            lots = new_lots

    # Now lots should represent current inventory (may be >, =, or < actual due to non-USD trades/withdrawals)
    if qty <= 0:
        return None

    # Take the last 'qty' from FIFO lots
    remaining = qty
    total_cost = 0.0
    total_vol = 0.0
    for lvol, lcost in lots:
        if remaining <= 0:
            break
        take = min(lvol, remaining)
        total_cost += take * lcost
        total_vol += take
        remaining -= take

    if total_vol <= 0:
        return None
    return total_cost / total_vol

# ---------- Core loop ----------
def run_once(kraken: KrakenClient, sheets: SheetClient):
    # 1) Fetch balances (spot)
    bal = kraken.balance()  # dict like {"XXBT": "0.5", "ZUSD": "123.45", ...}
    # Filter only crypto bases (exclude fiat balances like ZUSD)
    crypto_bal: Dict[str, float] = {}
    for asset_code, amount in bal.items():
        qty = to_float(amount)
        if not qty or qty <= 0:
            continue
        # asset_code examples: XXBT, XETH, ZUSD. We want base alt like XBT → BTC
        code = asset_code.replace("X", "").replace("Z", "")  # crude but usually OK (XXBT->XBT, ZUSD->USD)
        if code == QUOTE_CCY:
            continue
        common = KRKN_BASE_TO_COMMON.get(code, code)  # XBT->BTC
        crypto_bal[common] = crypto_bal.get(common, 0.0) + qty

    if not crypto_bal:
        logging.info("No crypto balances.")
        # Still write a header so sheet is clean
        header = [["Symbol","P/L %","Quantity","Avg Cost","Last Price","Armed","High P/L %","Status","Last Update","Notes"]]
        sheets.write_tab(SHEET_KRAKEN_TAB, header)
        return

    # 2) Build pair maps & prices
    pairs = kraken.get_asset_pairs()
    base_to_pair, pair_to_base = build_pair_maps(pairs, QUOTE_CCY)
    prices = current_prices(kraken, base_to_pair)

    # 3) Trades history (for cost basis)
    trades = kraken.trades_history()

    # 4) Load previous sheet state (for Armed & High P/L)
    df_prev = sheets.read_tab(SHEET_KRAKEN_TAB)
    prev_state = {}
    if not df_prev.empty and "Symbol" in df_prev.columns:
        for _, r in df_prev.iterrows():
            sym = (r.get("Symbol") or "").upper()
            armed = str(r.get("Armed") or "").strip().upper() in ("TRUE","1","YES","Y")
            highpl = to_float(r.get("High P/L %"))
            prev_state[sym] = {"armed": armed, "highpl": highpl}

    # 5) Build rows & decide actions
    rows = [["Symbol","P/L %","Quantity","Avg Cost","Last Price","Armed","High P/L %","Status","Last Update","Notes"]]
    for base, qty in sorted(crypto_bal.items()):
        note = ""
        armed_prev = prev_state.get(base, {}).get("armed", False)
        high_prev = prev_state.get(base, {}).get("highpl", None)

        pair = base_to_pair.get(base)
        last_price = prices.get(base)
        if not pair or last_price is None:
            rows.append([base, "", qty, "", "", str(armed_prev), high_prev if high_prev is not None else "", "HOLD", now_iso(), "No price/pair"])
            continue

        avg_cost = compute_avg_cost_for_position(base, qty, trades, base_to_pair)
        if avg_cost is None or avg_cost <= 0:
            # Fall back: unknown basis (treat P/L as 0 for display, do nothing)
            rows.append([base, "", qty, "", last_price, str(armed_prev), high_prev if high_prev is not None else "", "HOLD", now_iso(), "No cost basis"])
            continue

        pl_pct = ((last_price - avg_cost) / avg_cost) * 100.0

        # Determine next state/action
        armed = armed_prev
        high_pl = high_prev if high_prev is not None else (pl_pct if pl_pct >= ARM_THRESHOLD else None)
        status = "HOLD"

        # Stop loss first
        action = None
        if pl_pct <= LOSS_THRESHOLD:
            status = "STOP_LOSS"
            action = "SELL_ALL"

        # Arm if threshold crossed
        if action is None and (not armed) and pl_pct >= ARM_THRESHOLD:
            armed = True
            high_pl = pl_pct
            status = "ARMED"
            note = f"Armed at {pl_pct:.2f}%"

        # Update high water while armed
        if action is None and armed:
            if high_pl is None or pl_pct > high_pl:
                high_pl = pl_pct
            # Trailing giveback
            if high_pl is not None and pl_pct <= (high_pl - TRAIL_GIVEBACK):
                status = "TAKE_PROFIT"
                action = "SELL_ALL"

        # Execute action (sell 100%) if needed
        if action == "SELL_ALL":
            vol_to_sell = qty
            if LIVE_TRADING:
                try:
                    res = kraken.add_order_market_sell(pair, vol_to_sell)
                    note = f"Sold {vol_to_sell} {base} at ~{last_price} {QUOTE_CCY}"
                    # After selling, reset armed/high
                    armed = False
                    high_pl = None
                    status = "SOLD"
                except Exception as e:
                    note = f"SELL FAILED: {e}"
                    status = "ERROR"
            else:
                # Paper trade
                note = f"(PAPER) Would SELL {vol_to_sell} {base} at ~{last_price} {QUOTE_CCY}"
                armed = False
                high_pl = None
                status = "SOLD"

        # Append row
        rows.append([
            base,
            f"{pl_pct:.2f}",
            f"{qty:.10g}",
            f"{avg_cost:.8f}",
            f"{last_price:.8f}",
            "TRUE" if armed else "FALSE",
            f"{high_pl:.2f}" if high_pl is not None else "",
            status,
            now_iso(),
            note
        ])

    # 6) Write to sheet
    sheets.write_tab(SHEET_KRAKEN_TAB, rows)
    logging.info("Sheet updated.")

def main():
    # Clients
    sheets = SheetClient(GOOGLE_SHEET_NAME)
    kraken = KrakenClient(KRAKEN_KEY, KRAKEN_SECRET)

    logging.info("Starting Kraken tracker & seller (LIVE_TRADING=%s)", LIVE_TRADING)
    while True:
        try:
            run_once(kraken, sheets)
        except Exception as e:
            logging.error("Run error: %s", e)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
