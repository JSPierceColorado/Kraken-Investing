"""
Kraken Tracker & Seller (perpetual, trailing TP)
Fixes:
- Use pair CODE (e.g., XXRPZUSD) for prices/orders
- Match TradeHistory on pair CODE or ALTNAME (e.g., XXRPZUSD or XRPUSD)
- Remove SOLD rows from "Kraken Integration"; append to "Kraken Trades"
- Normalize balances (ADA.F -> ADA, XXRP -> XRP, XBT->BTC, XDG->DOGE)
"""

from __future__ import annotations

import os, json, time, logging
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
SHEET_KRAKEN_TAB  = os.getenv("SHEET_KRAKEN_TAB", "Kraken Integration")
SHEET_TRADES_TAB  = os.getenv("SHEET_TRADES_TAB", "Kraken Trades")
QUOTE_CCY         = os.getenv("QUOTE_CCY", "USD").upper()

LOSS_THRESHOLD = float(os.getenv("LOSS_THRESHOLD", "-3"))  # %
ARM_THRESHOLD  = float(os.getenv("ARM_THRESHOLD", "5"))    # %
TRAIL_GIVEBACK = float(os.getenv("TRAIL_GIVEBACK", "2"))   # %
INTERVAL_SEC   = int(os.getenv("INTERVAL_SEC", "300"))
LIVE_TRADING   = os.getenv("LIVE_TRADING", "false").lower() == "true"

KRAKEN_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Kraken base-code mappings
KRKN_BASE_TO_COMMON = {"XBT": "BTC", "XDG": "DOGE"}
COMMON_TO_KRKN_BASE = {"BTC": "XBT", "DOGE": "XDG"}

BALANCE_SUFFIXES = (".F", ".S", ".M", ".P")
FIAT_LIKE = {"USD","EUR","GBP","CAD","AUD","JPY","CHF","NZD"}

# ---------- Helpers ----------
def now_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def to_float(x) -> Optional[float]:
    try: return float(x)
    except Exception: return None

def normalize_balance_asset(asset_code: str) -> str:
    code = asset_code
    if code.startswith(("X","Z")) and len(code) >= 2:
        code = code[1:]
    for suf in BALANCE_SUFFIXES:
        if code.endswith(suf):
            code = code[:-len(suf)]
            break
    return KRKN_BASE_TO_COMMON.get(code.upper(), code.upper())

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

    def ensure_tab(self, tab: str, rows: int = 2000, cols: int = 20):
        try:
            return self.sheet.worksheet(tab)
        except gspread.WorksheetNotFound:
            return self.sheet.add_worksheet(title=tab, rows=str(rows), cols=str(cols))

    def read_tab(self, tab: str) -> pd.DataFrame:
        ws = self.ensure_tab(tab)
        data = ws.get_all_values()
        if not data: return pd.DataFrame()
        header, *rows = data
        if not header: return pd.DataFrame()
        return pd.DataFrame(rows, columns=header)

    def write_tab(self, tab: str, rows: List[List]):
        ws = self.ensure_tab(tab)
        ws.clear()
        if rows:
            ws.update(rows, "A1", value_input_option="RAW")  # values first

    def append_rows(self, tab: str, rows: List[List]):
        ws = self.ensure_tab(tab)
        if rows:
            ws.append_rows(rows, value_input_option="RAW")

# ---------- Kraken ----------
class KrakenClient:
    PUBLIC = "https://api.kraken.com/0/public"
    def __init__(self, key: str, secret: str):
        self.api = krakenex.API(key, secret)
        self.session = requests.Session()

    # Public
    def get_asset_pairs(self) -> dict:
        r = self.session.get(f"{self.PUBLIC}/AssetPairs", timeout=20)
        r.raise_for_status()
        res = r.json()
        if res.get("error"): raise RuntimeError(res["error"])
        return res["result"]

    def ticker_by_paircodes(self, pair_codes: List[str]) -> dict:
        params = {"pair": ",".join(pair_codes)}
        r = self.session.get(f"{self.PUBLIC}/Ticker", params=params, timeout=20)
        r.raise_for_status()
        res = r.json()
        if res.get("error"): raise RuntimeError(res["error"])
        return res["result"]  # keys are the pair codes you requested

    # Private via krakenex
    def balance(self) -> dict:
        res = self.api.query_private("Balance")
        if res.get("error"): raise RuntimeError(res["error"])
        return res["result"]

    def trades_history(self) -> dict:
        res = self.api.query_private("TradesHistory", {"trades": True})
        if res.get("error"): raise RuntimeError(res["error"])
        return res["result"]["trades"]

    def add_order_market_sell(self, pair_code: str, volume: float) -> dict:
        payload = {"pair": pair_code, "type": "sell", "ordertype": "market", "volume": str(volume)}
        res = self.api.query_private("AddOrder", payload)
        if res.get("error"): raise RuntimeError(res["error"])
        return res["result"]

# ---------- Pair mapping & pricing ----------
def build_pair_maps(asset_pairs: dict, quote_ccy: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      base_to_paircode:  BTC -> XXBTZUSD
      base_to_altname:   BTC -> XBTUSD
    Only pairs quoted in quote_ccy are considered.
    """
    quote_ccy = quote_ccy.upper()
    base_to_paircode: Dict[str, str] = {}
    base_to_altname: Dict[str, str] = {}

    for pair_code, info in asset_pairs.items():
        altname = info.get("altname")   # e.g., XBTUSD
        wsname  = info.get("wsname")    # e.g., XBT/USD
        if not altname or not wsname or "/" not in wsname:
            continue
        base, quote = wsname.split("/")
        if quote.upper() != quote_ccy:
            continue
        common = KRKN_BASE_TO_COMMON.get(base.upper(), base.upper())
        base_to_paircode[common] = pair_code
        base_to_altname[common] = altname
    return base_to_paircode, base_to_altname

def current_prices(kraken: KrakenClient, base_to_paircode: Dict[str, str]) -> Dict[str, float]:
    if not base_to_paircode: return {}
    pair_codes = list(base_to_paircode.values())
    out: Dict[str, float] = {}
    BATCH = 20
    for i in range(0, len(pair_codes), BATCH):
        batch = pair_codes[i:i+BATCH]
        t = kraken.ticker_by_paircodes(batch)
        for code in batch:
            data = t.get(code)
            if not data: continue
            price = to_float(data.get("c", [None])[0])
            if price is None: continue
            # map back by reverse lookup
            base = next((b for b, pc in base_to_paircode.items() if pc == code), None)
            if base: out[base] = price
    return out

# ---------- Cost basis ----------
def compute_avg_cost_fifo(base: str, qty: float, trades: dict,
                          base_to_paircode: Dict[str, str],
                          base_to_altname: Dict[str, str]) -> Optional[float]:
    """FIFO avg cost for current qty. Match trades on pair CODE or ALTNAME."""
    pair_code = base_to_paircode.get(base)
    altname   = base_to_altname.get(base)
    if not pair_code or qty <= 0:
        return None

    rows = []
    for _, tr in trades.items():
        p = tr.get("pair")
        if p != pair_code and p != altname:
            continue
        ttype = tr.get("type")              # buy/sell
        vol   = to_float(tr.get("vol"))
        cost  = to_float(tr.get("cost"))    # quote total
        fee   = to_float(tr.get("fee") or 0.0)
        time_ = float(tr.get("time", 0.0))
        if not vol or cost is None:
            continue
        unit_cost = (cost + fee) / vol if vol else None
        if unit_cost is None:
            continue
        rows.append((time_, ttype, vol, unit_cost))

    if not rows:
        return None

    rows.sort(key=lambda r: r[0])

    lots: List[Tuple[float, float]] = []  # (vol, unit_cost)
    for _, ttype, vol, ucost in rows:
        if ttype == "buy":
            lots.append((vol, ucost))
        elif ttype == "sell":
            remaining = vol
            new_lots = []
            for lvol, lcost in lots:
                if remaining <= 0:
                    new_lots.append((lvol, lcost)); continue
                take = min(lvol, remaining)
                lvol_after = lvol - take
                remaining -= take
                if lvol_after > 0:
                    new_lots.append((lvol_after, lcost))
            lots = new_lots

    remaining = qty
    total_cost = 0.0
    total_vol  = 0.0
    for lvol, lcost in lots:
        if remaining <= 0: break
        take = min(lvol, remaining)
        total_cost += take * lcost
        total_vol  += take
        remaining  -= take

    if total_vol <= 0:
        return None
    return total_cost / total_vol

# ---------- Core loop ----------
def run_once(kraken: KrakenClient, sheets: SheetClient):
    # 1) Balances → normalize → aggregate
    bal = kraken.balance()
    crypto_bal: Dict[str, float] = {}
    for asset_code, amount in bal.items():
        qty = to_float(amount)
        if not qty or qty <= 0:
            continue
        base = normalize_balance_asset(asset_code)  # e.g., XXRP -> XRP, ADA.F -> ADA
        if base in FIAT_LIKE:
            continue
        crypto_bal[base] = crypto_bal.get(base, 0.0) + qty

    # 2) Build pair maps (USD only) & filter balances to tradable
    pairs = kraken.get_asset_pairs()
    base_to_paircode, base_to_altname = build_pair_maps(pairs, QUOTE_CCY)
    crypto_bal = {b:q for b,q in crypto_bal.items() if b in base_to_paircode}
    if not crypto_bal:
        logging.info("No tradable USD pairs in balances.")
        sheets.write_tab(SHEET_KRAKEN_TAB, [["Symbol","P/L %","Quantity","Avg Cost","Last Price","Armed","High P/L %","Status","Last Update","Notes"]])
        return

    # 3) Prices (by pair code) & trades
    prices = current_prices(kraken, base_to_paircode)
    trades = kraken.trades_history()

    # 4) Previous state (armed/high)
    df_prev = sheets.read_tab(SHEET_KRAKEN_TAB)
    prev_state = {}
    if not df_prev.empty and "Symbol" in df_prev.columns:
        for _, r in df_prev.iterrows():
            sym = (r.get("Symbol") or "").upper()
            armed = str(r.get("Armed") or "").strip().upper() in ("TRUE","1","YES","Y")
            highpl = to_float(r.get("High P/L %"))
            prev_state[sym] = {"armed": armed, "highpl": highpl}

    # 5) Process positions; SOLD rows are removed (logged instead)
    header = ["Symbol","P/L %","Quantity","Avg Cost","Last Price","Armed","High P/L %","Status","Last Update","Notes"]
    keep_rows: List[List] = [header]
    trade_logs: List[List] = []  # [time, action, symbol, qty, price, mode, note]

    for base, qty in sorted(crypto_bal.items()):
        pair_code = base_to_paircode.get(base)
        last_price = prices.get(base)
        if not pair_code or last_price is None:
            logging.info("Skip %s: no price/pair", base)
            continue

        avg_cost = compute_avg_cost_fifo(base, qty, trades, base_to_paircode, base_to_altname)
        if avg_cost is None or avg_cost <= 0:
            logging.info("Skip %s: no cost basis from trade history", base)
            continue

        pl_pct = ((last_price - avg_cost) / avg_cost) * 100.0

        armed_prev = prev_state.get(base, {}).get("armed", False)
        high_prev  = prev_state.get(base, {}).get("highpl", None)

        armed   = armed_prev
        high_pl = high_prev if high_prev is not None else (pl_pct if pl_pct >= ARM_THRESHOLD else None)
        status  = "HOLD"
        action  = None
        qty_to_sell = 0.0
        note = ""

        # Stop loss first
        if pl_pct <= LOSS_THRESHOLD:
            status = "STOP_LOSS"; action = "SELL_ALL"; qty_to_sell = qty

        # Arm when threshold crossed
        if action is None and (not armed) and pl_pct >= ARM_THRESHOLD:
            armed = True; high_pl = pl_pct; status = "ARMED"; note = f"Armed at {pl_pct:.2f}%"

        # Trailing giveback
        if action is None and armed:
            if high_pl is None or pl_pct > high_pl:
                high_pl = pl_pct
            if high_pl is not None and pl_pct <= (high_pl - TRAIL_GIVEBACK):
                status = "TAKE_PROFIT"; action = "SELL_ALL"; qty_to_sell = qty

        # Execute sell → do not keep row
        if action == "SELL_ALL":
            ts = now_iso()
            if LIVE_TRADING and KRAKEN_KEY and KRAKEN_SECRET:
                try:
                    kraken.add_order_market_sell(pair_code, qty_to_sell)
                    note = f"Sold {qty_to_sell} {base} @~{last_price} {QUOTE_CCY}"
                    trade_logs.append([ts,"SELL",base,f"{qty_to_sell:.10g}",f"{last_price:.8f}","LIVE",note])
                    continue
                except Exception as e:
                    err = f"SELL FAILED: {e}"
                    logging.error("%s %s", base, err)
                    keep_rows.append([base,f"{pl_pct:.2f}",f"{qty:.10g}",f"{avg_cost:.8f}",f"{last_price:.8f}",
                                      "TRUE" if armed else "FALSE",
                                      f"{high_pl:.2f}" if high_pl is not None else "", "ERROR", ts, err])
                    continue
            else:
                note = f"(PAPER) Would SELL {qty_to_sell} {base} @~{last_price} {QUOTE_CCY}"
                trade_logs.append([ts,"SELL",base,f"{qty_to_sell:.10g}",f"{last_price:.8f}","PAPER",note])
                continue  # do not keep row

        # Keep active row
        keep_rows.append([
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

    # 6) Write active positions + append trade logs
    sheets.write_tab(SHEET_KRAKEN_TAB, keep_rows)
    logging.info("Positions sheet updated: %d active rows", len(keep_rows)-1)
    if trade_logs:
        if sheets.read_tab(SHEET_TRADES_TAB).empty:
            sheets.write_tab(SHEET_TRADES_TAB, [["Time","Action","Symbol","Quantity","Price","Mode","Note"]])
        sheets.append_rows(SHEET_TRADES_TAB, trade_logs)
        logging.info("Logged %d trade(s).", len(trade_logs))

def main():
    if not (KRAKEN_KEY and KRAKEN_SECRET):
        logging.warning("KRAKEN_API_KEY/SECRET missing or empty. Running in PAPER mode.")
    sheets = SheetClient(GOOGLE_SHEET_NAME)
    kraken = KrakenClient(KRAKEN_KEY, KRAKEN_SECRET)

    logging.info("Starting Kraken tracker & seller (LIVE_TRADING=%s)", LIVE_TRADING and bool(KRAKEN_KEY and KRAKEN_SECRET))
    while True:
        try:
            run_once(kraken, sheets)
        except Exception as e:
            logging.error("Run error: %s", e)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
