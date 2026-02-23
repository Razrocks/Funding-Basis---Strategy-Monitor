"""
Crypto Carry Monitor — Cross-Venue Arbitrage Dashboard
=======================================================
Live dashboard — fetches all data from the FastAPI backend.

Start the backend first:
  uvicorn api.main:app --reload --port 8000

Then run the dashboard:
  streamlit run app/streamlit_app.py
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Carry Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


_DEFAULTS = {
    "arb_size": 25000,
    "min_edge": 1.0,
    "hold_days": 30,
    "funding_only": False,
    "selected_asset": None,
    "exec_venue_a": None,
    "exec_venue_b": None,
    "ev_sev": ["critical", "warning", "info"],
    "ev_win": "Last 24h",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v



@st.cache_data(ttl=30, show_spinner=False)
def _fetch_arb_leaderboard(size: float, min_edge: float, hold_days: int, funding_only: bool = False) -> List[Dict]:
    r = httpx.get(f"{API_BASE}/api/arb-leaderboard",
                  params={"size": size, "min_edge": min_edge / 100, "hold_days": hold_days,
                          "funding_only": funding_only},
                  timeout=10)
    r.raise_for_status()
    return r.json().get("rows", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_cross_venue(symbol: str, size: float, hold_days: int, funding_only: bool = False) -> Dict:
    r = httpx.get(f"{API_BASE}/api/cross-venue/{symbol}",
                  params={"size": size, "hold_days": hold_days, "funding_only": funding_only},
                  timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_leaderboard() -> List[Dict]:
    r = httpx.get(f"{API_BASE}/api/leaderboard", timeout=8)
    r.raise_for_status()
    return r.json().get("rows", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_events(hours: float = 24) -> List[Dict]:
    r = httpx.get(f"{API_BASE}/api/events", params={"hours": hours, "limit": 200}, timeout=8)
    r.raise_for_status()
    return r.json().get("events", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_orderbook(venue: str, market: str) -> Dict:
    r = httpx.get(f"{API_BASE}/api/orderbook/{venue}/{market}", timeout=6)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_snapshots() -> List[Dict]:
    r = httpx.get(f"{API_BASE}/api/snapshots", timeout=8)
    r.raise_for_status()
    return r.json().get("snapshots", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_health() -> Dict:
    r = httpx.get(f"{API_BASE}/api/health", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_history(venue: str, symbol: str, hours: float = 168) -> Dict:
    r = httpx.get(f"{API_BASE}/api/history/{venue}/{symbol}", params={"hours": hours}, timeout=8)
    r.raise_for_status()
    return r.json()



@st.fragment(run_every=30)
def _silent_cache_buster():
    _fetch_arb_leaderboard.clear()
    _fetch_cross_venue.clear()
    _fetch_leaderboard.clear()
    _fetch_events.clear()
    _fetch_health.clear()
    _fetch_orderbook.clear()
    _fetch_history.clear()

_silent_cache_buster()



def pct(v, d=2):       return f"{v*100:+.{d}f}%"
def pct_plain(v, d=2): return f"{v*100:.{d}f}%"

def usd(v):
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def color_cls(v): return "green" if v >= 0 else "red"
def venue_display(v): return {"binance": "Binance", "hyperliquid": "Hyperliquid", "dydx": "dYdX"}.get(v, v)
def venue_cls(v): return {"binance": "venue-bnb", "hyperliquid": "venue-hl", "dydx": "venue-dydx"}.get(v, "venue-label")

def rank_badge(r):
    cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(r, "rank-n")
    return f'<span class="rank {cls}">{r}</span>'

_TAG_CLS = {
    "Funding spike": "funding-spike", "funding_spike": "funding-spike",
    "Crowded": "crowded", "crowded": "crowded",
    "OI shock": "oi-shock", "oi_shock": "oi-shock",
    "Basis inversion": "basis-inversion", "basis_inversion": "basis-inversion",
    "Carry unstable": "carry-unstable", "carry_unstable": "carry-unstable",
    "Funding flip": "funding-flip", "funding_flip": "funding-flip",
    "Carry opportunity": "carry-opportunity", "carry_opportunity": "carry-opportunity",
}

def tag_html(tags):
    if not tags: return '<span class="dim">—</span>'
    return "".join(f'<span class="tag {_TAG_CLS.get(t, "")}">{t}</span>' for t in tags)

def qual_bar(score, width=60):
    if score is None:
        return '<span class="dim">—</span>'
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
    return (f'<div class="bar-container" style="width:{width}px;display:inline-block;">'
            f'<div class="bar-fill" style="width:{int(score)}%;background:{color};"></div>'
            f'</div> <span class="mono" style="font-size:10px;color:#999;">{score:.0f}</span>')

def data_age_str(seconds):
    if seconds is None: return "N/A"
    s = int(seconds)
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"

def live_age_seconds(ts_iso):
    """Compute live data age from an ISO timestamp string. Returns seconds since collection."""
    if not ts_iso:
        return None
    try:
        if isinstance(ts_iso, str):
            dt = datetime.fromisoformat(ts_iso.replace("Z", ""))
        else:
            dt = ts_iso
        return max(0.0, (datetime.utcnow() - dt).total_seconds())
    except Exception:
        return None

def _fmt_next_funding(ts_val):
    """Format next funding time as countdown + HH:MM UTC."""
    if not ts_val:
        return "N/A"
    try:
        if isinstance(ts_val, str):
            dt = datetime.fromisoformat(ts_val.replace("Z", ""))
        else:
            dt = ts_val
        delta = (dt - datetime.utcnow()).total_seconds()
        hh_mm = dt.strftime("%H:%M")
        if delta <= 0:
            return f"{hh_mm} UTC (settling)"
        m = int(delta) // 60
        s = int(delta) % 60
        if m >= 60:
            return f"{m // 60}h {m % 60}m — {hh_mm} UTC"
        return f"{m}m {s}s — {hh_mm} UTC"
    except Exception:
        return "N/A"


_PLOTLY_CFG = {"displayModeBar": False}
_PLOT_LAYOUT = dict(
    paper_bgcolor="#000000", plot_bgcolor="#000000",
    font=dict(color="#888", size=10),
)



st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"]  { background:#000; }
[data-testid="stHeader"]            { background:#000; border-bottom:1px solid #1a1a1a; }
[data-testid="stSidebar"]           { background:#0a0a0a; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="collapsedControl"] { display:none; }
[data-testid="stStatusWidget"]   { display:none !important; }
div[data-testid="stToolbar"]     { display:none !important; }
.stApp > div:first-child         { animation:none !important; opacity:1 !important; }
html, body, [class*="css"] { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }

/* ── Topbar ── */
.topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 24px; background:#000;
    border-bottom:1px solid #1a1a1a; margin-bottom:20px;
}
.topbar-brand { font-size:14px; font-weight:600; color:#ccc; letter-spacing:.3px; }
.topbar-status { display:flex; gap:20px; align-items:center; }
.topbar-clock { font-family:'SF Mono',Consolas,monospace; font-size:12px; color:#888; }
.status-dot { width:6px; height:6px; border-radius:50%; background:#22c55e; display:inline-block; }

/* ── KPI ── */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:20px; }
.kpi-tile {
    background:#0a0a0a; border:1px solid #1a1a1a;
    border-radius:6px; padding:16px 18px;
    border-top:2px solid #333;
}
.kpi-tile.green { border-top-color:#22c55e; }
.kpi-tile.amber { border-top-color:#f59e0b; }
.kpi-tile.red   { border-top-color:#ef4444; }
.kpi-label { font-size:10px; font-weight:500; color:#666; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.kpi-value { font-family:'SF Mono',Consolas,monospace; font-size:20px; font-weight:600; color:#ddd; line-height:1; }
.kpi-value.green { color:#22c55e; }
.kpi-value.amber { color:#f59e0b; }
.kpi-value.red   { color:#ef4444; }
.kpi-delta { font-size:10px; margin-top:4px; color:#555; font-family:'SF Mono',Consolas,monospace; }

/* ── Section headers ── */
.sec-hdr {
    font-size:11px; font-weight:600; color:#888;
    letter-spacing:.5px; padding:0 0 8px; border-bottom:1px solid #1a1a1a;
    margin:24px 0 12px;
}

/* ── Data table ── */
.data-table { width:100%; border-collapse:collapse; font-size:12px; }
.data-table th {
    font-size:10px; font-weight:600; color:#666; text-transform:uppercase;
    letter-spacing:.5px; padding:8px 12px; border-bottom:1px solid #1a1a1a;
    text-align:left; background:#000; position:sticky; top:0; z-index:1;
}
.data-table td { padding:9px 12px; border-bottom:1px solid #111; color:#bbb; vertical-align:middle; }
.data-table tr:hover td { background:#0a0a0a; }
.data-table .mono  { font-family:'SF Mono',Consolas,monospace; }
.data-table .green { color:#22c55e; }
.data-table .red   { color:#ef4444; }
.data-table .amber { color:#f59e0b; }
.data-table .dim   { color:#555; }

/* ── Tags ── */
.tag { display:inline-block; padding:2px 7px; border-radius:3px; font-size:10px;
       font-weight:500; margin:1px; background:#1a1a1a; color:#999; border:1px solid #222; }
.tag.funding-spike    { background:#2d1518; color:#ef4444; border-color:#4a1d21; }
.tag.crowded          { background:#2d2415; color:#f59e0b; border-color:#4a3a1d; }
.tag.oi-shock         { background:#2d1518; color:#ef4444; border-color:#4a1d21; }
.tag.basis-inversion  { background:#1d1528; color:#a78bfa; border-color:#2e2245; }
.tag.carry-unstable   { background:#2d2415; color:#f59e0b; border-color:#4a3a1d; }
.tag.funding-flip     { background:#152028; color:#38bdf8; border-color:#1d3545; }
.tag.carry-opportunity{ background:#152d1a; color:#22c55e; border-color:#1d4a25; }

/* ── Venue labels ── */
.venue-bnb  { color:#f0b90b; font-weight:600; font-size:11px; }
.venue-hl   { color:#a78bfa; font-weight:600; font-size:11px; }
.venue-dydx { color:#6366f1; font-weight:600; font-size:11px; }
.venue-label{ color:#bbb; font-weight:600; font-size:11px; }

/* ── Events ── */
.event-row { display:flex; align-items:flex-start; gap:12px; padding:10px 0; border-bottom:1px solid #111; }
.event-dot { width:6px; height:6px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.event-dot.critical { background:#ef4444; box-shadow:0 0 6px #ef444466; }
.event-dot.warning  { background:#f59e0b; box-shadow:0 0 6px #f59e0b66; }
.event-dot.info     { background:#3b82f6; box-shadow:0 0 6px #3b82f666; }
.event-text  { flex:1; }
.event-title { font-size:12px; font-weight:500; color:#ccc; }
.event-detail{ font-size:11px; color:#666; margin-top:2px; }
.event-meta  { font-family:'SF Mono',Consolas,monospace; font-size:10px; color:#444; white-space:nowrap; }

/* ── Cards ── */
.card { background:#0a0a0a; border:1px solid #1a1a1a; border-radius:6px; padding:18px 20px; height:100%; }

/* ── Rank ── */
.rank { display:inline-flex; align-items:center; justify-content:center;
        width:20px; height:20px; border-radius:50%; font-size:10px; font-weight:700; }
.rank-1 { background:#f0b90b22; color:#f0b90b; border:1px solid #f0b90b44; }
.rank-2 { background:#94a3b822; color:#94a3b8; border:1px solid #94a3b844; }
.rank-3 { background:#cd7f3222; color:#cd7f32; border:1px solid #cd7f3244; }
.rank-n { background:#1a1a1a; color:#666; }

/* ── Scrollable ── */
.scroll-panel { max-height:520px; overflow-y:auto; }
.scroll-panel::-webkit-scrollbar { width:4px; }
.scroll-panel::-webkit-scrollbar-track { background:#000; }
.scroll-panel::-webkit-scrollbar-thumb { background:#222; border-radius:4px; }

/* ── Bars ── */
.bar-container { background:#1a1a1a; border-radius:3px; height:5px; overflow:hidden; }
.bar-fill { height:100%; border-radius:3px; }

/* ── Venue card ── */
.venue-card {
    background:#0a0a0a; border:1px solid #1a1a1a; border-radius:6px;
    padding:18px; height:100%;
}
.venue-card.best { border-color:#22c55e44; border-top:2px solid #22c55e; }
.venue-card .venue-name { font-size:13px; font-weight:600; color:#ccc; margin-bottom:12px; }
.venue-card .metric-row { display:flex; justify-content:space-between; padding:4px 0; font-size:11px; }
.venue-card .metric-label { color:#555; }
.venue-card .metric-val  { font-family:'SF Mono',Consolas,monospace; color:#bbb; }
.venue-card .big-val { font-family:'SF Mono',Consolas,monospace; font-size:22px; font-weight:600; margin:8px 0; }

/* ── Pair card ── */
.pair-card {
    background:#0a0a0a; border:1px solid #1a3a1d; border-radius:6px; padding:20px;
    display:flex; align-items:center; gap:20px; margin:16px 0;
}
.pair-leg { flex:1; text-align:center; }
.pair-leg .leg-venue { font-size:12px; font-weight:600; color:#bbb; }
.pair-leg .leg-val   { font-family:'SF Mono',Consolas,monospace; font-size:18px; font-weight:600; }
.pair-arrow { color:#444; font-size:18px; }
.pair-edge  { flex:1; text-align:center; }
.pair-edge .edge-label { font-size:10px; color:#666; text-transform:uppercase; letter-spacing:1px; }
.pair-edge .edge-val   { font-family:'SF Mono',Consolas,monospace; font-size:26px; font-weight:600; color:#22c55e; }

/* ── Data age ── */
.age-fresh { color:#22c55e; }
.age-warn  { color:#f59e0b; }
.age-stale { color:#ef4444; }

/* ── Streamlit overrides ── */
[data-testid="stHorizontalBlock"] { gap:12px; }
div[data-testid="metric-container"] { display:none; }
.stPlotlyChart { border-radius:6px; overflow:hidden; }
[data-baseweb="tab-list"] { gap:4px; background:transparent; }
[data-baseweb="tab"] { background:#0a0a0a; border:1px solid #1a1a1a; border-radius:4px;
                        font-size:11px; font-weight:500; color:#666; padding:6px 14px; }
[aria-selected="true"][data-baseweb="tab"] { background:#111; color:#ccc; border-color:#333; }
div.stSelectbox > div > div { background:#0a0a0a; border-color:#1a1a1a; color:#bbb; }
div[data-baseweb="select"] > div { background:#0a0a0a !important; border-color:#1a1a1a !important; }
label[data-testid="stWidgetLabel"] { font-size:10px; font-weight:500; color:#666; text-transform:uppercase; letter-spacing:.5px; }
</style>
""", unsafe_allow_html=True)



try:
    _lb_rows = _fetch_leaderboard()
except Exception as _api_err:
    st.error(
        f"**Cannot reach API backend at `{API_BASE}`.**\n\n"
        f"Start it with:\n```\nuvicorn api.main:app --reload --port 8000\n```\n\n"
        f"Error: `{_api_err}`"
    )
    st.stop()

if not _lb_rows:
    st.warning("API is reachable but returned no data yet. The backend may still be polling — refresh in a few seconds.")
    st.stop()

# Build list of all available symbols (from old leaderboard endpoint, which always has data)
_all_symbols = sorted(set(r.get("symbol", "") for r in _lb_rows))
NOW = datetime.utcnow()

# Grab BTC/ETH prices for topbar from SNAPSHOTS (not leaderboard — metrics may lag behind)
try:
    _all_snaps = _fetch_snapshots()
except Exception:
    _all_snaps = []
_btc_price = max((s.get("mark_price") or 0 for s in _all_snaps if s.get("symbol") == "BTC"), default=0)
_eth_price = max((s.get("mark_price") or 0 for s in _all_snaps if s.get("symbol") == "ETH"), default=0)



_btc_str = f"${_btc_price:,.0f}" if _btc_price > 0 else "—"
_eth_str = f"${_eth_price:,.0f}" if _eth_price > 0 else "—"
st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">Carry Monitor</div>
  <div class="topbar-status">
    <span class="topbar-clock">BTC {_btc_str}</span>
    <span class="topbar-clock">ETH {_eth_str}</span>
    <span class="topbar-clock">{NOW.strftime("%H:%M:%S")} UTC</span>
    <span class="status-dot"></span>
  </div>
</div>
""", unsafe_allow_html=True)



tabs = st.tabs([
    "Arb Leaderboard",
    "Asset Comparator",
    "Execution",
    "Events / Risk",
])



with tabs[0]:
    # Controls
    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.5, 1])
    with fc1:
        arb_size = st.select_slider(
            "Position Size",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=st.session_state["arb_size"],
            format_func=lambda x: usd(x),
            key="arb_size_w",
        )
        st.session_state["arb_size"] = arb_size
    with fc2:
        min_edge = st.slider("Min Edge APR %", 0.0, 20.0, st.session_state["min_edge"], 0.5, key="min_edge_w")
        st.session_state["min_edge"] = min_edge
    with fc3:
        hold_days = st.selectbox("Hold Days", [7, 14, 30, 60],
                                 index=[7,14,30,60].index(st.session_state["hold_days"]),
                                 key="hold_days_w")
        st.session_state["hold_days"] = hold_days
    with fc4:
        arb_funding_only = st.toggle("Funding Only", value=st.session_state["funding_only"],
                                     key="funding_only_w",
                                     help="Ignore basis for all venues (level playing field with dYdX)")
        st.session_state["funding_only"] = arb_funding_only

    # Fetch arb leaderboard
    try:
        arb_rows = _fetch_arb_leaderboard(arb_size, min_edge, hold_days, funding_only=arb_funding_only)
    except Exception as e:
        arb_rows = []
        st.warning(f"Could not load arb leaderboard: {e}")

    # KPIs
    if arb_rows:
        best_edge = max(r.get("pair_expected_apr", r.get("edge_apr", 0)) for r in arb_rows)
        n_viable = sum(1 for r in arb_rows if r.get("executable"))
        avg_cap = sum(r.get("pair_capacity_usd", 0) for r in arb_rows) / len(arb_rows) if arb_rows else 0
        worst_qual = min((r.get("quality_min") or 100) for r in arb_rows)
    else:
        best_edge, n_viable, avg_cap, worst_qual = 0, 0, 0, 0

    _eq_cls = "green" if best_edge > 0.05 else "amber" if best_edge > 0 else ""
    _wq_cls = "green" if worst_qual >= 70 else "amber" if worst_qual >= 45 else "red"
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-tile {_eq_cls}">
        <div class="kpi-label">Best Pair PnL</div>
        <div class="kpi-value {_eq_cls}">{pct_plain(best_edge)}</div>
        <div class="kpi-delta">Expected combined APR</div>
      </div>
      <div class="kpi-tile green">
        <div class="kpi-label">Viable Arbs</div>
        <div class="kpi-value">{n_viable}</div>
        <div class="kpi-delta">Both legs @ {usd(arb_size)}</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Avg Pair Capacity</div>
        <div class="kpi-value">{usd(avg_cap)}</div>
        <div class="kpi-delta">Min of both legs</div>
      </div>
      <div class="kpi-tile {_wq_cls}">
        <div class="kpi-label">Lowest Quality</div>
        <div class="kpi-value">{worst_qual:.0f}<span style="font-size:13px;color:#555;">/100</span></div>
        <div class="kpi-delta">Across pairs</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Main table
    _mode_label = "Funding Only" if arb_funding_only else "Funding + Basis"
    st.markdown(f'<div class="sec-hdr">Cross-Venue Arb Rankings — {_mode_label}</div>', unsafe_allow_html=True)

    if not arb_rows:
        st.markdown('<div style="color:#475569;padding:20px;">No arb opportunities found at current settings. '
                    'Try lowering the min edge filter or waiting for more data.</div>', unsafe_allow_html=True)
    else:
        header = """<table class="data-table"><thead><tr>
          <th>#</th><th>Asset</th><th>Venues</th>
          <th>Earn Venue</th><th>Dir</th><th>Earn Net</th>
          <th>Hedge Venue</th><th>Dir</th><th>Hedge Net</th>
          <th>Pair PnL</th><th>Pair Cap</th><th>Exec</th><th>Age</th><th>Quality</th><th>Tags</th>
        </tr></thead><tbody>"""

        body = ""
        for r in arb_rows:
            rk = r.get("rank", 0)
            pair_pnl = r.get("pair_expected_apr", r.get("edge_apr", 0))
            earn_net = r.get("earn_net_carry", 0)
            hedge_net = r.get("hedge_net_carry", 0)
            q_min = r.get("quality_min")
            exec_ok = r.get("executable", False)
            tags = r.get("trap_tags_union", [])
            # Compute age LIVE from collection timestamps (not cached value)
            _earn_age = live_age_seconds(r.get("earn_data_ts"))
            _hedge_age = live_age_seconds(r.get("hedge_data_ts"))
            max_age = max(_earn_age or 0, _hedge_age or 0)
            age_color = "#22c55e" if max_age < 30 else "#f59e0b" if max_age < 120 else "#ef4444"

            earn_v = r.get('earn_venue', '')
            hedge_v = r.get('hedge_venue', '')

            # Build PnL breakdown tooltip with explicit sum
            fd = r.get("funding_diff_apr", 0)
            bd = r.get("basis_diff_apr", 0)
            tf = r.get("total_fees_apr", 0)
            ts_v = r.get("total_slippage_apr", 0)
            _sum_check = fd + bd - tf - ts_v
            _tip = (f"Fund diff: {pct(fd)} + Basis diff: {pct(bd)} "
                    f"- Fees: {pct_plain(tf)} - Slip: {pct_plain(ts_v)} "
                    f"= {pct(_sum_check)}")

            body += f"""<tr>
              <td>{rank_badge(rk)}</td>
              <td style="color:#ddd;font-weight:500;">{r.get('symbol','')}</td>
              <td class="mono dim">{r.get('num_venues', 0)}</td>
              <td class="{venue_cls(earn_v)} mono">{venue_display(earn_v)}</td>
              <td><span class="tag">{r.get('earn_direction','')}</span></td>
              <td class="mono {color_cls(earn_net)}">{pct(earn_net)}</td>
              <td class="{venue_cls(hedge_v)} mono">{venue_display(hedge_v)}</td>
              <td><span class="tag">{r.get('hedge_direction','')}</span></td>
              <td class="mono dim">{pct(hedge_net)}</td>
              <td class="mono {'dim' if not exec_ok else color_cls(pair_pnl)}" style="font-weight:600;{'text-decoration:line-through;' if not exec_ok else ''}" title="{_tip}">{pct(pair_pnl)}</td>
              <td class="mono{' dim' if not exec_ok else ''}">{usd(r.get('pair_capacity_usd', 0))}{f'<div style="font-size:9px;color:#ef4444;">< {usd(arb_size)}</div>' if not exec_ok else ''}</td>
              <td style="color:{'#22c55e' if exec_ok else '#ef4444'};">{'✓' if exec_ok else '✗'}</td>
              <td class="mono" style="color:{age_color};">{data_age_str(max_age)}</td>
              <td>{qual_bar(q_min)}</td>
              <td>{tag_html(tags)}</td>
            </tr>"""

        st.markdown(f'<div class="scroll-panel">{header}{body}</tbody></table></div>', unsafe_allow_html=True)

    st.caption(f"Pair PnL = Funding diff + Basis diff − Fees − Slippage (hover for breakdown). "
               f"Exec = both legs fillable at {usd(arb_size)} within 10bps. Hold: {hold_days}d.")



with tabs[1]:
    ac1, ac2, ac3 = st.columns([2, 1, 1])
    with ac1:
        # Default to first arb row symbol or first available
        default_sym = (arb_rows[0]["symbol"] if arb_rows else _all_symbols[0]) if _all_symbols else "BTC"
        if st.session_state["selected_asset"] and st.session_state["selected_asset"] in _all_symbols:
            default_idx = _all_symbols.index(st.session_state["selected_asset"])
        elif default_sym in _all_symbols:
            default_idx = _all_symbols.index(default_sym)
        else:
            default_idx = 0
        selected_asset = st.selectbox("Asset", _all_symbols, index=default_idx, key="asset_sel_w")
        st.session_state["selected_asset"] = selected_asset
    with ac2:
        cmp_size = st.select_slider(
            "Size",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=st.session_state["arb_size"],
            format_func=lambda x: usd(x),
            key="cmp_size_w",
        )
    with ac3:
        cmp_hold = st.selectbox("Hold Days", [7, 14, 30, 60],
                                index=[7,14,30,60].index(st.session_state["hold_days"]),
                                key="cmp_hold_w")

    # Fetch cross-venue data for this symbol
    cv_data = None
    try:
        cv_data = _fetch_cross_venue(selected_asset, cmp_size, cmp_hold,
                                     funding_only=st.session_state["funding_only"])
    except Exception as e:
        st.warning(f"No cross-venue data for {selected_asset}: {e}")

    if cv_data and cv_data.get("venues"):
        venues_list = cv_data["venues"]
        best_pair = cv_data.get("best_pair")

        # ── 3-column venue comparison cards ──────────────────────────────────
        st.markdown('<div class="sec-hdr">Per-Venue Carry Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(max(len(venues_list), 1))

        # Find which venue is the earn leg for highlighting
        earn_venue = best_pair.get("earn_venue", "") if best_pair else ""

        for i, leg in enumerate(sorted(venues_list, key=lambda x: x.get("net_carry_apr", 0), reverse=True)):
            v_name = venue_display(leg.get("venue", ""))
            v_key = leg.get("venue", "")
            is_best = (v_key == earn_venue)
            net_carry = leg.get("net_carry_apr", 0)
            funding_apr = leg.get("funding_apr", 0)
            basis_apr = leg.get("basis_apr")
            gross = leg.get("gross_carry_apr", 0)
            direction = leg.get("carry_direction", "")
            slip_bps = leg.get("slippage_bps", 0)
            fee_bps = leg.get("taker_fee_bps", 0)
            cost_apr = leg.get("cost_apr", 0)
            cap_sell = leg.get("capacity_sell_10bps", 0)
            cap_buy = leg.get("capacity_buy_10bps", 0)
            spread = leg.get("spread_bps")
            quality = leg.get("quality_score")
            # Compute age LIVE from collection timestamp
            data_age = live_age_seconds(leg.get("data_ts")) or 0
            mark = leg.get("mark_price", 0)
            index_p = leg.get("index_price", 0)
            oi = leg.get("open_interest_usd", 0)

            # dYdX basis display
            if basis_apr is None:
                basis_display = '<span class="dim">N/A</span> <span style="font-size:9px;color:#555;">(oracle=mark)</span>'
            else:
                basis_display = f'<span class="mono {color_cls(basis_apr)}">{pct(basis_apr)}</span>'

            age_cls = "age-fresh" if data_age < 30 else "age-warn" if data_age < 120 else "age-stale"

            with cols[i]:
                st.markdown(f"""
                <div class="venue-card {'best' if is_best else ''}">
                  <div class="venue-name">{v_name}
                    {'<span class="tag carry-opportunity" style="margin-left:8px;">EARN</span>' if is_best else ''}
                  </div>

                  <div class="big-val" style="color:{'#22c55e' if net_carry >= 0 else '#ef4444'};">{pct(net_carry)}</div>
                  <div style="font-size:10px;color:#555;margin-bottom:12px;">Net Carry APR</div>

                  <div class="metric-row"><span class="metric-label">Direction</span>
                    <span class="tag">{direction}</span></div>
                  <div class="metric-row"><span class="metric-label">Funding APR</span>
                    <span class="metric-val {color_cls(funding_apr)}">{pct(funding_apr)}</span></div>
                  <div class="metric-row"><span class="metric-label">Basis APR</span>
                    {basis_display}</div>
                  <div class="metric-row"><span class="metric-label">Gross Carry</span>
                    <span class="metric-val" style="font-weight:500;">{pct(gross)}</span></div>

                  <div style="border-top:1px solid #1a1a1a;margin:8px 0;"></div>

                  <div class="metric-row"><span class="metric-label">Slippage</span>
                    <span class="metric-val">{slip_bps:.2f} bps{'<span style="font-size:9px;color:#555;"> (top-of-book)</span>' if slip_bps == 0 else ''}</span></div>
                  <div class="metric-row"><span class="metric-label">Taker Fee</span>
                    <span class="metric-val">{fee_bps:.1f} bps</span></div>
                  <div class="metric-row"><span class="metric-label">Total Cost APR</span>
                    <span class="metric-val">{pct(cost_apr)}</span></div>

                  <div style="border-top:1px solid #1a1a1a;margin:8px 0;"></div>

                  <div class="metric-row"><span class="metric-label">Spread</span>
                    <span class="metric-val">{f'{spread:.2f} bps' if spread else 'N/A'}</span></div>
                  <div class="metric-row"><span class="metric-label">Cap Sell @10bps</span>
                    <span class="metric-val">{usd(cap_sell)}</span></div>
                  <div class="metric-row"><span class="metric-label">Cap Buy @10bps</span>
                    <span class="metric-val">{usd(cap_buy)}</span></div>
                  <div class="metric-row"><span class="metric-label">Quality</span>
                    {qual_bar(quality, width=50)}</div>
                  <div class="metric-row"><span class="metric-label">OI</span>
                    <span class="metric-val dim">{usd(oi)}</span></div>

                  <div style="border-top:1px solid #1a1a1a;margin:8px 0;"></div>

                  <div class="metric-row"><span class="metric-label">Funding Every</span>
                    <span class="metric-val">{leg.get('funding_interval_seconds', 0) // 3600}h</span></div>
                  <div class="metric-row"><span class="metric-label">Next Funding</span>
                    <span class="metric-val">{_fmt_next_funding(leg.get('next_funding_time'))}</span></div>

                  <div style="margin-top:8px;font-size:10px;" class="{age_cls}">Data: {data_age_str(data_age)} ago</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Best Pair Card ────────────────────────────────────────────────────
        if best_pair:
            earn_v = venue_display(best_pair.get("earn_venue", ""))
            hedge_v = venue_display(best_pair.get("hedge_venue", ""))
            earn_net = best_pair.get("earn_net_carry", 0)
            hedge_net = best_pair.get("hedge_net_carry", 0)
            pair_pnl = best_pair.get("pair_expected_apr", best_pair.get("edge_apr", 0))
            pair_cap = best_pair.get("pair_capacity_usd", 0)
            executable = best_pair.get("executable", False)
            earn_dir = best_pair.get("earn_direction", "")
            hedge_dir = best_pair.get("hedge_direction", "")
            # PnL breakdown
            _fd = best_pair.get("funding_diff_apr", 0)
            _bd = best_pair.get("basis_diff_apr", 0)
            _tf = best_pair.get("total_fees_apr", 0)
            _ts = best_pair.get("total_slippage_apr", 0)

            _not_exec = not executable
            _pnl_color = "#555" if _not_exec else "#22c55e"
            _pnl_text = f'<span style="color:#555;text-decoration:line-through;">{pct(pair_pnl)}</span>' if _not_exec else pct(pair_pnl)
            _exec_banner = (
                f'<div style="background:#2d1518;border:1px solid #4a1d21;border-radius:4px;'
                f'padding:6px 12px;margin-top:8px;text-align:center;">'
                f'<span style="color:#ef4444;font-size:11px;font-weight:600;">NOT EXECUTABLE</span><br>'
                f'<span style="color:#999;font-size:10px;">Pair cap {usd(pair_cap)} &lt; size {usd(cmp_size)}</span>'
                f'</div>'
            ) if _not_exec else (
                f'<div style="font-size:10px;color:#555;margin-top:6px;">'
                f'Cap: {usd(pair_cap)} &middot; Executable'
                f'</div>'
            )

            st.markdown(f"""
            <div class="pair-card" style="{'border-color:#4a1d21;opacity:0.7;' if _not_exec else ''}">
              <div class="pair-leg">
                <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Earn Leg</div>
                <div class="leg-venue">{earn_v}</div>
                <div class="leg-val" style="color:#22c55e;">{pct(earn_net)}</div>
                <div style="font-size:10px;color:#555;">{earn_dir} {best_pair.get('earn_market','')}</div>
              </div>
              <div class="pair-arrow">→</div>
              <div class="pair-edge">
                <div class="edge-label">Pair PnL</div>
                <div class="edge-val" style="color:{_pnl_color};">{_pnl_text}</div>
                <div style="font-size:10px;color:#666;margin-top:6px;line-height:1.6;font-family:'SF Mono',Consolas,monospace;">
                  <span class="{color_cls(_fd)}">Fund diff {pct_plain(_fd)}</span><br>
                  <span class="{color_cls(_bd)}">Basis diff {pct_plain(_bd)}</span><br>
                  <span class="red">Fees −{pct_plain(_tf)}</span><br>
                  <span class="red">Slip −{pct_plain(_ts)}</span>
                </div>
                {_exec_banner}
              </div>
              <div class="pair-arrow">←</div>
              <div class="pair-leg">
                <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Hedge Leg</div>
                <div class="leg-venue">{hedge_v}</div>
                <div class="leg-val" style="color:#999;">{pct(hedge_net)}</div>
                <div style="font-size:10px;color:#555;">{hedge_dir} {best_pair.get('hedge_market','')}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Only one venue available for this asset — no cross-venue pair possible.")

        # ── Sensitivity Chart: Net Carry vs Size per venue ────────────────────
        st.markdown('<div class="sec-hdr">Net Carry vs Position Size — Per Venue</div>', unsafe_allow_html=True)
        sizes_sens = [1000, 5000, 10000, 25000, 50000, 100000]
        fig_sens = go.Figure()
        venue_colors = {"binance": "#f0b90b", "hyperliquid": "#a78bfa", "dydx": "#6366f1"}

        for leg in venues_list:
            v_key = leg.get("venue", "")
            v_name = venue_display(v_key)
            # Approximate net carry at each size from the leg's data
            # We use the gross carry and scale costs with size
            gross_v = leg.get("gross_carry_apr", 0)
            fee_v = leg.get("taker_fee_bps", 0)
            base_slip = leg.get("slippage_bps", 0)
            net_vals = []
            for sz in sizes_sens:
                # Scale slippage roughly: slippage ∝ sqrt(size/base_size)
                scaled_slip = base_slip * (sz / cmp_size) ** 0.5 if cmp_size > 0 else base_slip
                cost = ((fee_v * 2 + scaled_slip * 2) / 10_000) * (365 / cmp_hold)
                net_vals.append((gross_v - cost) * 100)

            fig_sens.add_trace(go.Scatter(
                x=[usd(s) for s in sizes_sens], y=net_vals,
                mode="lines+markers", name=v_name,
                line=dict(color=venue_colors.get(v_key, "#64748b"), width=2),
                marker=dict(size=6),
            ))

        fig_sens.add_hline(y=0, line_dash="dot", line_color="#333", line_width=1)
        fig_sens.update_layout(
            height=300, **_PLOT_LAYOUT,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(gridcolor="#1a1a1a", title="Position Size"),
            yaxis=dict(gridcolor="#1a1a1a", ticksuffix="%", title="Net Carry APR"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888", size=10)),
        )
        st.plotly_chart(fig_sens, use_container_width=True, config=_PLOTLY_CFG, key="cmp_sensitivity")

        # ── Debug Panel ────────────────────────────────────────────────────────
        with st.expander("Debug — Raw Venue Data"):
            for leg in venues_list:
                v_name = venue_display(leg.get("venue", ""))
                st.markdown(f"**{v_name}** — `{leg.get('market','')}`")
                debug_cols = st.columns(4)
                with debug_cols[0]:
                    st.code(f"funding_rate: {leg.get('funding_rate', 0):.8f}\n"
                            f"interval: {leg.get('funding_interval_seconds', 0)}s\n"
                            f"funding_apr: {leg.get('funding_apr', 0):.6f}")
                with debug_cols[1]:
                    basis_v = leg.get('basis_apr')
                    if basis_v is not None:
                        _basis_str = f"{basis_v:.6f}"
                    elif leg.get("venue") == "dydx":
                        _basis_str = "None (oracle=mark)"
                    else:
                        _basis_str = "None (funding-only)"
                    st.code(f"mark: {leg.get('mark_price', 0):.4f}\n"
                            f"index: {leg.get('index_price', 0):.4f}\n"
                            f"basis_apr: {_basis_str}")
                with debug_cols[2]:
                    st.code(f"slippage_bps: {leg.get('slippage_bps', 0):.4f}\n"
                            f"fee_bps: {leg.get('taker_fee_bps', 0)}\n"
                            f"cost_apr: {leg.get('cost_apr', 0):.6f}")
                with debug_cols[3]:
                    _dbg_age = live_age_seconds(leg.get("data_ts")) or 0
                    st.code(f"cap_sell: {leg.get('capacity_sell_10bps', 0):.0f}\n"
                            f"cap_buy: {leg.get('capacity_buy_10bps', 0):.0f}\n"
                            f"data_age: {_dbg_age:.1f}s (live)")

    elif cv_data:
        st.info(f"No venue data available for {selected_asset}. Waiting for backend to poll.")
    # else: warning already shown



with tabs[2]:
    exec_asset = st.session_state.get("selected_asset") or (_all_symbols[0] if _all_symbols else "BTC")

    # Build venue→market mapping from BOTH leaderboard AND cross-venue data for reliability
    asset_venues = {r.get("venue"): r.get("market") for r in _lb_rows if r.get("symbol") == exec_asset}

    # Also try cross-venue endpoint — it finds venues even when metrics haven't been derived yet
    try:
        _exec_cv = _fetch_cross_venue(exec_asset, st.session_state["arb_size"], st.session_state["hold_days"],
                                     funding_only=st.session_state["funding_only"])
        if _exec_cv and _exec_cv.get("venues"):
            for vleg in _exec_cv["venues"]:
                v_key = vleg.get("venue", "")
                if v_key and v_key not in asset_venues:
                    asset_venues[v_key] = vleg.get("market", "")
    except Exception:
        pass

    venue_keys = sorted(asset_venues.keys())

    if len(venue_keys) < 2:
        st.info(f"Need at least 2 venues for {exec_asset} to show execution comparison. "
                f"Available: {', '.join(venue_display(v) for v in venue_keys) or 'none yet'}. "
                f"The backend may still be polling — wait a few seconds and interact with any control.")
    else:
        ex1, ex2 = st.columns(2)
        with ex1:
            venue_a = st.selectbox("Venue A", venue_keys, index=0, format_func=venue_display, key="exec_va")
        with ex2:
            venue_b_default = 1 if len(venue_keys) > 1 else 0
            venue_b = st.selectbox("Venue B", venue_keys, index=venue_b_default, format_func=venue_display, key="exec_vb")

        st.markdown(f'<div class="sec-hdr">Live Orderbook — {exec_asset}</div>', unsafe_allow_html=True)

        ob_cols = st.columns(2)

        for col_idx, (v_key, col) in enumerate([(venue_a, ob_cols[0]), (venue_b, ob_cols[1])]):
            market = asset_venues.get(v_key, "")
            with col:
                st.markdown(f'<div style="font-size:13px;font-weight:500;color:#ccc;margin-bottom:8px;">'
                            f'{venue_display(v_key)} — {market}</div>', unsafe_allow_html=True)
                try:
                    ob = _fetch_orderbook(v_key, market)
                    bids = ob.get("bids", [])
                    asks = ob.get("asks", [])
                    spread_bps = ob.get("spread_bps")

                    if bids and asks:
                        bid_prices = [b["price"] for b in bids]
                        ask_prices = [a["price"] for a in asks]
                        bid_cum = list(np.cumsum([b["price"] * b["size"] / 1000 for b in bids]))
                        ask_cum = list(np.cumsum([a["price"] * a["size"] / 1000 for a in asks]))

                        fig_ob = go.Figure()
                        fig_ob.add_trace(go.Scatter(
                            x=bid_prices, y=bid_cum, name="Bids ($K)",
                            mode="lines", line=dict(color="#22c55e", width=1.2),
                            fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"))
                        fig_ob.add_trace(go.Scatter(
                            x=ask_prices, y=ask_cum, name="Asks ($K)",
                            mode="lines", line=dict(color="#ef4444", width=1.2),
                            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
                        fig_ob.update_layout(
                            height=250, **_PLOT_LAYOUT,
                            margin=dict(l=40, r=10, t=10, b=30),
                            xaxis=dict(gridcolor="#1a1a1a"),
                            yaxis=dict(gridcolor="#1a1a1a", ticksuffix="K"),
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888", size=10)),
                        )
                        st.plotly_chart(fig_ob, use_container_width=True, config=_PLOTLY_CFG,
                                        key=f"exec_ob_{col_idx}_{v_key}")

                        bb = ob.get("best_bid", 0)
                        ba = ob.get("best_ask", 0)
                        st.markdown(f"""
                        <div style="display:flex;gap:16px;font-size:11px;font-family:'SF Mono',Consolas,monospace;color:#888;">
                          <span>Bid: ${bb:,.{'2' if bb < 10 else '0'}f}</span>
                          <span>Ask: ${ba:,.{'2' if ba < 10 else '0'}f}</span>
                          <span>Spread: {spread_bps:.2f} bps</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("No orderbook data available.")
                except Exception as e:
                    st.caption(f"Orderbook unavailable: {e}")

        # Capacity summary
        st.markdown('<div class="sec-hdr">Capacity Summary</div>', unsafe_allow_html=True)
        cap_html = """<table class="data-table"><thead><tr>
            <th>Venue</th><th>Best Bid</th><th>Best Ask</th><th>Spread (bps)</th>
            <th>Bid Levels</th><th>Ask Levels</th>
        </tr></thead><tbody>"""

        for v_key in [venue_a, venue_b]:
            market = asset_venues.get(v_key, "")
            try:
                ob = _fetch_orderbook(v_key, market)
                bb = ob.get("best_bid", 0)
                ba = ob.get("best_ask", 0)
                sp = ob.get("spread_bps", 0)
                n_bids = len(ob.get("bids", []))
                n_asks = len(ob.get("asks", []))
                cap_html += f"""<tr>
                  <td class="mono" style="color:#bbb;">{venue_display(v_key)}</td>
                  <td class="mono">${bb:,.{'4' if bb < 10 else '2'}f}</td>
                  <td class="mono">${ba:,.{'4' if ba < 10 else '2'}f}</td>
                  <td class="mono">{sp:.2f}</td>
                  <td class="mono dim">{n_bids}</td>
                  <td class="mono dim">{n_asks}</td>
                </tr>"""
            except Exception:
                cap_html += f'<tr><td class="mono" style="color:#bbb;">{venue_display(v_key)}</td><td colspan="5" class="dim">unavailable</td></tr>'

        st.markdown(cap_html + "</tbody></table>", unsafe_allow_html=True)



with tabs[3]:
    ef1, ef2, ef3, ef4 = st.columns([1.5, 1.5, 1, 1])
    with ef1:
        sev_f = st.multiselect("Severity", ["critical", "warning", "info"],
                               default=st.session_state["ev_sev"], key="ev_sev_w")
        st.session_state["ev_sev"] = sev_f
    with ef2:
        ev_type_options = ["funding_spike", "oi_shock", "carry_unstable", "basis_inversion",
                           "funding_flip", "carry_opportunity", "regime_shift"]
        type_f = st.multiselect("Type", ev_type_options, default=ev_type_options, key="ev_type_w")
    with ef3:
        win_f = st.selectbox("Window", ["Last 1h", "Last 6h", "Last 24h", "Last 7d"],
                             index=["Last 1h", "Last 6h", "Last 24h", "Last 7d"].index(st.session_state["ev_win"]),
                             key="ev_win_w")
        st.session_state["ev_win"] = win_f
    with ef4:
        sym_filter = st.selectbox("Asset Filter", ["All"] + _all_symbols, index=0, key="ev_sym_w")

    win_h = {"Last 1h": 1, "Last 6h": 6, "Last 24h": 24, "Last 7d": 168}[win_f]
    win_s = win_h * 3600

    _etype_display = {
        "funding_spike": "Funding Spike", "oi_shock": "OI Shock",
        "carry_unstable": "Carry Unstable", "basis_inversion": "Basis Inversion",
        "funding_flip": "Funding Flip", "carry_opportunity": "Carry Opportunity",
        "regime_shift": "Regime Shift",
    }

    events = []
    try:
        for e in _fetch_events(hours=win_h):
            ev_ts = datetime.fromisoformat(e["ts"].replace("Z", ""))
            ago_s = int((NOW - ev_ts).total_seconds())
            etype_raw = e.get("event_type", "")
            etype_disp = _etype_display.get(etype_raw, etype_raw)
            events.append({
                "ts": ev_ts, "sev": e.get("severity", "info"),
                "type_raw": etype_raw, "type": etype_disp,
                "symbol": e.get("symbol", ""),
                "market": f"{e.get('symbol','')} @ {venue_display(e.get('venue',''))}",
                "detail": e.get("message", ""), "ago": ago_s,
            })
    except Exception as _ev_err:
        st.warning(f"Could not load events: {_ev_err}")

    events.sort(key=lambda x: x["ts"], reverse=True)

    n_crit = sum(1 for e in events if e["sev"] == "critical")
    n_warn = sum(1 for e in events if e["sev"] == "warning")
    n_info = sum(1 for e in events if e["sev"] == "info")

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
      <div class="kpi-tile red">
        <div class="kpi-label">Critical</div>
        <div class="kpi-value {'red' if n_crit else ''}">{n_crit}</div>
        <div class="kpi-delta">Active alerts</div>
      </div>
      <div class="kpi-tile amber">
        <div class="kpi-label">Warnings</div>
        <div class="kpi-value {'amber' if n_warn else ''}">{n_warn}</div>
        <div class="kpi-delta">Watch list</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Info</div>
        <div class="kpi-value">{n_info}</div>
        <div class="kpi-delta">Informational</div>
      </div>
      <div class="kpi-tile">
        <div class="kpi-label">Total ({win_f.replace('Last ','')})</div>
        <div class="kpi-value">{len(events)}</div>
        <div class="kpi-delta">All venues</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter events
    ev_filtered = [
        e for e in events
        if e["sev"] in sev_f
        and e["type_raw"] in type_f
        and e["ago"] <= win_s
        and (sym_filter == "All" or e["symbol"] == sym_filter)
    ]

    st.markdown('<div class="sec-hdr">Event Stream</div>', unsafe_allow_html=True)

    if not ev_filtered:
        st.markdown('<div style="color:#475569;padding:20px;font-size:13px;">No events match the current filters.</div>',
                    unsafe_allow_html=True)
    else:
        ev_html = '<div class="scroll-panel">'
        for e in ev_filtered:
            ago = e["ago"]
            age = f"{ago//3600}h {(ago%3600)//60}m ago" if ago >= 3600 else f"{ago//60}m {ago%60}s ago"
            tag_c = "tag"  # neutral styling for all
            ev_html += f"""
            <div class="event-row">
              <div class="event-dot {e['sev']}"></div>
              <div class="event-text">
                <div class="event-title">
                  <span class="{tag_c}" style="margin-right:8px;">{e['type']}</span>
                  {e['market']}
                </div>
                <div class="event-detail">{e['detail']}</div>
              </div>
              <div class="event-meta">{age}</div>
            </div>"""
        ev_html += '</div>'
        st.markdown(ev_html, unsafe_allow_html=True)

    # ── Venue Health ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Venue Health</div>', unsafe_allow_html=True)
    _health_data = {}
    try:
        _health_data = _fetch_health().get("venues", {})
    except Exception:
        pass

    vh_html = """<table class="data-table"><thead><tr>
    <th>Venue</th><th>Status</th><th>Latency</th>
    <th>Markets</th><th>Errors 1h</th></tr></thead><tbody>"""
    _status_color = {"ok": "#22c55e", "degraded": "#f59e0b", "down": "#ef4444", "unknown": "#666"}
    for vk, vn in [("binance", "Binance"), ("hyperliquid", "Hyperliquid"), ("dydx", "dYdX")]:
        vh = _health_data.get(vk, {})
        lat_v = vh.get("last_poll_latency_ms") or 0
        n_mkts = vh.get("markets_active", 0)
        n_errs = vh.get("total_errors_1h", 0)
        status = vh.get("status", "ok" if _health_data else "unknown")
        status_str = {"ok": "Operational", "degraded": "Degraded", "down": "Down",
                      "unknown": "Unknown"}.get(status, "OK")
        s_col = _status_color.get(status, "#666")
        vh_html += f"""<tr>
          <td class="{venue_cls(vk)} mono">{vn}</td>
          <td class="mono" style="color:{s_col};">{status_str}</td>
          <td class="mono dim">{lat_v:.0f} ms</td>
          <td class="mono dim">{n_mkts}</td>
          <td class="mono dim">{n_errs}</td>
        </tr>"""
    st.markdown(vh_html + "</tbody></table>", unsafe_allow_html=True)

    # ── Assumptions ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:20px;">Assumptions</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="data-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
      <tr><td class="dim">Basis horizon</td><td class="mono">1 day (86400s)</td></tr>
      <tr><td class="dim">Cost hold period</td><td class="mono">{st.session_state['hold_days']} days</td></tr>
      <tr><td class="dim">Borrow APR</td><td class="mono">0.00%</td></tr>
      <tr><td class="dim">Default size</td><td class="mono">{usd(st.session_state['arb_size'])}</td></tr>
      <tr><td class="dim">Taker fees</td><td class="mono">Binance 4bps | HL 2.5bps | dYdX 5bps</td></tr>
      <tr><td class="dim">dYdX basis</td><td class="mono">N/A (oracle = mark price)</td></tr>
    </tbody></table>
    """, unsafe_allow_html=True)
