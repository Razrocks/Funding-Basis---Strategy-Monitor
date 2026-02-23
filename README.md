# Crypto Funding + Basis Carry Monitor

Cross-venue carry arbitrage dashboard for perpetual futures across Binance, Hyperliquid, and dYdX v4. Finds the best earn/hedge pairs after all costs, with live orderbook execution analysis. No API keys required for default read-only mode (public endpoints).


## What It Does

> *For a given asset at a given size, which venue pair gives the best carry arbitrage edge after all costs?*

The system polls three venues every 30 seconds and computes:

- **Funding APR** — annualised funding rate (Binance 8h, Hyperliquid/dYdX 1h)
- **Basis APR** — mark/index premium annualised (disabled for dYdX where oracle = mark)
- **Net Carry** — gross carry minus fees, slippage, and borrow costs at a chosen position size
- **Cross-Venue Pair PnL** — funding diff + basis diff - fees - slippage across earn/hedge legs
- **Execution Capacity** — USD fillable at 10bps slippage per side
- **Quality Score** — Quality Score — 0-100, penalises funding volatility and extreme funding z-scores; optionally incorporates OI shocks where available.
- **Trap Tags** — Funding spike, OI shock, Carry unstable, Basis inversion



## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Backend

```bash
uvicorn api.main:app --reload --port 8000
```

On startup the backend creates `data/monitor.db`, polls all three venues immediately, and begins polling every 30 seconds.

### 3. Start the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Fetches live data from the FastAPI backend.

### 4. Run Tests

```bash
pytest tests/ -v
```



## Dashboard

| Tab | Description |
|-----|-------------|
| **Arb Leaderboard** | Cross-venue pairs ranked by Pair PnL (funding diff + basis diff - fees - slippage). Filters: position size, min edge, hold days, funding-only mode. Viability gating with NOT EXECUTABLE warnings. Live data age. |
| **Asset Comparator** | Per-venue carry breakdown cards for a selected asset. Best pair card with PnL decomposition. Net carry vs size sensitivity chart. Funding countdown per venue. Debug panel with raw data. |
| **Execution** | Side-by-side live orderbook depth charts. Capacity summary table. |
| **Events / Risk** | Real-time event stream (funding spikes, OI shocks, basis inversions). Severity/type/time filters. Venue health status. Assumptions table. |



## API Endpoints

```
GET /api/health                           System health + venue status
GET /api/leaderboard                      Single-venue leaderboard
GET /api/snapshots?venue=&symbol=         Latest raw snapshots
GET /api/events?limit=&severity=&hours=   Recent events
GET /api/orderbook/{venue}/{market}       Live orderbook levels
GET /api/history/{venue}/{symbol}?hours=  Time-series for charts
GET /api/cross-venue/{symbol}             Cross-venue comparison for one asset
GET /api/arb-leaderboard                  Ranked cross-venue arb opportunities
```

Cross-venue endpoints accept `size`, `hold_days`, `min_edge`, and `funding_only` query params.


## Assumptions

| Parameter | Value |
|-----------|-------|
| Basis horizon | 1 day (86400s) |
| Default hold period | 30 days (configurable via UI) |
| Borrow APR | 0% |
| Binance taker fee | 4 bps |
| Hyperliquid taker fee | 2.5 bps |
| dYdX taker fee | 5 bps |
| dYdX basis | Always 0 (oracle = mark price) |
| Capacity threshold | 10 bps slippage |
| Default position size | $25,000 (configurable via UI) |


## Known Data Gaps

| Gap | Venue | Handling |
|-----|-------|----------|
| No separate markPrice | dYdX | oraclePrice used for both. Basis = 0. Labelled "oracle=mark" in UI. |
| Predicted funding | dYdX | nextFundingRate is predicted, not realized. |
| Binance OI in base units | Binance | Multiplied by markPrice at fetch time for USD notional. |


## Data Retention

| Data | Retention |
|------|-----------|
| Raw snapshots | 24 hours |
| Derived metrics | 7 days |
| Events | 30 days |

Cleanup runs automatically on each polling cycle.
