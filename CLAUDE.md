# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the screener

```bash
pip install yfinance pandas requests beautifulsoup4 tqdm numpy
python advanced_breakout_screener_sp500_v3_4.py
```

Outputs:
- `breakout_report.html` — interactive dashboard (opened in browser)
- `breakout_results.csv` — raw data
- `.score_history.json` — score smoothing cache (do not delete)
- `.screener_cache/` — yfinance price cache (6h TTL)

Syntax check:
```bash
python -m py_compile advanced_breakout_screener_sp500_v3_4.py
```

## Architecture

Single-file script (~2900 lines). Execution order in `main()`:

1. **Market health** (`analyze_sp500_health`, `get_fear_greed_index`) — fetches SPX technicals + CNN Fear & Greed Index
2. **Ticker list** (`get_tickers`) — Wikipedia S&P 500 list with iShares IVV fallback
3. **Price download** (`download_prices`) — yfinance batch download with `.screener_cache/` file cache; `PARALLEL_WORKERS=4` threads
4. **Pre-filter** (`pre_filter`) — fast per-ticker rejection (price, volume, trend, failed breakout thresholds in constants block ~line 1000)
5. **Scoring** (`analyze`) — computes score_100 from `WEIGHTS` dict; adds ATR(14), signal_age, swing_target/stop, intraday adjustments
6. **Score smoothing** (`analyze_all`) — 3-day EMA via `.score_history.json`: `0.50×today + 0.35×yesterday + 0.15×day_before`
7. **News** (`analyze_news_top_n`) — fetches Yahoo Finance RSS for top 30 scored stocks
8. **HTML output** (`build_html`) — self-contained single-file dashboard; market health block from `build_market_health_html`

## Key constants (top of config block ~line 984)

| Constant | Purpose |
|---|---|
| `WEIGHTS` | Scoring weights dict — all edits here affect final score |
| `MIN_PRICE`, `MIN_AVG_VOLUME`, `MAX_DIST_BELOW_MA60` | Pre-filter thresholds |
| `CACHE_MAX_AGE_H` | Price cache TTL in hours |
| `NEWS_TOP_N` | How many top stocks get news analysis |
| `OUTPUT_HTML` | Output filename (`breakout_report.html`) |

## Score formula

`score_100 = (raw_points / realistic_max) * 100 + fine_adjustment`

- `realistic_max` is computed dynamically as 505 (not the theoretical max)
- `fine_adjustment` (±5 pts) uses continuous float values from RSI, MACD histogram, ADX, day-range position, volume ratio, and signal_age freshness to break ties
- Final score is clamped to [0, 100]

## Market status / live data

`get_market_status()` (stdlib only, no pytz) determines if NYSE is open using DST-aware UTC offset calculation. When open, `vol_scale` extrapolates intraday volume to full-day equivalent (capped at 8×). `MARKET_STATUS` is a global set once in `analyze_all()`.

## GitHub Actions / deployment

`.github/workflows/screener.yml` runs the script hourly Mo–Fr 13–22 UTC, copies `breakout_report.html` → `docs/index.html`, and commits. GitHub Pages serves from `docs/` on branch `sp500_analyzer`. The `.nojekyll` file in `docs/` disables Jekyll.
