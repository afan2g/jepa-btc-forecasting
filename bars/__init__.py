"""Notional-bar producer package (E0.3+, plan 2026-07-03-bar-label-producer).

T1 ships the causal Coinbase dollar-notional clock and its timing contract:
`bars.events` (the received-time-bearing clock input record), `bars.clock` (trailing
threshold schedule, hybrid time cap, monotone decision watermark, backlog-tie
coalescing), and `bars.modes` (the coinbase_only / cross_venue source-mode contract).
T2 ships `bars.snapshot` (the source-neutral dual-cut target-book reads: observable
feature/cost read vs true label anchor, plus the staleness gate). T3 ships
`bars.features` (the causal stationarized single-venue per-bar feature vector over
T1 members + T2 observable reads, with the top-K ladder exposed on the observable
read). T7 ships `bars.cost` (per-row `cost_bps`/`half_spread_bps` from the dual
reads + the explicit versioned venue fee/slippage assumption contract). T9 ships
`bars.produce` (the deterministic day-partitioned G0-BN `binance_single_venue`
orchestration: `produce_development` plus the sole 67-E blind-materializer
boundary `materialize_holdout`). Labels live in `data/` (T5/T6) and the manifest
writer in `eval/writer.py` (T8); the cross-venue increment (T4) and the
operational one-shot transaction (#69) do NOT live here.
"""
