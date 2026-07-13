"""Notional-bar producer package (E0.3+, plan 2026-07-03-bar-label-producer).

T1 ships the causal Coinbase dollar-notional clock and its timing contract:
`bars.events` (the received-time-bearing clock input record), `bars.clock` (trailing
threshold schedule, hybrid time cap, monotone decision watermark, backlog-tie
coalescing), and `bars.modes` (the coinbase_only / cross_venue source-mode contract).
T2 ships `bars.snapshot` (the source-neutral dual-cut target-book reads: observable
feature/cost read vs true label anchor, plus the staleness gate). Features, labels,
costs, and the orchestrator are T3-T10 and do NOT live here yet.
"""
