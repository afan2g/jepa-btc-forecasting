"""Notional-bar producer package (E0.3+, plan 2026-07-03-bar-label-producer).

T1 ships the causal Coinbase dollar-notional clock and its timing contract:
`bars.events` (the received-time-bearing clock input record), `bars.clock` (trailing
threshold schedule, hybrid time cap, monotone decision watermark, backlog-tie
coalescing), and `bars.modes` (the coinbase_only / cross_venue source-mode contract).
Snapshots, features, labels, costs, and the orchestrator are T2-T10 and do NOT live
here yet.
"""
