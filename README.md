# JEPA BTC Forecasting

Research infrastructure for testing short-horizon BTC microstructure forecasts.
The first project-defining gate uses one venue and one instrument: Binance
BTC-USDT perpetual L2 book and trades predicting that instrument's own future
mid-price returns.

The project is testing a staged question: does a reproducible, cost-surviving
single-venue signal exist; do spot, derivatives-state, Coinbase/cross-exchange,
or multi-asset inputs add incremental OOS value; and only then does causal
forward JEPA pretraining improve on a strong supervised baseline? It is not a
latency-arbitrage, order-routing, or production trading system.

## Current Status

The repository is in Phase 0: data integrity, staged acquisition, deterministic
reconstruction, and the measurement harness. Current code includes:

- vendor-aware verification, download planning, and normalized ingestion;
- deterministic Python order-book reconstruction and an optional Rust/PyO3
  accelerator with Python as the correctness oracle;
- Coinbase cross-vendor parity and quality-map tooling;
- fail-closed, local-only Binance Stage-2 reconstruction;
- purged/embargoed CV, cost and statistical evaluation, a versioned feature
  manifest contract, and the supervised baseline ladder.

The production bar/feature dataset and CF-JEPA model/training stack are not yet
implemented. The [experiment plan](docs/experiment-plan.md) deliberately requires
the supervised signal gates to pass before deep or JEPA model work begins.

## Research Design

- **First target and signal:** Binance BTC-USDT perpetual own-book/trade
  microstructure predicting Binance future mid returns.
- **First data scope:** `2025-11-01..2026-01-31`; November-December development,
  January untouched OOS.
- **Conditional increments:** Binance spot, derivatives state, Coinbase transfer
  and cross-exchange context, then multi-asset signals. Each must beat the
  preceding rung on identical target rows and costs.
- **Input clock:** trade-driven notional bars with a wall-clock cap.
- **Timing:** event-time reconstruction on a deterministic merged stream; labels
  use fixed physical time rather than a fixed number of bars.
- **Evaluation:** purged and embargoed CPCV, explicit leakage checks, realistic
  costs, no-trade bands, DSR, PBO, and regime slices.
- **Model sequence:** persistence/microprice/linear/LightGBM first; CF-JEPA only
  after the single-venue signal and later increment gates justify complexity.

The binding design rationale and timing rules live in the
[implementation spec](jepa_btc_forecasting_spec.md).

## Pipeline

```text
Crypto Lake / CryptoHFTData / CoinAPI
        |
        v
ingest/ normalized, partitioned market data
        |
        v
recon/ + native/recon_native deterministic top-K books
        |
        v
bar/feature producer + versioned ModelMatrix manifest (planned)
        |
        v
eval/ supervised signal gates
        |
        v
CF-JEPA experiments only after the baseline gates pass (planned)
```

Vendor-specific schemas stop at the ingestion boundary. Downstream components
consume normalized contracts and explicit manifests rather than inferring model
features from arbitrary columns.

## Repository Map

| Path | Purpose |
| --- | --- |
| `ingest/` | Vendor-facing verification, planning, download, and normalization |
| `recon/` | Python reconstruction oracle, parity, reseed, and stitch logic |
| `native/recon_native/` | Optional Rust/PyO3 reconstruction accelerator |
| `data/` | CV/calendar code and ignored generated data products |
| `eval/` | Baselines, cost model, statistical gates, and manifest validation |
| `scripts/` | Bounded operational and local pipeline entry points |
| `tests/` | Synthetic, contract, conformance, and integration tests |
| `docs/` | Data decisions, experiment contracts, plans, and workflow guidance |

Raw market data, generated reports, caches, credentials, and local environments
are ignored and must not be committed.

## Setup

Python 3.12 or newer is required.

```bash
git clone https://github.com/afan2g/jepa-btc-forecasting.git
cd jepa-btc-forecasting

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e .
```

Install only the extras needed for the work at hand:

```bash
# Supervised baseline and Parquet support
.venv/bin/pip install -e '.[baseline]'

# Crypto Lake ingestion support
.venv/bin/pip install -e '.[lake]'

# Full local development environment
.venv/bin/pip install -e '.[baseline,lake]' pytest
```

The native engine is optional. Its build, tick-scale, conformance, and benchmark
instructions are in [docs/native-recon.md](docs/native-recon.md).

## Local Validation

After installing the full local development environment, these focused checks
use synthetic/local fixtures and do not call vendor APIs:

```bash
.venv/bin/python -m pytest -q \
  tests/test_orderbook.py \
  tests/test_reconstruct_no_lookahead.py \
  tests/test_manifest.py \
  tests/test_gate_synthetic.py
```

Run the full suite only when workstation capacity is available:

```bash
.venv/bin/python -m pytest -q
```

Two important local entry points are:

```bash
# Reconstruct already-downloaded Binance partitions; no vendor I/O
.venv/bin/python scripts/run_binance_recon.py --help

# Evaluate an existing ModelMatrix against its versioned manifest
.venv/bin/python scripts/run_baseline.py \
  path/to/model_matrix.parquet path/to/feature_manifest.json
```

The baseline command requires a preregistered v1 manifest and real local matrix;
it is not a bundled demo dataset.

## Data And Operational Safety

- Do not commit `.env`, credentials, raw Parquet/CSV data, caches, or generated
  reports.
- Do not run bulk downloads, live verification, paid vendor calls, or broad raw
  scans without the approval and spend controls required by
  [docs/data.md](docs/data.md).
- CoinAPI and Crypto Lake commands have distinct quota, billing, and credential
  requirements. Read [ingest/README.md](ingest/README.md) before vendor work.
- Serialize full test suites, reconstruction jobs, benchmarks, and data pulls on
  the shared workstation.
- Roadmap dates are planning estimates, not technical commitments or approval
  for vendor spend or compute-heavy work.

## Canonical Documentation

| Document | Authority |
| --- | --- |
| [Implementation spec](jepa_btc_forecasting_spec.md) | Product, data, timing, feature, and model design |
| [Data decisions](docs/data.md) | Vendors, coverage, costs, schemas, and acquisition gates |
| [Experiment plan](docs/experiment-plan.md) | Phases, quantitative gates, and stop/pivot decisions |
| [Feature manifest](docs/feature-manifest.md) | Modeling-data schema, timing, leakage, and training contract |
| [Ingest guide](ingest/README.md) | Vendor tooling and bounded operational commands |
| [Native reconstruction](docs/native-recon.md) | Rust accelerator semantics, build, and validation |
| [Agent workflow](docs/agent-workflow.md) | Issue, Project, branch, review, and merge lifecycle |
| [GitHub Roadmap](https://github.com/users/afan2g/projects/2) | Operational stage, status, dates, and dependency visualization |

Plans and specs remain the technical source of truth. GitHub issues are
authoritative for scope, dependencies, acceptance criteria, and execution state.
The Roadmap is an operational visualization and does not override either one.

## Contributing

Implementation starts from a ready GitHub issue with explicit acceptance
criteria and synchronized Project state. Use one agent, one issue, and one
Conventional Branch branch/worktree; keep changes reviewable; run the relevant
cheap checks; and leave final merge decisions to a human. See
[AGENTS.md](AGENTS.md) and the [agent workflow](docs/agent-workflow.md) for the
complete repository contract.
