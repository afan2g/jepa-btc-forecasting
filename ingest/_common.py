"""Shared CoinAPI helpers: env loading, REST GET, quota-gate, and the backfill gate.

Stdlib-only on purpose: the backfill gate lives here (not in download_coinapi.py) so a CI-safe
unit test can import it without pulling the downloader's pyarrow/boto3/coinapi_flatfiles deps,
which are NOT in pyproject's default dependencies.
"""
from __future__ import annotations
import datetime as dt
import os
import json
import sys
import urllib.request
import urllib.error

REST_BASE = "https://rest.coinapi.io"

QUOTA_HINT = (
    "\n*** CoinAPI quota gate hit (HTTP 403, $0 usable credit). ***\n"
    "The key authenticates, but the organization has no usable credit/subscription.\n"
    "The $25 free credit is granted only after you VERIFY A PAYMENT METHOD, and it must\n"
    "be present as Usage Credits. Fix in the Customer Portal:\n"
    "  Billing -> verify payment method -> Add Usage Credits (and/or enable auto-recharge).\n"
    "Note: Market Data REST/WS credit and Flat Files credit are SEPARATE pools — fund the\n"
    "one you intend to use. Re-run this script once credit shows > $0.\n"
)


class QuotaExceeded(Exception):
    pass


def load_env(path: str = ".env") -> dict:
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    merged = {**env, **os.environ}
    if "COINAPI_KEY" not in merged:
        raise SystemExit("COINAPI_KEY not found in .env or environment.")
    return merged


def is_quota_error(body: str) -> bool:
    return "Insufficient Usage Credits" in body or "Quota exceeded" in body


# --- backfill gate (docs/data.md §5a/§8) ---------------------------------------------------------
BACKFILL_GATE_EXIT = 4        # mirrors run_coinbase_parity.py's small-int exit-code convention
SMOKE_SAMPLE_CAP_MB = 64      # a multi-day "sample" larger than this is a near-full (billable) pull,
                              # not a smoke test — process_day uses --sample-mb directly as the per-day
                              # S3 byte range, so an uncapped sample would bypass the gate.


def check_backfill_gate(start: dt.date, end: dt.date, *, sample_mb: int, allow_backfill: bool) -> None:
    """Block a backfill-scale CoinAPI pull before the §5a parity + reseed gates pass. Prints the
    reason to stderr and raises SystemExit(4) (a string SystemExit would exit 1 and skip the int-code
    contract). Allowed without override: a single day (the parity pilot), or a multi-day range whose
    `--sample-mb` is a small smoke (1..SMOKE_SAMPLE_CAP_MB). Blocked: a multi-day FULL pull, or a
    multi-day `--sample-mb` large enough to fetch near-full daily files. `--allow-backfill` overrides."""
    n_days = (end - start).days + 1
    if n_days <= 1 or allow_backfill:
        return
    if 0 < sample_mb <= SMOKE_SAMPLE_CAP_MB:
        return
    why = (f"--sample-mb {sample_mb} exceeds the {SMOKE_SAMPLE_CAP_MB}MB smoke cap (≈ a near-full "
           f"per-day pull across {n_days} days)" if sample_mb else f"a full pull across {n_days} days")
    print(
        f"REFUSING multi-day backfill pull ({start}..{end}): {why}. The §5a Coinbase vendor-parity "
        "gate has NOT passed (Lake book_delta_v2 reseed pending — docs/data.md §5a); bulk backfill is "
        "blocked until parity + reseed pass.\n"
        "  • For the parity pilot, pull ONE overlap day at a time: --start D --end D\n"
        f"  • For a cheap smoke test, use a multi-day range with --sample-mb ≤ {SMOKE_SAMPLE_CAP_MB}\n"
        "  • To override once the gate passes (or for a deliberate, budgeted pull), pass "
        "--allow-backfill (ensure CoinAPI Spend Management is enabled, §8).",
        file=sys.stderr,
    )
    raise SystemExit(BACKFILL_GATE_EXIT)


def rest_get(key: str, path: str, timeout: int = 45):
    """GET rest.coinapi.io{path} -> parsed JSON. Raises QuotaExceeded on the 403 quota gate."""
    req = urllib.request.Request(REST_BASE + path, headers={"X-CoinAPI-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        if e.code == 403 and is_quota_error(body):
            raise QuotaExceeded(body) from None
        raise RuntimeError(f"HTTP {e.code}: {body[:300]}") from None
