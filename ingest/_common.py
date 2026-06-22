"""Shared CoinAPI helpers: env loading, REST GET, and quota-gate handling."""
from __future__ import annotations
import os
import json
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
