"""ModelMatrix contract: reserved-column registry + explicit feature manifest."""
from __future__ import annotations
import pandas as pd

RESERVED = (
    "y_fwd_bps", "label", "t_event", "t_barrier", "t_feature_start", "t_available",
    "cost_bps", "half_spread_bps", "uniqueness", "regime", "horizon",
)


def validate_matrix(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Validate the contract. Features come from the explicit manifest, never inferred."""
    for c in RESERVED:
        if c not in df.columns:
            raise ValueError(f"ModelMatrix missing reserved column {c!r}")
    reserved_in_manifest = set(feature_cols) & set(RESERVED)
    if reserved_in_manifest:
        raise ValueError(f"feature manifest includes reserved columns: {reserved_in_manifest}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"manifest features not in matrix: {missing}")
    if not (df["t_barrier"] >= df["t_event"]).all():
        raise ValueError("invalid span: require t_barrier >= t_event")
    if not (df["t_available"] >= df["t_event"]).all():
        raise ValueError("invalid timing: require t_available >= t_event")
    if not (df["t_feature_start"] <= df["t_event"]).all():
        raise ValueError("invalid timing: require t_feature_start <= t_event")
    if not (df["t_available"] == df["t_event"]).all():
        raise ValueError("baseline requires t_available == t_event (synchronous decide-and-act; "
                         "model cross-venue latency upstream by lagging features)")
    if not df["label"].isin((-1, 0, 1)).all():
        # lgbm_clf only maps classes +1/-1 to up/down; a stray class (e.g. {0,1,2} from a
        # mislabeled job) would be silently ignored and corrupt the forecast -> fail closed.
        raise ValueError("label must be in {-1, 0, +1}")
    if not ((df["uniqueness"] > 0) & (df["uniqueness"] <= 1)).all():
        raise ValueError("uniqueness must be in (0, 1]")
    # Fail closed on malformed cost rows: negative cost_bps or a crossed/negative
    # half_spread_bps would make total_cost negative, inverting the no-trade band (every row
    # trades) and turning the cost charge into credited PnL -> a bad book row could inflate G1.
    if not (df["cost_bps"] >= 0).all():
        raise ValueError("cost_bps must be non-negative (fees + slippage)")
    if not (df["half_spread_bps"] >= 0).all():
        raise ValueError("half_spread_bps must be non-negative (no crossed/negative spread)")
