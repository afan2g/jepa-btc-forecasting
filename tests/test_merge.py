from recon.events import order_key
from recon.merge import merge_sorted
from recon.ingest import deltas_from_df, trades_from_df
from recon.synthetic import simple_world
import pandas as pd


def _events():
    draw, traw = simple_world()
    # Rename normalized synthetic columns to RAW Lake names before ingest:
    # deltas use sequence_number, trades use id (see recon/ingest.py).
    d = deltas_from_df(pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time", "seq": "sequence_number"}),
                       engine_time_col="origin_time")
    t = trades_from_df(pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"}),
                       engine_time_col="timestamp")
    return d, t


def test_merge_is_globally_sorted_by_order_key():
    d, t = _events()
    merged = merge_sorted(d, t)
    keys = [order_key(e) for e in merged]
    assert keys == sorted(keys)


def test_merge_is_order_invariant_to_input_permutation():
    d, t = _events()
    a = [order_key(e) for e in merge_sorted(d, t)]
    b = [order_key(e) for e in merge_sorted(list(reversed(d)), list(reversed(t)))]
    assert a == b
