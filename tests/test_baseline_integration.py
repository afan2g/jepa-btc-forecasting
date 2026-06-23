import pathlib, json, pytest
import pandas as pd
from eval.runner import run_from_manifest

MATRIX = pathlib.Path("data/processed/model_matrix.parquet")
MANIFEST = pathlib.Path("data/processed/feature_manifest.json")

pytestmark = pytest.mark.skipif(not (MATRIX.exists() and MANIFEST.exists()),
    reason="needs real ModelMatrix + manifest from bars (E0.3) + labels (E0.4)")

def test_real_matrix_runs_through_manifest():
    m = pd.read_parquet(MATRIX); man = json.load(open(MANIFEST))
    res = run_from_manifest(m, man)                      # same path the CLI uses
    assert res["gate"] and res["horizons"]
    any_h = next(iter(res["horizons"].values()))
    assert "g1_pass" in any_h and any_h["per_regime"]
