"""The Phase-1 evaluator imports scikit-learn + lightgbm at module load and the CLI reads
parquet; a fresh metadata-based install must declare these or it fails before any G1 run."""
import pathlib
import tomllib


def test_baseline_runtime_deps_declared_in_metadata():
    pp = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    proj = pp["project"]
    declared = list(proj.get("dependencies", []))
    for group in proj.get("optional-dependencies", {}).values():
        declared += group
    blob = " ".join(declared).lower()
    for dep in ("lightgbm", "scikit-learn", "pyarrow"):
        assert dep in blob, f"{dep!r} not declared in pyproject project metadata"


def test_baseline_packages_are_shipped():
    # A non-editable install must ship eval/ and data/, not only recon/, or imports like
    # `from eval.runner import ...` fail outside the repo root even with deps present.
    pp = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    include = pp["tool"]["setuptools"]["packages"]["find"]["include"]
    assert any(p.startswith("eval") for p in include), include
    assert any(p.startswith("data") for p in include), include
