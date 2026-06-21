"""No-API structural smoke test for stagev5 source layout."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.self_check import run_self_check


def test_stagev5_source_layout_smoke():
    result = run_self_check(require_features=False)
    assert result["passed"] is True
    assert result["missing"] == []
    assert result["source_hash_mismatches"] == []


if __name__ == "__main__":
    run_self_check(require_features=False)
