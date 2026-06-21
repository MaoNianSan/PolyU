from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.self_check import run_self_check

def test_stagev6_source_layout_smoke():
    result=run_self_check(require_features=True)
    assert result['passed'] is True
