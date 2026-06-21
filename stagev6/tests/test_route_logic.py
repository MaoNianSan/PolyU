from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cascade import hard_route

def test_gate_positive_is_direct_ad_even_when_branch_is_negative():
    pred,pmix,route=hard_route(np.array([0.7,0.2]),np.array([0.1,0.8]))
    assert pred.tolist()==[1,1]
    assert route.tolist()==[1,0]
    assert np.all((pmix>0)&(pmix<1))
