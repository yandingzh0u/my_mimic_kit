from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from summarize_dare_reppo import common_control_mask
from dare_reppo_stress import KINDS


def test_all_arms_share_control_valid_noise_not_separate_convenient_subsets():
    count=len(KINDS)*4
    base=dict(active=np.ones((3,count),dtype=bool),done=np.zeros((3,count),dtype=int),
              errors=np.ones((3,count,7)))
    repeat={k:v.copy() for k,v in base.items()}
    changed={k:v.copy() for k,v in base.items()}
    repeat["errors"][2,4+1,0]+=.01  # Arm1 noise1 invalid in repeated BASELINE.
    changed["errors"][2,2,0]+=.01   # Untouched arm noise2 invalid in intervention.
    # Large errors in a TREATED arm are not a control exclusion criterion.
    changed["errors"][2,8+3,0]+=100
    valid,_,_=common_control_mask(base,repeat,changed,4,True)
    np.testing.assert_array_equal(valid,[True,False,False,True])
    invalid,_,_=common_control_mask(base,repeat,changed,4,False)
    assert not invalid.any()
