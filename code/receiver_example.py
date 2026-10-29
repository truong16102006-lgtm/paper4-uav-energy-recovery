"""Minimal received-data-only path; no dense-reference argument."""
from common4 import ROOT,write
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from fast_center import recover_clipped
from calibration_enclosure import enclosure
from threshold_audit import audit_thresholds

d=json.loads((ROOT/'examples/threshold_history.json').read_text());a=np.array(d['time_lower_s']);b=np.array(d['time_upper_s']);y=np.array(d['counter_Wh']);eps=d['epsilon_Wh'];cap=d['cap_W']
if 'calibration' in d:
    s=d['calibration'];e=enclosure(a,b,y,cap,eps,s['relative_gain_bound'],s['bias_bound_W']);assert e['status']=='CONDITIONAL_CALIBRATION_RELAXATION';y=e['centers'];eps=e['epsilon']
anchors=np.array(d.get('nominal_times_s',(a+b)/2));grid=np.unique(np.r_[np.linspace(a[0],b[-1],513),anchors]);prior=PchipInterpolator(anchors,y-y[0])(grid)
g=recover_clipped(grid,prior,a,b,y,cap,eps);assert g['status']=='CHECKED_CLIPPED_MINIMAX_CENTER'
audit=audit_thresholds(g['grid'],g['lower'],g['upper'],d['energy_thresholds_Wh'],d.get('audit_times_s'))
write(ROOT/'example_output.json',dict(recovery_status=g['status'],radius_Wh=g['radius_Wh'],threshold_audit=audit,reference_used=False))
print(json.dumps(audit,indent=2))
