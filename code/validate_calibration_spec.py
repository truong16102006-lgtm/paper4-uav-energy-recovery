"""Check component-to-power uncertainty bounds using independently drawn errors."""
from common4 import *
from calibration_spec import power_bounds
import numpy as np

def main(out):
    rng=np.random.default_rng(2026102715);maximum=0.;n=40000
    for batch in range(40):
        gv,gi=rng.uniform(0,.2,2);bv,bi=rng.uniform(0,1,2);Vmax,Imax=rng.uniform(5,100,2);bound=power_bounds(gv,gi,bv,bi,Vmax,Imax);V=rng.uniform(0,Vmax,1000);I=rng.uniform(0,Imax,1000);dv=rng.uniform(-gv,gv,1000);di=rng.uniform(-gi,gi,1000);ev=rng.uniform(-bv,bv,1000);ei=rng.uniform(-bi,bi,1000);observed=((1+dv)*V+ev)*((1+di)*I+ei);excess=np.abs(observed-V*I)-bound['relative_power_gain_bound']*V*I-bound['additive_power_bias_bound_W'];maximum=max(maximum,float(excess.max()));assert np.all(excess<=1e-10)
    out.mkdir(parents=True,exist_ok=True);write(out/'component_spec_gate.json',dict(passed=True,independent_component_draws=n,max_violation_W=max(0.,maximum),physical_sensor_calibration_performed=False));print('COMPONENT SPECIFICATION CHECKS PASSED',n)

if __name__=='__main__':main(ROOT/'results/robust_validation')
