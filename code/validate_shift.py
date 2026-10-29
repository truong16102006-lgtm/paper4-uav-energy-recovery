"""Independent scalar-allocation and polynomial-reference checks."""
from common4 import *
from local_mass import allocate_rates,adapt_local,distorted_prior
from quadratic_reference import Reference
from scipy.optimize import brentq
from scipy.integrate import quad
import numpy as np,time

def main(output):
    output.mkdir(exist_ok=True,parents=True);rng=np.random.default_rng(2026102709);maximum=0.;invariance=0.;integration=0.;started=time.perf_counter()
    for trial in range(400):
        n=int(rng.integers(3,25));dt=rng.uniform(.1,3,n);cap=float(rng.uniform(.01,2));s=rng.uniform(0,3,n);s[rng.random(n)<.2]=0;target=float(rng.uniform(0,cap*dt.sum()));got=allocate_rates(s,dt,target,cap);positive=s>0;capacity=cap*dt[positive].sum()
        if not positive.any():expected=np.full(n,target/dt.sum())
        elif target>=capacity:
            expected=np.where(positive,cap,0.);expected[~positive]=(target-capacity)/dt[~positive].sum()
        else:
            root=brentq(lambda z:float(np.minimum(cap,z*s)@dt-target),0,max(cap/max(s[positive].min(),1e-300),1),xtol=1e-13);expected=np.minimum(cap,root*s)
        maximum=max(maximum,float(np.max(abs(got-expected))));assert maximum<1e-9
        t=np.r_[0.,np.cumsum(dt)];anchors=t[np.unique(np.r_[0,n//3,2*n//3,n])];prior=np.r_[0.,np.cumsum(s*dt)];y=np.r_[0.,np.cumsum(rng.uniform(0,cap,len(anchors)-1)*np.diff(anchors))]
        g,q,_=adapt_local(t,prior,anchors,y,cap*3600);tt,pp=distorted_prior(t,prior,anchors,'interval_gain');g2,q2,_=adapt_local(tt,pp,anchors,y,cap*3600);invariance=max(invariance,float(np.max(abs(np.interp(g2,g,q)-q2))));assert invariance<1e-9
        power=rng.uniform(0,500,len(t));ref=Reference(t,power);estimate=np.r_[0.,np.cumsum(rng.uniform(0,.1,len(dt))*dt)];exact,_=ref.error(t,estimate)
        numerical=sum(quad(lambda x:(float(np.interp(x,t,estimate))-float(ref.value(x)))**2,a,b,epsabs=1e-12)[0] for a,b in zip(t[:-1],t[1:]));numerical=np.sqrt(numerical/(t[-1]-t[0]));integration=max(integration,abs(exact-numerical));assert integration<1e-10
        budget=ref.linearization_budget();linear,_=ref.error(t,ref.energy);assert abs(linear-budget['RMSE_Wh'])<1e-10
    assert np.all(allocate_rates([0,0],[1,2],5,0)==0)
    _,edge,diagnostic=adapt_local([0,1,2,3],[0,0,0,0],[0,1,2,3],[0,-1,5,5],3600)
    assert np.allclose(edge,[0,0,1,1]) and diagnostic['clamped_nominal_intervals']==2 and diagnostic['flat_prior_intervals']==3
    write(output/'validation_gate.json',dict(passed=True,mechanisms=400,allocation_root_comparisons=400,quadrature_comparisons=400,max_rate_difference=maximum,max_interval_gain_invariance_Wh=invariance,max_integration_RMSE_difference_Wh=integration,seconds=time.perf_counter()-started));print('SHIFT VALIDATION PASSED',maximum,invariance,integration)

if __name__=='__main__':main(ROOT/'results/shift_validation')
