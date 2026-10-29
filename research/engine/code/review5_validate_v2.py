"""Independent LP and metamorphic validation of continuous increment recovery."""
from review5_common import ROOT,write,check_registration
from review5_envelopes_v1 import envelopes,evaluate
import argparse,json
import numpy as np
from scipy.optimize import linprog


def solve(t,y,e,cap,probe=None,sign=1):
    # Independent actual-time LP: no extremal-envelope formula or prefix scan.
    grid=np.unique(np.r_[t,[] if probe is None else [probe]])
    n=len(grid);rows=[];rhs=[]
    for a,b in zip(range(n-1),range(1,n)):
        row=np.zeros(n);row[b]=1;row[a]=-1
        rows.extend([row,-row]);rhs.extend([cap*(grid[b]-grid[a])/3600,0.])
    for time,value,error in zip(t,y,e):
        k=int(np.searchsorted(grid,time));row=np.zeros(n);row[k]=1
        rows.extend([row,-row]);rhs.extend([value+error,-value+error])
    objective=np.zeros(n)
    if probe is not None:objective[np.searchsorted(grid,probe)]+=sign;objective[0]-=sign
    result=linprog(objective,A_ub=rows,b_ub=rhs,bounds=[(None,None)]*n,method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.status not in [0,2]:raise RuntimeError(result.message)
    return result


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026102705);solves=0;maxerror=0.;checks=0;incompatible=0;boundary=0
    example=envelopes([0,10],[0,5],3600,1.)
    assert abs(example['volume_Wh_s']-41)<1e-10 and abs(example['absolute_counter_volume_Wh_s']-45)<1e-10
    assert abs(example['minimax_radius_Wh']-3.5)<1e-10
    assert np.allclose(evaluate(example,[10]),[[3],[7]])
    for trial in range(400):
        n=int(rng.integers(3,13));t=np.r_[0.,np.cumsum(rng.uniform(.1,3,n-1))]
        cap=0. if trial%17==0 else float(rng.uniform(50,3600));m=cap/3600
        truth=np.r_[0.,np.cumsum(rng.uniform(0,m,n-1)*np.diff(t))]
        e=rng.uniform(.01,.5,n);e[rng.random(n)<.15]=0.
        y=truth+rng.uniform(-1,1,n)*e
        if trial%5==0:y[n//2]+=10.;e[:]=0.
        got=envelopes(t,y,cap,e);lp=solve(t,y,e,cap);solves+=1
        if not lp.success:
            assert got['status']=='CONTRACT_INCOMPATIBLE';incompatible+=1;continue
        if got['status']=='NUMERIC_BOUNDARY':boundary+=1;continue
        assert got['status']=='CONDITIONAL_ENVELOPE'
        points=t[-1]*np.array([.0,.137,.491,.839,1.])
        lower,upper=evaluate(got,points)
        for j,p in enumerate(points):
            low=solve(t,y,e,cap,float(p),1);high=solve(t,y,e,cap,float(p),-1);solves+=2
            assert low.success and high.success
            error=max(abs(lower[j]-low.fun),abs(upper[j]+high.fun));maxerror=max(maxerror,error)
            assert error<1e-8
        # Both extremal increment FUNCTIONS and their midpoint are feasible
        # after restoring the corresponding permitted origin value.
        knots=np.asarray(got['knots_elapsed_s']);lo,hi=evaluate(got,knots)
        L0,U0=got['initial_counter_interval_centered_Wh']
        for curve,origin in [(lo,U0),(hi,L0),((lo+hi)/2,(L0+U0)/2)]:
            assert np.min(np.diff(curve))>-1e-8
            assert np.max(np.diff(curve)-m*np.diff(knots))<1e-8
            values=np.interp(t,knots,curve)+origin+y[0]
            assert np.max(np.abs(values-y)-e)<1e-8
        # Heterogeneous box enlargement and a subset of readings.
        expanded=envelopes(t,y,cap,e+.1)
        subset=np.unique(np.r_[0,np.arange(0,n,2),n-1]);sparse=envelopes(t[subset],y[subset],cap,e[subset])
        assert expanded['volume_Wh_s']>=got['volume_Wh_s']-1e-8
        assert sparse['volume_Wh_s']>=got['volume_Wh_s']-1e-8
        shift=envelopes(t+1000,y+1000,cap,e)
        scale=envelopes(t,y*3,cap*3,e*3)
        assert abs(shift['volume_Wh_s']-got['volume_Wh_s'])<1e-7
        assert abs(scale['volume_Wh_s']-3*got['volume_Wh_s'])<1e-7
        exact=envelopes(t,truth,cap,0.)
        E,T=np.diff(truth),np.diff(t)
        expected=0. if cap==0 else float(np.sum(E*(T-3600*E/cap)))
        assert abs(exact['volume_Wh_s']-expected)<1e-8
        checks+=1
        if (trial+1)%100==0:print('RECOVERY VALIDATION',trial+1,'/400',flush=True)
    result=dict(passed=True,synthetic_mechanisms=400,LP_solves=solves,feasible_mechanisms_checked=checks,
                incompatible_mechanisms=incompatible,numeric_boundary_mechanisms=boundary,
                max_probe_bound_difference_Wh=maxerror,analytic_increment_volume_Wh_s=example['volume_Wh_s'],
                analytic_absolute_counter_volume_Wh_s=example['absolute_counter_volume_Wh_s'],
                zero_error_reduction=True,heterogeneous_boxes=True,extremal_paths_and_midpoint_feasible=True,
                addition_and_error_monotonicity=True,offset_time_translation_and_scale_invariance=True)
    write(out/'envelope_validation.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review5');a=p.parse_args();main(ROOT/a.output)
