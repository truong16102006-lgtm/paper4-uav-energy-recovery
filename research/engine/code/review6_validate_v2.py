"""Independent LPs in clock/counter variables, enumerating the probe's position."""
from review6_common import ROOT,write,check_registration
from review6_envelopes_v2 import envelopes,evaluate
from review5_envelopes_v1 import envelopes as fixed
import numpy as np
from scipy.optimize import linprog
import argparse,json


def lp(a,b,y,e,cap,probe=None,slot=None,sign=1):
    n=len(y);size=2*n+(probe is not None);rows=[];rhs=[];m=cap/3600
    def add(entries,value):
        row=np.zeros(size)
        for k,v in entries:row[k]+=v
        rows.append(row);rhs.append(value)
    for i in range(n-1):
        add([(i,1),(i+1,-1)],0.)
        add([(n+i,1),(n+i+1,-1)],0.)
        add([(n+i+1,1),(n+i,-1),(i+1,-m),(i,m)],0.)
    objective=np.zeros(size);bounds=list(zip(a,b))+list(zip(y-e,y+e))
    if probe is not None:
        i=slot;j=i+1;q=2*n;bounds.append((None,None))
        add([(i,1)],probe);add([(j,-1)],-probe)
        add([(n+i,1),(q,-1)],0.);add([(q,1),(n+j,-1)],0.)
        add([(q,1),(n+i,-1),(i,m)],m*probe)
        add([(n+j,1),(q,-1),(j,-m)],-m*probe)
        objective[q]=sign;objective[n]=-sign
    result=linprog(objective,A_ub=rows,b_ub=rhs,bounds=bounds,method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.status not in [0,2]:raise RuntimeError(result.message)
    return result


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026102706)
    solves=0;feasible=0;incompatible=0;boundary=0;error=0.;maxfixed=0.
    for trial in range(200):
        n=int(rng.integers(3,8));t=np.r_[0.,np.cumsum(rng.uniform(.3,2,n-1))];T=t[-1]
        h=float(rng.uniform(.1,2.5));a=np.maximum(t-h,0);b=np.minimum(t+h,T);a[0]=b[0]=0.;a[-1]=b[-1]=T
        cap=0. if trial%19==0 else float(rng.uniform(200,3600));m=cap/3600
        truth=np.r_[0.,np.cumsum(rng.uniform(0,m,n-1)*np.diff(t))];e=rng.uniform(.01,.4,n)
        y=truth+rng.uniform(-1,1,n)*e
        if trial%5==0:y[n//2]+=20;e[:]=0.
        got=envelopes(a,b,y,cap,e);original=lp(a,b,y,e,cap);solves+=1
        if not original.success:
            assert got['status']=='CONTRACT_INCOMPATIBLE';incompatible+=1;continue
        if got['status']=='NUMERIC_BOUNDARY':boundary+=1;continue
        assert got['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE';feasible+=1
        for probe in [.239*T,.713*T]:
            values=[]
            for sign in [1,-1]:
                outcomes=[]
                for slot in range(n-1):
                    result=lp(a,b,y,e,cap,probe,slot,sign);solves+=1
                    if result.success:outcomes.append(result.fun)
                assert outcomes;values.append(min(outcomes))
            lower,upper=evaluate(got,[probe]);diff=max(abs(lower[0]-values[0]),abs(upper[0]+values[1]));error=max(error,float(diff));assert diff<1e-8
        precise=envelopes(t,t,y,cap,e);old=fixed(t,y,cap,e)
        if old['status']=='CONDITIONAL_ENVELOPE' and precise['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':
            diff=max(abs(precise['volume_Wh_s']-old['volume_Wh_s']),abs(precise['minimax_radius_Wh']-old['minimax_radius_Wh']))
            maxfixed=max(maxfixed,diff);assert diff<1e-8
            assert got['volume_Wh_s']>=precise['volume_Wh_s']-1e-8
        expanded=envelopes(a,b,y,cap,e+.1);assert expanded['volume_Wh_s']>=got['volume_Wh_s']-1e-8
        lo,hi=evaluate(got,t);assert np.max(lo-truth)<1e-8 and np.max(truth-hi)<1e-8
        if (trial+1)%50==0:print('TIME-BOX VALIDATION',trial+1,'/200',flush=True)
    report=dict(passed=True,synthetic_mechanisms=200,LP_solves=solves,feasible=feasible,incompatible=incompatible,
                numeric_boundary=boundary,maximum_probe_bound_difference_Wh=error,maximum_fixed_time_reduction_difference=maxfixed,
                overlapping_boxes=True,ordered_common_clock=True,midpoint_clock_witness_checked=True,
                time_and_error_expansion_checked=True,closed_clock_not_strict_half_open=True)
    write(out/'clock_envelope_validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review6');a=p.parse_args();main(ROOT/a.output)
