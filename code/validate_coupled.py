"""Independent dense-state assembly and generating-mechanism checks."""
from common4 import *
from coupled_calibration import CoupledClass
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import time

def independent(t,anchors,y,cap,eps,gain,bias,k):
    n=len(t);A=[];b=[]
    # Independent model uses E and reading errors nu at all grid nodes, rather
    # than the production implementation's measured-counter state Q.
    ix=np.searchsorted(t,anchors);z=np.interp(t,anchors,y-y[0]);bounds=[(0,cap*x/3600) for x in t]+[(None,None)]*n;bounds[0]=(0,0)
    for j,index in enumerate(ix):bounds[n+index]=(-eps[j],eps[j])
    for j,dt in enumerate(np.diff(t)):
        v=np.zeros(2*n);v[j]=1;v[j+1]=-1;A.append(v);b.append(0);A.append(-v);b.append(cap*dt/3600)
        # Q=z-nu; all non-observation nu nodes are free.
        v=np.zeros(2*n);v[j]=1+gain;v[j+1]=-1-gain;v[n+j]=1;v[n+j+1]=-1;A.append(v);b.append(bias*dt/3600-(z[j+1]-z[j]))
        v=np.zeros(2*n);v[j]=-1+gain;v[j+1]=1-gain;v[n+j]=-1;v[n+j+1]=1;A.append(v);b.append(bias*dt/3600+(z[j+1]-z[j]))
    c=np.zeros(2*n);c[k]=1.;lo=linprog(c,A_ub=np.asarray(A),b_ub=b,bounds=bounds,method='highs');hi=linprog(-c,A_ub=np.asarray(A),b_ub=b,bounds=bounds,method='highs');assert lo.success and hi.success
    return lo.fun,-hi.fun

def main(out):
    rng=np.random.default_rng(2026102719);maximum=0.;risk=0.;residual=0.;fallbacks=0;start=time.perf_counter();midpoint_failures=0
    with threadpool_limits(limits=2):
        for trial in range(200):
            t=np.r_[0.,np.cumsum(rng.uniform(.3,3,8))];cap=float(rng.uniform(20,300));gain=float(rng.choice([0,.02,.1,.25]));bias=float(rng.choice([0,2,10]));p=rng.uniform(0,cap,8);dt=np.diff(t);truth=np.r_[0,np.cumsum(p*dt/3600)];g=rng.uniform(-gain,gain,8);beta=rng.uniform(-bias,bias,8);Q=np.r_[0,np.cumsum(((1+g)*p+beta)*dt/3600)];idx=np.array([0,2,4,6,8]);a=t[idx];eps=rng.uniform(0,.02,5);y=Q[idx]+rng.uniform(-eps,eps)+37;model=CoupledClass(a,y,cap,eps,gain,bias)
            lo,hi=model.extrema(t);k=int(rng.integers(len(t)));L,U=independent(t,a,y,cap,eps,gain,bias,k);maximum=max(maximum,abs(L-lo[k]),abs(U-hi[k]));assert maximum<1e-7
            assert model.membership(t,truth)
            got=model.recover(t,truth,grid_nodes=17);assert got['status']=='CHECKED_COUPLED_CENTER';fallbacks+=not got['shape_QP_optimized'];residual=max(residual,got['joint_witness']['primal_residual_Wh']);probe=np.unique(np.r_[t,got['t']]);error=np.max(abs(np.interp(probe,t,truth)-np.interp(probe,got['t'],got['E'])));risk=max(risk,float(error-got['continuous_radius_upper_Wh']));assert risk<1e-7
            assert got['optimality_gap_Wh']<=got['analytic_grid_gap_bound_Wh']+1e-7
            midpoint_failures+=not model.membership(t,(lo+hi)/2)
    a=np.array([0,1,2.]);bad=CoupledClass(a,[0,1,-10],100,0,.05,1);assert bad.recover(a,a)['status']=='COUPLED_CALIBRATION_INCOMPATIBLE'
    zero=CoupledClass(a,[5,5,5],0,0,0,0).recover(a,np.zeros(3));assert zero['status']=='CHECKED_COUPLED_CENTER' and zero['continuous_radius_upper_Wh']<1e-9
    try:CoupledClass(a,[0,1,2],100,.01,.05,1,[.1,.1,0]);raise RuntimeError('Uncertain current initial time accepted')
    except ValueError:pass
    report=dict(passed=True,independent_dense_LP_comparisons=200,generated_mechanisms=200,max_extrema_difference_Wh=maximum,max_true_risk_excess_Wh=max(0,risk),max_joint_primal_residual_Wh=residual,shape_QP_fallbacks=fallbacks,pointwise_midpoint_joint_membership_failures=midpoint_failures,zero_power_case=True,incompatible_reset_refused=True,seconds=time.perf_counter()-start);write(out/'validation_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main(ROOT/'results/coupled_validation')
