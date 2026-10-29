"""Independent dense fixed-origin quadratic comparison and exact-time reduction."""
from common4 import *
from integrated_core import boxes
from minimax_centers import CenterProblem
from fast_center import recover_clipped,norm
from scipy.optimize import minimize,LinearConstraint
from scipy.linalg import null_space
import numpy as np,pandas as pd,time

def dense_fixed(c,p,origin,k):
    eye=np.eye(c.n+1);store={}
    def add(row,lo,hi):
        first=np.flatnonzero(row)[0]
        if row[first]<0:row=-row;lo,hi=-hi,-lo
        key=tuple(row);old=store.get(key,(-np.inf,np.inf));store[key]=(max(old[0],lo),min(old[1],hi))
    for i,dt in enumerate(np.diff(c.t)):add(eye[i+1]-eye[i],0,c.m*dt/c.scale)
    for ai,bi,li,ui in zip(c.a,c.b,c.l,c.u):
        add(eye[np.searchsorted(c.t,ai)]+eye[-1],-np.inf,ui/c.scale);add(eye[np.searchsorted(c.t,bi)]+eye[-1],li/c.scale,np.inf)
    add(eye[0],0,0);add(eye[-1],origin/c.scale,origin/c.scale)
    for i in range(c.n):add(eye[i],(c.upper[i]-k*c.R)/c.scale,(c.lower[i]+k*c.R)/c.scale)
    A=np.array(list(store));lo=np.array([x[0] for x in store.values()]);hi=np.array([x[1] for x in store.values()]);eq=abs(hi-lo)<1e-12;B=A[eq];d=(hi[eq]+lo[eq])/2;x0=np.linalg.lstsq(B,d,rcond=None)[0];Z=null_space(B)
    dt=np.diff(c.t)/(c.t[-1]-c.t[0])
    def objective(x):
        e=x[:-1]-p/c.scale;return float(np.sum(dt*(e[:-1]**2+e[:-1]*e[1:]+e[1:]**2)/3))
    if Z.shape[1]==0:return objective(x0)
    M=A[~eq]@Z;lower=lo[~eq]-A[~eq]@x0;upper=hi[~eq]-A[~eq]@x0;keep=np.linalg.norm(M,axis=1)>1e-12;assert np.all(lower[~keep]<=1e-8) and np.all(upper[~keep]>=-1e-8)
    initial=Z.T@(np.r_[c.mid/c.scale,origin/c.scale]-x0);constraints=[LinearConstraint(M[keep],lower[keep],upper[keep])] if keep.any() else []
    def fun(v):return objective(x0+Z@v)
    result=minimize(fun,initial,method='SLSQP',constraints=constraints,options={'maxiter':3000,'ftol':1e-13});assert result.success,result.message
    return result.fun

def main(output):
    rng=np.random.default_rng(2026102706);rows=[];started=time.perf_counter();max_obj=0.;max_curve=0.;maxrisk=0.
    for trial in range(200):
        t=np.r_[0.,np.cumsum(rng.uniform(.2,2.,rng.integers(3,10)))];cap=float(rng.choice([0,360,3600]));m=cap/3600;truth=np.r_[0,np.cumsum(rng.uniform(0,m,len(t)-1)*np.diff(t))];eps=float(rng.choice([0,.01,.1]));h=float(rng.choice([0,.1,.5]));a,b=boxes(t,h);y=truth+rng.uniform(-eps,eps,len(t));prior=np.r_[0,np.cumsum(rng.uniform(-m*.2,m*1.2,len(t)-1)*np.diff(t))]
        for k in [1.,2.]:
            got=recover_clipped(t,prior,a,b,y,cap,eps,k);assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER';grid=got['grid'];c=CenterProblem(grid,a,b,y,cap,eps);p=np.interp(c.t,grid,got['prior_repaired']);E=np.interp(c.t,grid,got['increment']);origin=got['origin_centered_Wh'];W=c.W.toarray()
            value=(norm(c.t,E-p)/c.scale)**2;dense=dense_fixed(c,p,origin,k);error=abs(value-dense);max_obj=max(max_obj,error);assert error<1e-7,(trial,k,error)
            clipped=np.interp(c.t,grid,got['increment']);gradient=2*(W@(clipped-p));mid=c.mid
            assert float(gradient@(mid-clipped))>=-1e-7
            maxrisk=max(maxrisk,got['risk_residual_Wh']);assert got['conditional_worst_Wh']<=k*c.R+1e-8
            if eps==0:
                free,diag=c.solve(p,k)
                if diag['optimized']:
                    diff=norm(c.t,free-clipped);max_curve=max(max_curve,diff);assert diff<2e-6,(trial,k,diff)
            rows.append(dict(mechanism=trial,risk_factor=k,objective_difference=error,primal_residual_Wh=got['primal_residual_Wh'],risk_residual_Wh=got['risk_residual_Wh'],slope_repair_Wh=got['prior_slope_repair_RMSE_Wh']))
    output.mkdir(exist_ok=True,parents=True);pd.DataFrame(rows).to_csv(output/'fast_validation_cases.csv',index=False);report=dict(passed=True,mechanisms=200,dense_fixed_origin_comparisons=400,max_objective_difference=max_obj,max_exact_origin_free_QP_curve_difference_Wh=max_curve,max_risk_residual_Wh=maxrisk,seconds=time.perf_counter()-started);write(output/'fast_validation_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/fast_validation');a=p.parse_args();main(ROOT/a.output)
