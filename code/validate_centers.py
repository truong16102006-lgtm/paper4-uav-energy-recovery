"""Independent dense SLSQP validation of the center-class sparse QP."""
from common4 import *
from integrated_core import boxes
from minimax_centers import CenterProblem,l2_error
import numpy as np,time
from scipy.optimize import minimize,LinearConstraint
from scipy.linalg import null_space

def independent_dense(c,prior,k):
    # Assemble constraints independently in energy increments and one origin.
    n=len(c.t);eye=np.eye(n+1);store={}
    def add(row,lo,hi):
        first=np.flatnonzero(row)[0]
        if row[first]<0:row=-row;lo,hi=-hi,-lo
        key=tuple(row);old=store.get(key,(-np.inf,np.inf));store[key]=(max(old[0],lo),min(old[1],hi))
    for i,dt in enumerate(np.diff(c.t)):add(eye[i+1]-eye[i],0,c.m*dt/c.scale)
    for ai,bi,li,ui in zip(c.a,c.b,c.l,c.u):
        add(eye[np.searchsorted(c.t,ai)]+eye[-1],-np.inf,ui/c.scale);add(eye[np.searchsorted(c.t,bi)]+eye[-1],li/c.scale,np.inf)
    add(eye[0],0,0)
    for i in range(n):add(eye[i],(c.upper[i]-k*c.R)/c.scale,(c.lower[i]+k*c.R)/c.scale)
    A=np.array(list(store));lower=np.array([z[0] for z in store.values()]);upper=np.array([z[1] for z in store.values()]);eq=abs(upper-lower)<1e-12;B=A[eq];d=(upper[eq]+lower[eq])/2
    x0=np.linalg.lstsq(B,d,rcond=None)[0] if len(B) else np.zeros(n+1);Z=null_space(B) if len(B) else np.eye(n+1);p=prior/c.scale
    dt=np.diff(c.t)/(c.t[-1]-c.t[0])
    def full_objective(x):
        e=x[:-1]-p;return float(np.sum(dt*(e[:-1]**2+e[:-1]*e[1:]+e[1:]**2)/3))
    if not Z.shape[1]:return full_objective(x0),'unique affine completion'
    inequality=A[~eq];matrix=inequality@Z;lower=lower[~eq]-inequality@x0;upper=upper[~eq]-inequality@x0;keep=np.linalg.norm(matrix,axis=1)>1e-12;assert np.all(lower[~keep]<=1e-8) and np.all(upper[~keep]>=-1e-8)
    initial=Z.T@(np.r_[c.mid/c.scale,c.origin/c.scale]-x0)
    constraints=[LinearConstraint(matrix[keep],lower[keep],upper[keep])] if keep.any() else []
    def fun(v):return full_objective(x0+Z@v)
    result=minimize(fun,initial,method='SLSQP',constraints=constraints,options={'maxiter':3000,'ftol':1e-13})
    assert result.success,result.message
    return result.fun,result.message

def main():
    rng=np.random.default_rng(2026102705);rows=[];maxerr=0.;maxstab=0.;start=time.perf_counter();fallbacks=0
    for trial in range(100):
        t=np.r_[0.,np.cumsum(rng.uniform(.2,2.,rng.integers(3,10)))];cap=float(rng.choice([0,360,3600]));m=cap/3600;truth=np.r_[0,np.cumsum(rng.uniform(0,m,len(t)-1)*np.diff(t))];eps=float(rng.choice([0,.01,.1]));h=float(rng.choice([0,.1,.5]));a,b=boxes(t,h);y=truth+rng.uniform(-eps,eps,len(t));c=CenterProblem(t,a,b,y,cap,eps);prior=np.interp(c.t,t,truth)+rng.normal(0,.2,len(c.t));prior[0]=0
        for k in [1,2]:
            E,d=c.solve(prior,k);fallbacks+=not d['optimized'];assert d['optimized'],d
            dense,message=independent_dense(c,prior,k)
            value=(l2_error(c.t,E-prior)/c.scale)**2;diff=abs(value-dense);maxerr=max(maxerr,diff);assert diff<1e-7,(trial,k,message,diff)
            # Standard projection inequality with a feasible comparator (the midpoint).
            lhs=l2_error(c.t,E-prior)**2+l2_error(c.t,E-c.mid)**2;rhs=l2_error(c.t,prior-c.mid)**2;assert lhs<=rhs+1e-7
            perturb=rng.normal(0,.01,len(prior));D,e=c.solve(prior+perturb,k);stab=l2_error(c.t,E-D)-l2_error(c.t,perturb);maxstab=max(maxstab,stab);assert stab<1e-7
            identity,di=c.solve(c.mid,k);assert l2_error(c.t,identity-c.mid)<1e-7
            assert d['conditional_worst_Wh']<=k*c.R+1e-8
            rows.append(dict(trial=trial,risk_factor=k,objective_difference=diff,stability_excess=stab))
    import pandas as pd
    out=ROOT/'results/new_centers_validation';out.mkdir(exist_ok=True,parents=True);pd.DataFrame(rows).to_csv(out/'independent_QP_checks.csv',index=False)
    report=dict(passed=True,mechanisms=100,dense_SLSQP_comparisons=200,sparse_QP_solves=600,maximum_objective_difference=maxerr,maximum_nonexpansiveness_excess_Wh=maxstab,midpoint_identity=True,exact_continuous_mass_matrix=True,fallbacks=fallbacks,seconds=time.perf_counter()-start);write(out/'centers_validation_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
