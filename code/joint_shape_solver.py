"""Scale-aware shape QP with a declared weighted-node-L1 LP fallback."""
import numpy as np,osqp,time
from scipy.sparse import bmat,csc_matrix,vstack,hstack,eye,coo_matrix
from minimax_centers import mass_matrix

def refine(model,t,lower,upper,radius,prior,seed,kind):
    start=time.perf_counter();n=len(t);S=model.scale;A,b,bounds=model.polytope(t);variables=len(bounds);W=mass_matrix(t);P=bmat([[2*W,None],[None,csc_matrix((variables-n,variables-n))]],format='csc');q=np.r_[-2*(W@(prior/S)),np.zeros(variables-n)];lo=np.array([-np.inf if x[0] is None else x[0] for x in bounds]);hi=np.array([np.inf if x[1] is None else x[1] for x in bounds]);lo[:n]=np.maximum(lo[:n],(upper-radius)/S);hi[:n]=np.minimum(hi[:n],(lower+radius)/S);reverse=lo>hi
    if reverse.any():assert np.max(lo[reverse]-hi[reverse])*S<1e-8;mid=(lo[reverse]+hi[reverse])/2;lo[reverse]=mid;hi[reverse]=mid
    B=vstack([A,eye(variables,format='csc')],format='csc');low=np.r_[np.full(len(b),-np.inf),lo];high=np.r_[b,hi];statuses=[];x=seed;mode='checked_original'
    def check(x):
        residual=max(0.,float(np.max(A@x-b)),float(np.max(lo-x)),float(np.max(x-hi)));energy=x[:n]*S;risk=float(np.max(np.maximum(upper-energy,energy-lower)));return residual*S<2e-8 and risk<=radius+2e-8
    for precision,iterations in [(1e-10,100000),(1e-11,200000)]:
        solver=osqp.OSQP();solver.setup(P=P,q=q,A=B,l=low,u=high,verbose=False,eps_abs=min(precision,precision/S),eps_rel=precision/10,max_iter=iterations,polishing=True,adaptive_rho=True);solver.warm_start(x=seed);got=solver.solve(raise_error=False);statuses.append(got.info.status)
        if got.x is not None and got.info.status_val==1 and got.info.dual_res<=1e-7 and check(got.x):x=got.x;mode='continuous_L2_QP';break
    if mode!='continuous_L2_QP':
        # Explicit surrogate: trapezoidal node weights times absolute node error.
        # This is not asserted to be an exact continuous L1 or L2 objective.
        dt=np.diff(t)/(t[-1]-t[0]);weights=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2];base=hstack([A,csc_matrix((len(b),n))],format='csc');rr=np.r_[np.arange(n),np.arange(n),np.arange(n,2*n),np.arange(n,2*n)];cc=np.r_[np.arange(n),variables+np.arange(n),np.arange(n),variables+np.arange(n)];vv=np.r_[np.ones(n),-np.ones(n),-np.ones(n),-np.ones(n)];absrows=coo_matrix((vv,(rr,cc)),shape=(2*n,variables+n)).tocsc();AA=vstack([base,absrows],format='csc');bb=np.r_[b,prior/S,-prior/S];objective=np.r_[np.zeros(variables),weights];lp=model.lp(objective,AA,bb,list(zip(lo,hi))+[(0,None)]*n);statuses.append('weighted_L1_LP:'+lp.message)
        if lp.success and check(lp.x[:variables]):x=lp.x[:variables];mode='weighted_node_L1_LP'
    assert check(x),'No checked fallback survived the original tube'
    E=x[:n]*S
    if kind=='coupled_calibration':Q=x[n:2*n]*S;state=dict(Q=Q,witness=model.witness(t,E,Q))
    else:
        E,R,Q,v,w,z=model.decode(t,x);state=dict(Q=Q,R=R,witness=model.check(t,x))
    return dict(E=E,**state,mode=mode,statuses=statuses,seconds=time.perf_counter()-start,numerical_risk_excess_Wh=max(0.,float(np.max(np.maximum(upper-E,E-lower)))-radius))
