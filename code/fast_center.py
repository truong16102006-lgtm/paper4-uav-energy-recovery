"""Exact fixed-origin center projection by bounded-slope envelope clipping."""
import numpy as np,time
from review6_envelopes_v2 import envelopes,evaluate,prepare

def norm(t,error):
    d=np.diff(t);return float(np.sqrt(max(0.,np.sum(d*(error[:-1]**2+error[:-1]*error[1:]+error[1:]**2)/3)/(t[-1]-t[0]))))

def crossing_grid(t,functions):
    """Every crossing of finitely many affine branches, O(n) with fixed branches."""
    extra=[];width=np.diff(t)
    for i in range(len(functions)):
        for j in range(i):
            difference=functions[i]-functions[j];left=difference[:-1];right=difference[1:];den=left-right
            valid=(den!=0)&((left*right)<0)
            if valid.any():extra.append(t[:-1][valid]+width[valid]*left[valid]/den[valid])
    return np.unique(np.r_[t,*extra])

def recover_clipped(t,prior,a,b,y,cap,epsilon,risk_factor=1,origin=None):
    begin=time.perf_counter();assert risk_factor>=1
    env=envelopes(a,b,y,cap,epsilon)
    if env['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':return dict(status=env['status'],envelope=env)
    aa,bb,l,u,m,t0,y0,*_=prepare(a,b,y,cap,epsilon);t=np.asarray(t,float)-t0;p=np.asarray(prior,float)
    assert t[0]==0 and t[-1]==aa[-1] and np.all(np.diff(t)>0) and p.shape==t.shape and np.isfinite(p).all()
    base=np.unique(np.r_[t,env['knots_elapsed_s'],aa,bb]);raw=np.interp(base,t,p-p[0]);dt=np.diff(base);inc=np.diff(raw);q=np.r_[0.,np.cumsum(np.clip(inc,0,m*dt))]
    repair=norm(base,q-raw);LE,UE=evaluate(env,base);L=np.interp(base,env['knots_elapsed_s'],env['lower_counter_centered_Wh']);U=np.interp(base,env['knots_elapsed_s'],env['upper_counter_centered_Wh']);R=env['minimax_radius_Wh'];A=UE-risk_factor*R;B=LE+risk_factor*R
    origin_low=max(float(L[0]),float(np.max(L-B)));origin_high=min(float(U[0]),float(np.min(U-A)));assert origin_low<=origin_high+1e-8
    c=(L[0]+U[0])/2 if origin is None else float(origin);assert origin_low-1e-8<=c<=origin_high+1e-8
    functions=np.array([np.zeros(len(base)),L-c,A,m*base,U-c,B,q]);grid=crossing_grid(base,functions);g=np.array([np.interp(grid,base,z) for z in functions]);lower=np.max(g[:3],axis=0);upper=np.min(g[3:6],axis=0);curve=np.maximum(lower,np.minimum(g[6],upper));lo,hi=evaluate(env,grid)
    assert np.max(lower-upper)<1e-8
    absolute=curve+c;d=np.diff(curve);feasible=max(0.,abs(curve[0]),float(np.max(-d)),float(np.max(d-m*np.diff(grid))),float(np.max(np.interp(aa,grid,absolute)-u)),float(np.max(l-np.interp(bb,grid,absolute))))
    risk=max(0.,float(np.max(hi-risk_factor*R-curve)),float(np.max(curve-lo-risk_factor*R)));assert max(feasible,risk)<1e-8
    return dict(status='CHECKED_CLIPPED_MINIMAX_CENTER',grid=grid,increment=curve,prior_repaired=np.interp(grid,base,q),prior_raw=np.interp(grid,base,raw),midpoint=(lo+hi)/2,lower=lo,upper=hi,origin_centered_Wh=float(c),origin_interval_Wh=[origin_low,origin_high],radius_Wh=float(R),risk_factor=float(risk_factor),conditional_worst_Wh=float(np.max(np.maximum(curve-lo,hi-curve))),primal_residual_Wh=feasible,risk_residual_Wh=risk,prior_slope_repair_RMSE_Wh=repair,prior_distance_Wh=norm(grid,curve-np.interp(grid,base,q)),knots=len(grid),seconds=time.perf_counter()-begin)
