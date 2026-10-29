"""Ordered closed-time-box counter recovery; separate from native power integrals."""
import numpy as np


def prepare(a,b,y,cap,epsilon):
    a,b,y=map(lambda z:np.asarray(z,float),(a,b,y));e=np.broadcast_to(np.asarray(epsilon,float),y.shape)
    if a.ndim!=1 or len(a)<2 or a.shape!=b.shape or a.shape!=y.shape:raise ValueError('At least two ordered observation boxes required')
    if not all(np.isfinite(z).all() for z in [a,b,y,e]) or not np.isfinite(cap) or cap<0 or (e<0).any():raise ValueError('Finite data and nonnegative cap/errors required')
    if (a>b).any() or (np.diff(a)<0).any() or (np.diff(b)<0).any():raise ValueError('Both time-bound lists must be nondecreasing and lower <= upper')
    if a[0]!=b[0] or a[-1]!=b[-1] or a[-1]<=a[0]:raise ValueError('Exact first/last observation times and positive horizon required')
    t0,y0=float(a[0]),float(y[0]);a,b=a-t0,b-t0;l,u=y-y0-e,y-y0+e
    l=np.maximum.accumulate(l);u=np.minimum.accumulate(u[::-1])[::-1];m=cap/3600
    pl=np.r_[-np.inf,np.maximum.accumulate(l)];sl=np.r_[np.maximum.accumulate((l-m*b)[::-1])[::-1],-np.inf]
    pu=np.r_[np.inf,np.minimum.accumulate(u-m*a)];su=np.r_[np.minimum.accumulate(u[::-1])[::-1],np.inf]
    def absolute(t):
        t=np.asarray(t,float);j=np.searchsorted(b,t,side='right');k=np.searchsorted(a,t,side='right')
        return np.maximum(pl[j],m*t+sl[j]),np.minimum(m*t+pu[k],su[k])
    lower_at_a=absolute(a)[0];residual=float(max(0.,np.max(l-u),np.max(lower_at_a-u)))
    return a,b,l,u,m,t0,y0,pl,sl,pu,su,absolute,residual


def feasibility_residual(a,b,y,cap,epsilon):return prepare(a,b,y,cap,epsilon)[-1]


def envelopes(a,b,y,cap,epsilon):
    a,b,l,u,m,t0,y0,pl,sl,pu,su,absolute,residual=prepare(a,b,y,cap,epsilon)
    base=dict(readings=len(y),cap_W=float(cap),time_origin_s=t0,counter_offset_Wh=y0,
              max_feasibility_residual_Wh=residual,decision_tolerance_Wh=1e-8,
              clock_contract='one nondecreasing clock in closed ordered time boxes; exact horizon endpoints')
    if residual>0:return dict(**base,status='CONTRACT_INCOMPATIBLE' if residual>1e-8 else 'NUMERIC_BOUNDARY')
    L0,U0=map(float,absolute(0.));edges=np.unique(np.r_[a,b]);knots=[0.]
    for left,right in zip(edges[:-1],edges[1:]):
        mid=(left+right)/2;j=int(np.searchsorted(b,mid,side='right'));k=int(np.searchsorted(a,mid,side='right'))
        A,B,C,D=pl[j],sl[j],pu[k],su[k]
        Ai,Bi,Ci,Di=max(A-U0,0.),B-U0,min(C-L0,0.),D-L0
        crossings=[] if m==0 else [(A-B)/m,(D-C)/m,(Ai-Bi)/m,(Di-Ci)/m]
        knots.extend(sorted({float(z) for z in crossings if left<z<right}));knots.append(float(right))
    z=np.asarray(knots);lower,upper=absolute(z);lo=np.maximum(0.,lower-U0);hi=np.minimum(m*z,upper-L0)
    assert np.min(hi-lo)>-1e-8
    width=np.maximum(hi-lo,0.);mid=(lo+hi)/2;origin=(L0+U0)/2;counter=mid+origin
    # Earliest intersection with each tightened value box, bounded below by a_i.
    witness=[]
    for ai,bi,li,ui in zip(a,b,l,u):
        if np.interp(ai,z,counter)>=li-1e-12:x=float(ai)
        else:
            k=int(np.searchsorted(counter,li,side='left'))
            if k>=len(z):raise ArithmeticError('Counter witness outside numerical range')
            x=float(z[k-1]+(li-counter[k-1])*(z[k]-z[k-1])/(counter[k]-counter[k-1]))
            x=max(float(ai),x)
        witness.append(min(float(bi),x))
    clock=np.asarray(witness);values=np.interp(clock,z,counter)
    witness_residual=float(max(0.,np.max(a-clock),np.max(clock-b),np.max(l-values),np.max(values-u),np.max(-np.diff(clock))))
    assert witness_residual<1e-8
    return dict(**base,status='CONDITIONAL_CLOSED_CLOCK_ENVELOPE',volume_Wh_s=float(np.trapezoid(width,z)),
                minimax_radius_Wh=float(width.max()/2),knots_elapsed_s=z.tolist(),
                lower_increment_Wh=lo.tolist(),upper_increment_Wh=hi.tolist(),midpoint_increment_Wh=mid.tolist(),
                lower_counter_centered_Wh=lower.tolist(),upper_counter_centered_Wh=upper.tolist(),
                initial_counter_interval_centered_Wh=[L0,U0],
                midpoint_clock_witness_elapsed_s=clock.tolist(),midpoint_witness_residual_Wh_or_s=witness_residual,
                minimum_midpoint_clock_gap_s=float(np.diff(clock).min()))


def evaluate(result,t):
    t=np.asarray(t,float);z=result['knots_elapsed_s']
    if (t<z[0]).any() or (t>z[-1]).any():raise ValueError('Outside fixed horizon')
    return np.interp(t,z,result['lower_increment_Wh']),np.interp(t,z,result['upper_increment_Wh'])
