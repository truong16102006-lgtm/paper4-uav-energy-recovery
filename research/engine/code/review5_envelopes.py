"""Sharp continuous bounded-counter INCREMENT envelopes, fixed ordered times.

Classical extremal monotone/Lipschitz geometry, specialized to reading boxes.
Float64 implementation; not an outward-rounded numerical certificate.
"""
import numpy as np


def envelopes(t, y, cap, epsilon):
    t,y=np.asarray(t,float),np.asarray(y,float)
    e=np.broadcast_to(np.asarray(epsilon,float),y.shape).copy()
    if t.ndim!=1 or y.shape!=t.shape or len(t)<2:
        raise ValueError('At least two one-dimensional time/counter readings required')
    if not (np.isfinite(t).all() and np.isfinite(y).all() and np.isfinite(e).all()
            and np.isfinite(cap) and cap>=0 and (e>=0).all() and (np.diff(t)>0).all()):
        raise ValueError('Finite data, increasing times and nonnegative cap/errors required')
    # Center both coordinates before subtracting slope terms.
    time0,energy0=float(t[0]),float(y[0]);x=t-time0;v=y-energy0;m=cap/3600
    l,u=v-e,v+e;lq,uq=l-m*x,u-m*x
    pl=np.maximum.accumulate(l);puq=np.minimum.accumulate(uq)
    slq=np.maximum.accumulate(lq[::-1])[::-1];su=np.minimum.accumulate(u[::-1])[::-1]
    residual=float(max(0.,np.max(pl-u),np.max(lq-puq)))
    base=dict(readings=len(t),cap_W=float(cap),max_feasibility_residual_Wh=residual,
              time_origin_s=time0,counter_offset_Wh=energy0,decision_tolerance_Wh=1e-8)
    if residual>0:
        return dict(**base,status='CONTRACT_INCOMPATIBLE' if residual>1e-8 else 'NUMERIC_BOUNDARY')
    L0,U0=float(slq[0]),float(su[0]);knots=[0.]
    for k in range(len(x)-1):
        a,b,c,d=pl[k],slq[k+1],puq[k],su[k+1]
        A,B,C,D=max(a-U0,0.),b-U0,min(c-L0,0.),d-L0
        candidates=[] if m==0 else [(a-b)/m,(d-c)/m,(A-B)/m,(D-C)/m]
        knots.extend(sorted({float(z) for z in candidates if x[k]<z<x[k+1]}))
        knots.append(float(x[k+1]))
    z=np.asarray(knots);k=np.minimum(np.searchsorted(x,z,side='right')-1,len(x)-2)
    lower=np.maximum(pl[k],m*z+slq[k+1]);upper=np.minimum(m*z+puq[k],su[k+1])
    inc_lower=np.maximum(0.,lower-U0);inc_upper=np.minimum(m*z,upper-L0)
    width=inc_upper-inc_lower
    assert width.min()>-1e-8
    width=np.maximum(width,0.)
    volume=float(np.trapezoid(width,z));absolute_volume=float(np.trapezoid(upper-lower,z))
    return dict(**base,status='CONDITIONAL_ENVELOPE',volume_Wh_s=volume,
                absolute_counter_volume_Wh_s=absolute_volume,minimax_radius_Wh=float(width.max()/2),
                initial_counter_interval_centered_Wh=[L0,U0],
                knots_elapsed_s=z.tolist(),lower_increment_Wh=inc_lower.tolist(),
                upper_increment_Wh=inc_upper.tolist(),midpoint_increment_Wh=((inc_lower+inc_upper)/2).tolist(),
                lower_counter_centered_Wh=lower.tolist(),upper_counter_centered_Wh=upper.tolist())


def evaluate(result, elapsed):
    if result['status']!='CONDITIONAL_ENVELOPE':raise ValueError('No feasible envelope')
    points=np.asarray(elapsed,float);knots=np.asarray(result['knots_elapsed_s'])
    if (points<knots[0]).any() or (points>knots[-1]).any():raise ValueError('Outside observation horizon')
    return (np.interp(points,knots,result['lower_increment_Wh']),
            np.interp(points,knots,result['upper_increment_Wh']))
