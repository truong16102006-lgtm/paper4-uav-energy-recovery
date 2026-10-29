"""Bounded local mass normalization using requested readings only."""
import numpy as np

def allocate_rates(rates,dt,target,cap):
    rates=np.maximum(np.asarray(rates,float),0.);dt=np.asarray(dt,float)
    assert np.all(dt>0) and np.isfinite(rates).all() and cap>=0
    target=float(np.clip(target,0,cap*dt.sum()));out=np.zeros_like(rates)
    if target==0 or cap==0:return out
    positive=rates>0;capacity=cap*dt[positive].sum()
    if not positive.any():return np.full_like(rates,target/dt.sum())
    if target>=capacity:
        out[positive]=cap
        if (~positive).any():out[~positive]=(target-capacity)/dt[~positive].sum()
        return out
    s=rates[positive]/rates[positive].max();w=dt[positive];order=np.argsort(-s);s=s[order];w=w[order]
    saturated=0.;remaining=float(s@w)
    for j in range(len(s)):
        scale=(target-saturated)/remaining
        if scale<=cap/s[j]:break
        saturated+=cap*w[j];remaining-=s[j]*w[j]
    out[positive]=np.minimum(cap,scale*rates[positive]/rates[positive].max())
    assert abs(float(out@dt)-target)<=1e-9*max(1.,target)
    return out

def adapt_local(t,prior,anchors,readings,cap_W):
    t=np.asarray(t,float);p=np.asarray(prior,float);anchors=np.asarray(anchors,float);y=np.asarray(readings,float)
    assert p.shape==t.shape and anchors.shape==y.shape and np.all(np.diff(t)>0) and np.all(np.diff(anchors)>0)
    assert t[0]==anchors[0] and t[-1]==anchors[-1]
    grid=np.unique(np.r_[t,anchors]);raw=np.interp(grid,t,p);increments=np.zeros(len(grid)-1);clamped=0;flat=0;m=cap_W/3600
    for j in range(len(anchors)-1):
        left=np.searchsorted(grid,anchors[j]);right=np.searchsorted(grid,anchors[j+1]);dt=np.diff(grid[left:right+1]);rates=np.diff(raw[left:right+1])/dt;wanted=float(y[j+1]-y[j]);target=float(np.clip(wanted,0,m*dt.sum()));clamped+=abs(target-wanted)>1e-10;flat+=not np.any(rates>0)
        increments[left:right]=allocate_rates(rates,dt,target,m)*dt
    result=np.r_[0.,np.cumsum(increments)]
    return grid,result,dict(clamped_nominal_intervals=int(clamped),flat_prior_intervals=int(flat),adaptation_uses_only_requested_values=True)

def distorted_prior(t,p,anchors,kind):
    # Insert the exact requested boundaries before any interval-wise distortion.
    grid=np.unique(np.r_[t,anchors]);raw=np.interp(grid,t,p);middle=(grid[:-1]+grid[1:])/2;u=(middle-grid[0])/(grid[-1]-grid[0])
    if kind=='identity':gain=np.ones(len(middle))
    elif kind=='global_2':gain=np.full(len(middle),2.)
    elif kind=='interval_gain':gain=np.where(np.searchsorted(anchors,middle,side='right')%2==1,.4,1.8)
    elif kind=='ramp':gain=.4+1.4*u
    elif kind=='smooth':gain=1+.8*np.sin(2*np.pi*u)
    else:raise ValueError(kind)
    return grid,np.r_[0.,np.cumsum(np.diff(raw)*gain)]
