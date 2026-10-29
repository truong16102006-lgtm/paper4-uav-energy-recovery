"""One scalar queried-data blend; no unrequested electrical reference values."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from local_mass import adapt_local

def hinge_weight(A,B):
    A=np.asarray(A,float);B=np.asarray(B,float);scale=max(1.,float(np.max(abs(A))) if len(A) else 0.,float(np.max(abs(B))) if len(B) else 0.);a=A/scale;b=B/scale
    if not len(A):return 0.,0.
    nonzero=b!=0;cross=-a[nonzero]/b[nonzero];knots=np.unique(np.r_[0.,cross[(cross>0)&(cross<1)],1.]);candidates=list(knots)
    for left,right in zip(knots[:-1],knots[1:]):
        active=a+b*(left+right)/2>0;den=float(b[active]@b[active])
        if den>0:
            w=-float(a[active]@b[active])/den
            if left<=w<=right:candidates.append(w)
    candidates=np.array(sorted(set(candidates)));scores=np.sum(np.maximum(0,a[:,None]+b[:,None]*candidates)**2,axis=0);index=int(np.argmin(scores));w=float(candidates[index]);return w,float(scores[index]*scale*scale)

def candidates(t,state,anchors,y,cap_W):
    # Every call receives an explicitly reduced anchor set for a held-out fold.
    base=np.unique(np.r_[t,anchors]);state=np.interp(base,t,state);interp=PchipInterpolator(anchors,y-y[0])(base)
    grid,p,_=adapt_local(base,interp,anchors,y,cap_W);g,h,_=adapt_local(base,state,anchors,y,cap_W);assert np.array_equal(grid,g)
    return grid,p,h

def select_prior(t,state,anchors,y,a,b,cap_W,epsilon):
    t=np.asarray(t,float);state=np.asarray(state,float);anchors=np.asarray(anchors,float);y=np.asarray(y,float);eps=np.broadcast_to(np.asarray(epsilon,float),y.shape)
    A=[];B=[];folds=[]
    for held in range(1,len(anchors)-1):
        keep=np.arange(len(anchors))!=held;grid,p,h=candidates(t,state,anchors[keep],y[keep],cap_W)
        pl,pu=np.interp([a[held],b[held]],grid,p);hl,hu=np.interp([a[held],b[held]],grid,h);lo=y[held]-y[0]-eps[held]-eps[0];hi=y[held]-y[0]+eps[held]+eps[0]
        A.extend([lo-pu,pl-hi]);B.extend([pu-hu,hl-pl]);folds.append(dict(held_index=held,fit_indices=np.flatnonzero(keep).tolist(),held_value_lower_Wh=float(lo),held_value_upper_Wh=float(hi)))
    weight,score=hinge_weight(A,B);grid,p,h=candidates(t,state,anchors,y,cap_W);curve=(1-weight)*p+weight*h
    endpoints=[float(np.sum(np.maximum(0,np.array(A)+w*np.array(B))**2)) for w in [0.,1.]]
    assert score<=min(endpoints)+1e-8*max(1.,min(endpoints))
    return grid,curve,dict(weight_on_historical_shape=weight,held_query_count=len(folds),validation_loss_Wh2=score,PCHIP_validation_loss_Wh2=endpoints[0],historical_validation_loss_Wh2=endpoints[1],no_interior_fallback=not folds,folds=folds)
