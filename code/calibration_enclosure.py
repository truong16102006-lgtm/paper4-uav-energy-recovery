"""Conservative energy boxes for bounded gain/bias before counter integration."""
import numpy as np

def enclosure(a,b,y,cap_W,epsilon,relative_gain=0.,bias_W=0.):
    a=np.asarray(a,float);b=np.asarray(b,float);y=np.asarray(y,float);eps=np.broadcast_to(np.asarray(epsilon,float),y.shape)
    assert y.shape==a.shape==b.shape and len(y)>=2 and np.isfinite(np.r_[a,b,y,eps]).all()
    assert np.all(eps>=0) and cap_W>=0 and 0<=relative_gain<1 and bias_W>=0
    elapsed=b-a[0];assert a[0]==b[0] and np.all(elapsed>=0)
    z=y-y[0];error=eps+eps[0]+bias_W*elapsed/3600
    lower=np.maximum(0,(z-error)/(1+relative_gain));upper=np.minimum(cap_W*elapsed/3600,(z+error)/(1-relative_gain));lower[0]=upper[0]=0.
    if np.any(lower>upper+1e-10):return dict(status='CALIBRATION_CONTRACT_INCOMPATIBLE',lower=lower,upper=upper)
    upper=np.maximum(upper,lower)
    return dict(status='CONDITIONAL_CALIBRATION_RELAXATION',centers=(lower+upper)/2,epsilon=(upper-lower)/2,lower=lower,upper=upper,relative_power_gain_bound=float(relative_gain),additive_power_bias_bound_W=float(bias_W),sharpness_scope='Rectangular energy-box relaxation, not the exact correlated calibration model')

def distorted_counter(ref,query_times,gain_bound,bias_bound_W,reading_error):
    left=ref.t[:-1];dt=np.diff(ref.t);middle=(left+ref.t[1:])/2;T=ref.t[-1]-ref.t[0]
    gain=gain_bound*np.sin(2*np.pi*(middle-ref.t[0])/T);bias=bias_bound_W*np.cos(2*np.pi*(middle-ref.t[0])/T)
    p0=(1+gain)*ref.p[:-1]+bias;p1=(1+gain)*ref.p[1:]+bias;prefix=np.r_[0.,np.cumsum((p0+p1)*dt/7200)]
    query_times=np.asarray(query_times,float);j=np.clip(np.searchsorted(ref.t,query_times,side='right')-1,0,len(dt)-1);u=query_times-ref.t[j]
    values=prefix[j]+(p0[j]*u+(p1[j]-p0[j])*u*u/(2*dt[j]))/3600
    return values+reading_error*(2*(query_times-ref.t[0])/T-1)
