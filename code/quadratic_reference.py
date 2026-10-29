"""Exact integral of the declared piecewise-linear native power, not calibration."""
import numpy as np

class Reference:
    def __init__(self,t,p):
        self.t=np.asarray(t,float);self.p=np.asarray(p,float);assert np.all(np.diff(self.t)>0) and np.all(self.p>=0)
        self.energy=np.r_[0.,np.cumsum(np.diff(self.t)*(self.p[:-1]+self.p[1:])/7200)]
    def value(self,x):
        x=np.asarray(x,float);j=np.clip(np.searchsorted(self.t,x,side='right')-1,0,len(self.t)-2);u=x-self.t[j];dt=self.t[j+1]-self.t[j]
        return self.energy[j]+(self.p[j]*u+(self.p[j+1]-self.p[j])*u*u/(2*dt))/3600
    def error(self,t,curve):
        t=np.asarray(t,float);curve=np.asarray(curve,float);grid=np.unique(np.r_[self.t,t]);left=grid[:-1];right=grid[1:];middle=(left+right)/2;half=(right-left)/2;v=np.sqrt(3/5)
        locations=np.array([middle-half*v,middle,middle+half*v]);difference=np.interp(locations,t,curve)-self.value(locations);squared=float(np.sum(half*(5*difference[0]**2+8*difference[1]**2+5*difference[2]**2)/9)/(grid[-1]-grid[0]))
        d0=np.interp(grid,t,curve)-self.value(grid);slopes=np.diff(np.interp(grid,t,curve))/np.diff(grid);p0=np.interp(left,self.t,self.p);p1=np.interp(right,self.t,self.p);delta=p1-p0
        offset=np.zeros_like(delta);np.divide((3600*slopes-p0)*(right-left),delta,out=offset,where=delta!=0);inside=(delta!=0)&(offset>0)&(offset<right-left);critical=left[inside]+offset[inside]
        peak=max(float(np.max(abs(d0))),float(np.max(abs(np.interp(critical,t,curve)-self.value(critical)))) if len(critical) else 0.)
        return float(np.sqrt(max(0.,squared))),peak
    def linearization_budget(self):
        dt=np.diff(self.t);dp=np.diff(self.p);T=self.t[-1]-self.t[0]
        return dict(peak_Wh=float(np.max(abs(dp)*dt/28800)),RMSE_Wh=float(np.sqrt(np.sum(dp*dp*dt**3)/(120*3600**2*T))))
