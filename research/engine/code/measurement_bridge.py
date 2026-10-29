"""Coupled energy-duration margin. Classical chain-polytope optimization."""
import numpy as np
from scipy.optimize import linprog

def energy_coefficients(power):
 p=np.asarray(power,float);w=(p[:-1]+p[1:])/7200.;c=np.zeros(len(p));c[:-1]-=w;c[1:]+=w
 return c

def linear_range(c,bins,h=1.):
 c=np.asarray(c,float);b=np.asarray(bins,dtype=np.int64)
 assert len(c)==len(b)>=2 and np.isfinite(c).all() and h>0 and np.all(np.diff(b)>=0)
 q=(b-b[0])*h;starts=np.r_[0,np.flatnonzero(np.diff(b))+1];ends=np.r_[starts[1:],len(b)];prefix=np.r_[0.,np.cumsum(c)];hi=np.maximum(np.maximum.reduceat(prefix[:-1],starts),prefix[ends]);lo=np.minimum(np.minimum.reduceat(prefix[:-1],starts),prefix[ends]);base=float(c@q)
 return base+h*float(np.sum(prefix[ends]-hi)),base+h*float(np.sum(prefix[ends]-lo))

def margin_coefficients(power,cap,kappa=1.):
 c=kappa*energy_coefficients(power);c[0]+=cap/3600.;c[-1]-=cap/3600.
 return c

def common_clock(power,bins,idx,cap,kappa=1.,h=1.):
 p=np.asarray(power,float);b=np.asarray(bins,dtype=np.int64);n=len(p);q=(b-b[0])*h
 order=np.zeros((n-1,n));order[np.arange(n-1),np.arange(n-1)]=1;order[np.arange(n-1),np.arange(1,n)]=-1
 margins=[]
 for a,z in zip(idx[:-1],idx[1:]):
  c=np.zeros(n);c[a:z+1]=margin_coefficients(p[a:z+1],cap,kappa);margins.append(c)
 margins=np.array(margins);A=np.r_[np.c_[order,np.zeros(n-1)],np.c_[margins,-np.ones(len(margins))]];rhs=np.zeros(len(A));bounds=[(v,v+h) for v in q]+[(None,None)];objective=np.r_[np.zeros(n),1.]
 ep=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs');assert ep.success
 feasible=linprog(np.zeros(n),A_ub=np.r_[order,margins],b_ub=np.zeros(n-1+len(margins)),bounds=bounds[:-1],method='highs')
 return ep,feasible,margins,order,bounds[:-1]
