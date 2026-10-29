"""Shared-origin projection and radius-aware query backbones, float64."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from review4_selectors import excess_pair,gap_fill
from review6_envelopes_v2 import envelopes,evaluate,prepare

def guarded_indices(t,proxy,budget,cap,y0,yT,epsilon,kind='guard_dense'):
 endpoint,info=excess_pair(t,proxy,budget,cap,y0,yT,epsilon,True)
 if kind=='endpoint':return endpoint,info
 if kind=='uniform':return np.unique(np.rint(np.linspace(0,len(t)-1,min(budget,len(t)))).astype(int)),info
 if info['uniform_fallback']:return endpoint,info
 k=max(2,budget-2) if kind=='guard_dense' else max(2,int(np.ceil(budget/2)))
 backbone=np.unique(np.rint(np.linspace(0,len(t)-1,min(k,len(t)))).astype(int))
 idx=gap_fill(t,{*backbone,int(info['witness_left']),int(info['witness_right'])},budget)
 assert set(backbone).issubset(idx) and len(idx)<=budget
 return idx,{**info,'backbone_size':len(backbone)}

def boxes(t,h):
 a=np.maximum(t-h,t[0]);b=np.minimum(t+h,t[-1]);a[0]=b[0]=t[0];a[-1]=b[-1]=t[-1]
 return a,b

def radius_bound(t,cap,epsilon,h):
 # Uniform reading errors, exact horizon endpoints, nominal anchor mesh.
 return min(cap*(t[-1]-t[0])/7200,2*epsilon+cap*h/3600+cap*np.diff(t).max()/14400)

def constraints(grid,a,b,l,u,m,prior,sparse=True):
 n=len(grid);rr=[];cc=[];vv=[];rhs=[];row=0
 def add(cols,vals,right):
  nonlocal row
  rr.extend([row]*len(cols));cc.extend(cols);vv.extend(vals);rhs.append(right);row+=1
 for i,dt in enumerate(np.diff(grid)):
  add([i,i+1],[1.,-1.],0.);add([i,i+1],[-1.,1.],m*dt)
 for ai,bi,li,ui in zip(a,b,l,u):
  ia=int(np.searchsorted(grid,ai));ib=int(np.searchsorted(grid,bi))
  add([ia],[1.],ui);add([ib],[-1.],-li)
 for i in range(1,n):
  add([i,0,n],[1.,-1.,-1.],prior[i]);add([i,0,n],[-1.,1.,-1.],-prior[i])
 A=coo_matrix((vv,(rr,cc)),shape=(row,n+1)).tocsr();c=np.r_[np.zeros(n),1.]
 return A if sparse else A.toarray(),np.array(rhs),c

def project_prior(t,prior,a,b,y,cap,epsilon):
 env=envelopes(a,b,y,cap,epsilon)
 if env['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':return {'status':env['status']}
 aa,bb,l,u,m,t0,y0,*_=prepare(a,b,y,cap,epsilon)
 grid=np.unique(np.r_[np.asarray(t)-t0,aa,bb,env['knots_elapsed_s']]);p=np.interp(grid,np.asarray(t)-t0,prior)
 if not np.isfinite(p).all():raise ValueError('Finite prior required')
 A,rhs,obj=constraints(grid,aa,bb,l,u,m,p)
 result=linprog(obj,A_ub=A,b_ub=rhs,bounds=[(None,None)]*len(grid)+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if not result.success:return {'status':'UNRESOLVED_SOLVER','solver_status':int(result.status)}
 z=result.x[:-1];E=z-z[0];res=float(max(0.,np.max(A@result.x-rhs)))
 if res>1e-8:return {'status':'UNRESOLVED_RESIDUAL','residual_Wh':res}
 lo,hi=evaluate(env,grid);mid=(lo+hi)/2;R=env['minimax_radius_Wh']
 assert np.max(np.maximum(lo-E,E-hi))<1e-8
 return dict(status='FEASIBLE_PROJECTED_PRIOR',grid=grid,projected_increment=E,midpoint_increment=mid,
             projected_absolute_centered=z,midpoint_origin=sum(env['initial_counter_interval_centered_Wh'])/2,
             lower=lo,upper=hi,radius_Wh=R,volume_Wh_s=env['volume_Wh_s'],objective_Wh=float(result.fun),
             residual_Wh=res,variables=len(obj),nonzeros=int(A.nnz))

def checked_blend(result,weight,a,b,y,cap,epsilon):
 assert result['status']=='FEASIBLE_PROJECTED_PRIOR' and 0<=weight<=1
 E=(1-weight)*result['midpoint_increment']+weight*result['projected_increment']
 origin=(1-weight)*result['midpoint_origin']+weight*result['projected_absolute_centered'][0]
 z=E+origin;grid=result['grid'];aa,bb,l,u,m,t0,y0,*_=prepare(a,b,y,cap,epsilon)
 residual=max(0.,float(np.max(-np.diff(z))),float(np.max(np.diff(z)-m*np.diff(grid))),
              float(np.max(np.interp(aa,grid,z)-u)),float(np.max(l-np.interp(bb,grid,z))))
 assert residual<1e-8
 worst=float(np.max(np.maximum(E-result['lower'],result['upper']-E)))
 assert worst<=(1+weight)*result['radius_Wh']+1e-8
 return E,dict(residual_Wh=residual,worst_case_bound_Wh=(1+weight)*result['radius_Wh'],actual_conditional_worst_Wh=worst)
