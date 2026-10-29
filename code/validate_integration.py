"""Independent dense optimization checks and a shared-origin counterexample."""
from common4 import *
from integrated_core import *
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import argparse,time

def dense_projection(grid,a,b,y,eps,cap,prior):
 # Independent direct-value bounds, adjacent increments and explicit error rows.
 n=len(grid);rows=[];rhs=[]
 for j in range(n-1):
  v=np.zeros(n+1);v[j+1]=1;v[j]=-1;rows.extend([v,-v]);rhs.extend([cap*np.diff(grid)[j]/3600,0.])
 l=np.maximum.accumulate(np.asarray(y)-eps);u=np.minimum.accumulate((np.asarray(y)+eps)[::-1])[::-1]
 for j in range(len(y)):
  v=np.zeros(n+1);v[np.where(grid==a[j])[0][0]]=1;rows.append(v);rhs.append(u[j])
  v=np.zeros(n+1);v[np.where(grid==b[j])[0][0]]=-1;rows.append(v);rhs.append(-l[j])
 for j in range(1,n):
  v=np.zeros(n+1);v[j]=1;v[0]=-1;v[-1]=-1;rows.append(v);rhs.append(prior[j])
  v=np.zeros(n+1);v[j]=-1;v[0]=1;v[-1]=-1;rows.append(v);rhs.append(-prior[j])
 obj=np.r_[np.zeros(n),1.];return linprog(obj,A_ub=rows,b_ub=rhs,bounds=[(None,None)]*n+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})

def main(output):
 check();output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026102704);maxdiff=0.;count=0;start=time.perf_counter()
 with threadpool_limits(limits=2):
  for trial in range(200):
   n=int(rng.integers(3,13));t=np.r_[0,np.cumsum(rng.uniform(.2,2,n-1))];cap=float(rng.choice([0.,360.,3600.]));m=cap/3600;truth=np.r_[0,np.cumsum(rng.uniform(0,m,n-1)*np.diff(t))];eps=float(rng.choice([0.,.01,.1]));y=truth+rng.uniform(-eps,eps,n);h=float(rng.choice([0.,.1,.5]));a,b=boxes(t,h)
   env=envelopes(a,b,y,cap,eps);assert env['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE';assert env['minimax_radius_Wh']<=radius_bound(t,cap,eps,h)+1e-8
   prior=truth+rng.normal(0,.8,n);prior[0]=0.;got=project_prior(t,prior,a,b,y,cap,eps);assert got['status']=='FEASIBLE_PROJECTED_PRIOR'
   grid=got['grid'];dense=dense_projection(grid,a,b,y-y[0],eps,cap,np.interp(grid,t,prior));assert dense.success
   diff=abs(dense.fun-got['objective_Wh']);maxdiff=max(maxdiff,diff);assert diff<1e-8;count+=2
   for weight in [0.,.5,1.]:checked_blend(got,weight,a,b,y,cap,eps)
  # A monotone, slope-compatible pointwise band member has no common origin.
  t=np.array([0.,5.,10.]);y=np.array([0.,5.,10.]);naive=np.array([0.,7.,8.]);env=envelopes(t,t,y,7200.,1.);lo,hi=evaluate(env,t)
  assert (naive>=lo).all() and (naive<=hi).all() and np.all(np.diff(naive)/np.diff(t)<=2)
  origin_lo=float(np.max(y-1-naive));origin_hi=float(np.min(y+1-naive));assert origin_lo>origin_hi
  got=project_prior(t,naive,t,t,y,7200.,1.);assert got['status']=='FEASIBLE_PROJECTED_PRIOR';checked_blend(got,.5,t,t,y,7200.,1.)
 write(output/'shared_origin_counterexample.json',dict(time_s=t.tolist(),readings_Wh=y.tolist(),naive_increment_Wh=naive.tolist(),reading_error_Wh=1.,cap_W=7200.,naive_inside_sharp_pointwise_envelope=True,naive_monotone_and_cap_slope=True,required_origin_lower_Wh=origin_lo,required_origin_upper_Wh=origin_hi,common_origin_exists=False))
 write(output/'validation_gate.json',dict(passed=True,mechanisms=200,independent_dense_and_sparse_LP_solves=count,max_objective_difference_Wh=maxdiff,radius_bound_checked=True,feasible_blends_checked=600,shared_origin_counterexample=True,seconds=time.perf_counter()-start))
 print('INTEGRATION VALIDATION PASSED',count,maxdiff)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/new_validation');a=p.parse_args();main(ROOT/a.output)
