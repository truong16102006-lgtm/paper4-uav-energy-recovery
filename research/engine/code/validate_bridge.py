from common import *
from measurement_bridge import *
import pandas as pd

def main():
 check();log('REGISTERED BRIDGE CHECK START:400 random cases; joint unit/time LP versus direct feasibility, no measured flight simulation.')
 rng=np.random.default_rng(2026102609);rows=[]
 for trial in range(400):
  n=int(rng.integers(4,25));p=rng.uniform(100,1600,n);bins=np.cumsum(rng.choice([0,0,1,2],n));h=[1.,.1,.01][trial%3];cap=[300.,800.,2000.][trial%3];idx=np.unique(np.rint(np.linspace(0,n-1,min(n,5))).astype(int))
  for k in [.5,1.,2.]:
   c=margin_coefficients(p,cap,k);lo,hi=linear_range(c,bins,h);ep,feas,A,order,bd=common_clock(p,bins,idx,cap,k,h);low=linprog(c,A_ub=order,b_ub=np.zeros(n-1),bounds=bd,method='highs');high=linprog(-c,A_ub=order,b_ub=np.zeros(n-1),bounds=bd,method='highs');assert low.success and high.success
   error=max(abs(lo-low.fun),abs(hi+high.fun));assert error<1e-8
   if abs(ep.fun)>1e-8:assert (ep.fun<0)==feas.success
   mins=[linear_range(a,bins,h)[0] for a in A];assert max(mins)<=ep.fun+1e-8
   assert (order@ep.x[:-1]<=1e-8).all() and (A@ep.x[:-1]<=ep.fun+1e-8).all()
   rows.append({'trial':trial,'nodes':n,'unit_factor':k,'h_s':h,'cap_W':cap,'range_LP_max_error_Wh':error,'joint_minimax_Wh':float(ep.fun),'largest_single_interval_minimum_Wh':max(mins),'direct_feasible':bool(feas.success),'role':'registered numerical mechanism, not flight evidence'})
 f=pd.DataFrame(rows);f.to_csv(ROOT/'results/bridge_validation.csv',index=False);result={'passed':True,'synthetic_cases':400,'unit_cases':len(rows),'LP_solves':len(rows)*4,'max_range_error_Wh':float(f.range_LP_max_error_Wh.max()),'common_clock_vs_direct_feasibility':True,'no_world_priority_claim':True};write(ROOT/'results/bridge_validation.json',result);log('BRIDGE CHECK PASS '+json.dumps(result));print(result)
if __name__=='__main__':main()
