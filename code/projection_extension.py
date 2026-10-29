"""Match frozen state-only priors with the actual sparse observation contract."""
from common4 import *
from integrated_core import project_prior,checked_blend,boxes
import numpy as np,pandas as pd,joblib,time,argparse
from threadpoolctl import threadpool_limits

def main(output):
 check();output.mkdir(parents=True,exist_ok=True);rpath=ROOT/'research/04_RobustMoments/results';rec={r['flight_id']:r for r in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib') if r['dataset']=='IDF-DS'}
 prior={};alignment=0.
 for model in ['ResidualTCN','IRM']:
  for scenario in ['original','wind_missing']:
   frames=[pd.read_parquet(p) for p in rpath.joinpath('stress_predictions').glob(f'{model}_*_{scenario}.parquet')];f=pd.concat(frames,ignore_index=True)
   for fid,z in f.groupby('flight_id'):
    z=z.sort_values('time_s');t=z.time_s.to_numpy(copy=True);t-=t[0];dt=np.diff(t);h=np.r_[0,np.where((dt>0)&(dt<=2),dt/7200,0.)];pwr=z.prediction_w.to_numpy();y=z.power_w.to_numpy();C=np.cumsum(h*(pwr+np.r_[pwr[0],pwr[:-1]]));truth=np.cumsum(h*(y+np.r_[y[0],y[:-1]]));r=rec[fid]
    alignment=max(alignment,float(np.max(abs(np.interp(r['t'],t,truth)-r['truth']))));assert alignment<1e-9
    prior[fid,model,scenario]=np.interp(r['t'],t,C)
 history=pd.read_csv(ROOT/'results/new_query/query_histories.csv');history=history[history.flight_id.isin(rec)&history.budget.eq(8)&history.cap_scenario.eq('q0.9')&history.policy.isin(['uniform','guard_dense'])&((history.epsilon_Wh.eq(0)&history.mechanism.eq('clean'))|(history.epsilon_Wh.eq(.1)&history.mechanism.eq('drift')))]
 rows=[];timings=[];count=0;start=time.perf_counter()
 with threadpool_limits(limits=2):
  for row in history.itertuples():
   r=rec[row.flight_id];t=r['t'];idx=np.array(json.loads(row.indices),int);y=np.array(json.loads(row.observed_energy_Wh));eps=float(row.epsilon_Wh);actual=r['truth']-r['truth'][0]
   for h in [0.,.5]:
    a,b=boxes(t[idx],h)
    for model in ['ResidualTCN','IRM']:
     for scenario in ['original','wind_missing']:
      pre=prior[row.flight_id,model,scenario];begin=time.perf_counter();got=project_prior(t,pre,a,b,y,2000.,eps);elapsed=time.perf_counter()-begin;count+=1
      base=dict(flight_id=row.flight_id,source_id=row.source_id,platform=row.platform,policy=row.policy,epsilon_Wh=eps,mechanism=row.mechanism,halfwidth_s=h,model=model,input_scenario=scenario,status=got['status'])
      if got['status']!='FEASIBLE_PROJECTED_PRIOR':rows.append(base);continue
      timings.append({**base,'seconds':elapsed,'variables':got['variables'],'nonzeros':got['nonzeros']})
      for weight in [0.,.5,1.]:
       E,diagnostic=checked_blend(got,weight,a,b,y,2000.,eps);p=np.interp(t,got['grid'],E);error=p-actual
       assert np.max(abs(error))<=(1+weight)*got['radius_Wh']+1e-8
       rows.append({**base,'weight':weight,'RMSE_Wh':float(np.sqrt(np.mean(error**2))),'max_error_Wh':float(np.max(abs(error))),'radius_Wh':got['radius_Wh'],'prior_RMSE_Wh':float(np.sqrt(np.mean((pre-actual)**2))),**diagnostic})
   if count%320==0:print('PROJECT',count,'LPs',round(time.perf_counter()-start,1),'s',flush=True)
 f=pd.DataFrame(rows);f.to_csv(output/'projection_cases.csv',index=False);pd.DataFrame(timings).to_csv(output/'projection_runtime.csv',index=False)
 group=['policy','epsilon_Wh','halfwidth_s','model','input_scenario','weight'];metrics=['RMSE_Wh','max_error_Wh','radius_Wh','prior_RMSE_Wh','worst_case_bound_Wh']
 summary=[dict(zip(group,key),records=len(z),**{c:macro(z,c) for c in metrics}) for key,z in f.dropna(subset=['weight']).groupby(group,sort=True)]
 pd.DataFrame(summary).to_csv(output/'projection_summary.csv',index=False)
 write(output/'projection_gate.json',dict(passed=bool(f.status.eq('FEASIBLE_PROJECTED_PRIOR').all()),LPs=count,metric_rows=len(f),reference_alignment_max_Wh=alignment,all_source_records=231,new_neural_fits=0,seconds=time.perf_counter()-start,status_counts=f.status.value_counts().to_dict()))
 print('PROJECT COMPLETE',count,len(f),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/new_projection');a=p.parse_args();main(ROOT/a.output)
