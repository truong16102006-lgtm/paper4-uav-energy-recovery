"""All predeclared query, provider and timestamp conditions; no refitting."""
from common4 import *
from integrated_core import guarded_indices,boxes,radius_bound
from review6_envelopes_v2 import envelopes,evaluate,feasibility_residual
from review3_noisy import observations
import numpy as np,pandas as pd,joblib,csv,time,argparse
from threadpoolctl import threadpool_limits

def main(output):
 check();output.mkdir(parents=True,exist_ok=True);eng=ROOT/'research/engine'
 rec=joblib.load(eng/'evidence/v3/data/dev_records.joblib');caps=pd.read_csv(eng/'results/review1/all_caps.csv');caps=caps[caps.cap_scenario.isin(['q0.9','q0.999'])]
 capmap={s:dict(zip(z.cap_scenario,z.cap_W)) for s,z in caps.groupby('source_id')};conditions=[(0.,'clean')]+[(e,k) for e in [.01,.1] for k in ['iid0','iid1','drift','rounding']]
 rows=[];hist=0;start=time.perf_counter();max_contain=0.;max_bound=0.
 with (output/'query_histories.csv').open('w',encoding='utf-8',newline='') as stream,threadpool_limits(limits=2):
  fields=['flight_id','source_id','platform','budget','cap_scenario','policy','epsilon_Wh','mechanism','indices','observed_energy_Wh'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  for si,source in enumerate(sorted(capmap)):
   proxy_model=joblib.load(eng/'evidence/v3/models/proxy'/(hashlib.sha256(source.encode()).hexdigest()[:12]+'.joblib'))
   for r in [r for r in rec if r['source_id']==source]:
    t,truth=r['t'],r['truth'];proxy=np.maximum(proxy_model.predict((r['x'][:-1]+r['x'][1:])/2),1.)
    for B in [4,8,16]:
     for eps,kind in conditions:
      y=observations(r,eps,kind)
      for capname,cap in capmap[source].items():
       V=bool((np.diff(truth)*3600/np.diff(t)>cap+1e-8).any())
       for policy in ['uniform','endpoint','guard_dense','guard_half']:
        begin=time.perf_counter();idx,info=guarded_indices(t,proxy,B,cap,y[0],y[-1],eps,policy);select_time=time.perf_counter()-begin
        base=dict(flight_id=r['flight_id'],source_id=source,platform=r['platform'],budget=B,cap_scenario=capname,policy=policy,epsilon_Wh=eps,mechanism=kind)
        writer.writerow({**base,'indices':json.dumps(idx.tolist()),'observed_energy_Wh':json.dumps(y[idx].tolist())});hist+=1
        last_volume=-1.;last_reject=True
        for h in [0.,.1,.5]:
         a,b=boxes(t[idx],h);env=envelopes(a,b,y[idx],2000.,eps);assert env['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE'
         lo,hi=evaluate(env,t);actual=truth-truth[0];contain=max(float(np.max(lo-actual)),float(np.max(actual-hi)),0.);max_contain=max(max_contain,contain);assert contain<1e-8
         bound=radius_bound(t[idx],2000.,eps,h);max_bound=max(max_bound,env['minimax_radius_Wh']-bound);assert env['minimax_radius_Wh']<=bound+1e-8
         residual=feasibility_residual(a,b,y[idx],cap,eps);D=bool(residual>1e-8);assert not D or V;assert not D or last_reject;last_reject=D
         assert env['volume_Wh_s']>=last_volume-1e-8;last_volume=env['volume_Wh_s']
         rows.append({**base,'halfwidth_s':h,'anchors':len(idx),'reference_violates':V,'rejected':D,'missed':int(V and not D),'volume_Wh_s':env['volume_Wh_s'],'radius_Wh':env['minimax_radius_Wh'],'midpoint_RMSE_Wh':float(np.sqrt(np.mean(((lo+hi)/2-actual)**2))),'maximum_gap_s':float(np.diff(t[idx]).max()),'radius_bound_Wh':bound,'selector_seconds':select_time,'proxy_fallback':info['uniform_fallback']})
   print('QUERY SOURCE',si+1,'/56',round(time.perf_counter()-start,1),'s',flush=True)
 f=pd.DataFrame(rows);f.to_csv(output/'query_cases.csv',index=False)
 group=['budget','cap_scenario','policy','epsilon_Wh','mechanism','halfwidth_s'];metrics=['missed','rejected','volume_Wh_s','radius_Wh','midpoint_RMSE_Wh','maximum_gap_s','radius_bound_Wh']
 summary=[dict(zip(group,key),records=len(z),**{c:macro(z,c) for c in metrics}) for key,z in f.groupby(group,sort=True)]
 pd.DataFrame(summary).to_csv(output/'query_summary.csv',index=False)
 write(output/'query_gate.json',dict(passed=True,histories=hist,cases=len(f),records=len(rec),maximum_containment_residual_Wh=max_contain,maximum_radius_bound_excess_Wh=max_bound,new_proxy_fits=0,seconds=time.perf_counter()-start,all_registered_conditions_retained=True))
 print('QUERY COMPLETE',hist,len(f),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/new_query');a=p.parse_args();main(ROOT/a.output)
