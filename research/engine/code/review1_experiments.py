from review1_common import *
from review1_selectors import *
import sys,hashlib,time
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(ROOT/'evidence/v3/code'))
from v3_data import sequence,query_next
from v3_geometry import exact_geometry

def main():
 check_review();out=ROOT/'results/review1';out.mkdir(parents=True,exist_ok=True);log('REVIEW1 QUERY STUDY START:fixed predeclared controls/caps; unchanged source-excluded proxies; oracle evaluator only; no new fit.')
 rng=np.random.default_rng(2026102610);checks=0
 for trial in range(150):
  n=int(rng.integers(4,90));t=np.r_[0.,np.cumsum(rng.uniform(.05,2,n-1))];power=rng.uniform(0,1900,n-1);truth=np.r_[0.,np.cumsum(power*np.diff(t)/3600)];pp=rng.uniform(1,1800,n-1)
  for B in [2,4,8,16]:
   for brackets in [False,True]:
    idx=fixed_proxy(t,pp,B,brackets);assert len(idx)==min(B,n) and idx[0]==0 and idx[-1]==n-1
   idx=oracle_indices(t,truth,B)
   if B>=4:
    for cap in [300.,900.,1800.]:assert bool((power>cap+1e-8).any())==bool((np.diff(truth[idx])*3600/np.diff(t[idx])>cap+1e-8).any())
   i1,_=sequence(t,lambda j:truth[j],B,300.,'audit',pp);i2,_=sequence(t,lambda j:truth[j],B,2000.,'audit',pp);assert np.array_equal(i1,i2);checks+=1
 write(out/'selector_preflight.json',{'passed':True,'mechanism_budget_cases':checks,'oracle_requires_at_most_four_nodes':True,'fixed_selector_energy_arguments':False,'audit_cap_scale_invariance':True})
 rec=joblib.load(ROOT/'evidence/v3/data/dev_records.joblib');oldcaps=pd.read_csv(ROOT/'evidence/v3/results/g4/fit_caps.csv').set_index('source_id');oldhist=pd.read_csv(ROOT/'evidence/v3/results/g4/query_histories.csv');history={(r.flight_id,r.policy,int(r.budget)):np.array(json.loads(r.indices),int) for r in oldhist.itertuples() if r.policy in ['uniform','audit']};old=pd.read_csv(ROOT/'evidence/v3/results/g4/query_per_segment.csv');old=old[old.policy.isin(['uniform','audit'])].set_index(['flight_id','policy','budget'])
 source_samples={};sources=sorted({r['source_id'] for r in rec});platform_of={r['source_id']:r['platform'] for r in rec}
 for s in sources:
  power=np.concatenate([np.diff(r['truth'])*3600/np.diff(r['t']) for r in rec if r['source_id']==s]);source_samples[s]=np.interp(np.linspace(0,len(power)-1,129),np.arange(len(power)),power)
 qs=[.9,.95,.99,.995,.999];caps={};caprows=[]
 for s in sources:
  train=np.concatenate([source_samples[a] for a in sources if a!=s and platform_of[a]==platform_of[s]])
  a=[('q'+format(q,'g'),float(np.quantile(train,q))) for q in qs]
  assert abs(a[0][1]-oldcaps.loc[s,'cap_W'])<1e-10
  fixed=[588.,756.,1176.,1512.,2000.] if platform_of[s]!='cmu_quadrotor' else [2000.]
  a+=[('fixed'+str(int(v)),v) for v in fixed];caps[s]=a
  for name,cap in a:caprows.append({'source_id':s,'platform':platform_of[s],'cap_scenario':name,'cap_W':cap,'fit_sources_of_same_platform':sum(platform_of[k]==platform_of[s] and k!=s for k in sources),'scope':'fit-only point-tail quantile or documentary scenario, not physical incident threshold'})
 pd.DataFrame(caprows).to_csv(out/'all_caps.csv',index=False)
 rows=[];histories=[];cost=[];start=time.perf_counter();matched=0
 for k,s in enumerate(sources):
  model=joblib.load(ROOT/'evidence/v3/models/proxy'/(hashlib.sha256(s.encode()).hexdigest()[:12]+'.joblib'))
  for r in [z for z in rec if z['source_id']==s]:
   t=r['t'];truth=r['truth'];pp=np.maximum(model.predict((r['x'][:-1]+r['x'][1:])/2),1.);power=np.diff(truth)*3600/np.diff(t)
   for B in [2,4,8,16]:
    policies={'uniform':history[r['flight_id'],'uniform',B],'audit':history[r['flight_id'],'audit',B],'proxy_centers':fixed_proxy(t,pp,B,False),'proxy_brackets':fixed_proxy(t,pp,B,True),'oracle':oracle_indices(t,truth,B)}
    baseline_V=exact_geometry(t,policies['uniform'],truth[policies['uniform']])[1]
    for policy,idx in policies.items():
     y=truth[idx];queried_power=np.diff(y)*3600/np.diff(t[idx]);vol=exact_geometry(t,idx,y)[1];histories.append({'flight_id':r['flight_id'],'source_id':s,'platform':r['platform'],'budget':B,'policy':policy,'indices':json.dumps(idx.tolist()),'queried_energy_Wh':json.dumps(y.tolist()),'forbidden_label_oracle':policy=='oracle'})
     cost.append({'flight_id':r['flight_id'],'source_id':s,'platform':r['platform'],'budget':B,'policy':policy,'continuous_volume_Wh_s':vol,'uniform_volume_Wh_s':baseline_V,'relative_volume_increase':(vol-baseline_V)/baseline_V if baseline_V>1e-12 else np.nan,'role':'separate exact2000W recovery objective; hindsight oracle not deployable'})
     for name,cap in caps[s]:
      V=bool((power>cap+1e-8).any());D=bool((queried_power>cap+1e-8).any());assert not D or V
      if policy=='oracle' and B>=4:assert V==D
      if name=='q0.9' and policy in ['uniform','audit']:
       prior=old.loc[(r['flight_id'],policy,B)];assert bool(prior.true_reference_violation)==V and bool(prior.rejected)==D and int(prior.missed_violation)==int(V and not D);matched+=1
      rows.append({'flight_id':r['flight_id'],'source_id':s,'platform':r['platform'],'budget':B,'policy':policy,'cap_scenario':name,'cap_W':cap,'reference_violates':V,'rejected':D,'missed_violation':int(V and not D),'false_alarm':int(D and not V),'anchors':len(idx),'role':'post-review already-seen DEV; oracle uses forbidden labels for upper bound only' if policy=='oracle' else 'post-review already-seen DEV, unchanged primary protocol'})
  print(k+1,'/56',s,'elapsed',round(time.perf_counter()-start,1),flush=True)
 pd.DataFrame(rows).to_csv(out/'query_per_record.csv',index=False);pd.DataFrame(histories).to_csv(out/'query_histories.csv',index=False);pd.DataFrame(cost).to_csv(out/'recovery_cost.csv',index=False)
 assert matched==3520
 write(out/'experiment_gate.json',{'passed':True,'records':len(rec),'source_groups':len(sources),'query_cap_policy_cases':len(rows),'histories':len(histories),'original_Q90_uniform_audit_rows_identical':matched,'new_fits':0,'new_blind_cases':0,'reference_false_alarms':sum(r['false_alarm'] for r in rows),'oracle_hindsight_only':True});log('REVIEW1 QUERY STUDY COMPLETE:all predeclared controls/scenarios retained;3520oldprimaryrowsidentical;oracleceilingonly.');print('PASS',len(rows))
if __name__=='__main__':main()
