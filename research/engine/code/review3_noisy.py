"""Re-run selectors through bounded noisy providers and matched random controls."""
from review3_common import ROOT,sha,write,check_registration,stable_seed,macro
from review2_counter import repair_radius
from review1_selectors import fixed_proxy
import argparse,csv,json,sys,time,platform as system_platform
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'evidence/v3/code'))
from v3_data import sequence,uniform

TOL=1e-8
BUDGETS=[2,4,8,16]
POLICIES=['uniform','audit','proxy_centers','proxy_brackets']+['random'+str(i) for i in range(5)]


def observations(r,eps,mechanism):
    truth=r['truth'];t=r['t']
    if mechanism=='clean':return np.asarray(truth,float)
    if mechanism.startswith('iid'):
        rng=np.random.default_rng(stable_seed('noise',r['flight_id'],eps,mechanism));y=truth+rng.uniform(-eps,eps,len(truth))
    elif mechanism=='drift':y=truth+eps*(2*t/t[-1]-1)
    elif mechanism=='rounding':y=np.rint(truth/(2*eps))*(2*eps)
    else:raise ValueError(mechanism)
    assert np.max(np.abs(y-truth))<=eps+1e-10
    return y


def payload_bytes(t,y,idx):
    request=json.dumps({'indices':list(map(int,idx))},separators=(',',':')).encode('ascii')
    response=json.dumps({'time_s':list(map(float,t[idx])),'energy_Wh':list(map(float,y[idx]))},separators=(',',':')).encode('ascii')
    return len(request)+len(response)


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True)
    records=joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')
    old=pd.read_csv(ROOT/'results/review1/query_histories.csv')
    history={(r.flight_id,r.policy,int(r.budget)):np.array(json.loads(r.indices),int) for r in old.itertuples()}
    caps=pd.read_csv(ROOT/'results/review1/all_caps.csv');caps=caps[caps.cap_scenario.isin(['q0.9','q0.999'])]
    capmap={s:dict(zip(z.cap_scenario,z.cap_W)) for s,z in caps.groupby('source_id')}
    conditions=[(0.,'clean')]+[(eps,kind) for eps in [.01,.1] for kind in ['iid0','iid1','drift','rounding']]
    rows=[];timing=[];cost=[];changed=[];matched=0;histories=0;started=time.perf_counter()
    # CSV streaming avoids retaining all transcript JSON in RAM.
    with (out/'noisy_histories.csv').open('w',encoding='utf-8',newline='') as stream,threadpool_limits(limits=2):
        columns=['flight_id','source_id','platform','budget','policy','epsilon_Wh','mechanism','indices','observed_energy_Wh']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for k,source in enumerate(sorted({r['source_id'] for r in records})):
            model=joblib.load(ROOT/'evidence/v3/models/proxy'/(sha_string(source)[:12]+'.joblib'))
            if hasattr(model,'n_jobs'):model.n_jobs=1
            for r in [z for z in records if z['source_id']==source]:
                fid=r['flight_id'];t=r['t'];truth=r['truth'];n=len(t)
                begin=time.perf_counter();proxy=np.maximum(model.predict((r['x'][:-1]+r['x'][1:])/2),1.)
                proxy_time=time.perf_counter()-begin
                fixed={}
                for B in BUDGETS:
                    for policy in POLICIES:
                        begin=time.perf_counter()
                        if policy=='uniform':idx=uniform(n,B)
                        elif policy=='audit':idx,_=sequence(t,lambda i:truth[i],B,capmap[source]['q0.9'],'audit',proxy)
                        elif policy in ['proxy_centers','proxy_brackets']:idx=fixed_proxy(t,proxy,B,policy=='proxy_brackets')
                        else:
                            rng=np.random.default_rng(stable_seed('random_nodes',fid,policy))
                            idx=np.array(sorted({0,n-1,*rng.choice(np.arange(1,n-1),size=min(max(B-2,0),n-2),replace=False).tolist()}),int)
                        elapsed=time.perf_counter()-begin;fixed[policy,B]=idx
                        if not policy.startswith('random'):
                            assert np.array_equal(idx,history[fid,policy,B]),(fid,policy,B)
                            matched+=1
                        cost.append(dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,policy=policy,
                                         anchors=len(idx),reference_nodes=n,sparse_JSON_bytes=payload_bytes(t,truth,idx),
                                         dense_JSON_bytes=payload_bytes(t,truth,np.arange(n)),kinematic_binary_bytes=int(r['x'].nbytes)))
                        timing.append(dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,policy=policy,
                                           selector_s=elapsed,proxy_predict_s=proxy_time))
                for eps,mechanism in conditions:
                    y=observations(r,eps,mechanism)
                    for B in BUDGETS:
                        for policy in POLICIES:
                            if policy=='audit' and mechanism!='clean':
                                idx,_=sequence(t,lambda i:y[i],B,capmap[source]['q0.9'],'audit',proxy)
                            else:idx=fixed[policy,B]
                            writer.writerow(dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,policy=policy,
                                epsilon_Wh=eps,mechanism=mechanism,indices=json.dumps(idx.tolist()),observed_energy_Wh=json.dumps(y[idx].tolist())))
                            histories+=1
                            is_changed=not np.array_equal(idx,fixed[policy,B])
                            if policy=='audit':changed.append(dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,
                                                                 epsilon_Wh=eps,mechanism=mechanism,changed=int(is_changed)))
                            for scenario,cap in capmap[source].items():
                                nominal=bool((np.diff(truth)*3600/np.diff(t)>cap+1e-8).any())
                                critical=repair_radius(t[idx],y[idx],cap)[0]
                                rejects=bool(2*critical>2*eps+TOL)
                                assert not rejects or nominal,(fid,policy,mechanism,eps)
                                clean_idx=fixed['audit',B]
                                fixed_audit=bool(2*repair_radius(t[clean_idx],y[clean_idx],cap)[0]>2*eps+TOL)
                                rows.append(dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,policy=policy,
                                    epsilon_Wh=eps,mechanism=mechanism,cap_scenario=scenario,reference_violates=nominal,
                                    rejected=rejects,missed=int(nominal and not rejects),critical_epsilon_Wh=critical,
                                    fixed_clean_audit_rejected=fixed_audit,changed_indices=is_changed,anchors=len(idx)))
            print('NOISY SOURCE',k+1,'/56',round(time.perf_counter()-started,1),'s',flush=True)
    f=pd.DataFrame(rows);assert len(f)==285120 and histories==142560 and matched==7040
    f.to_csv(out/'noisy_cases.csv',index=False);pd.DataFrame(changed).to_csv(out/'adaptive_changed.csv',index=False)
    pd.DataFrame(cost).to_csv(out/'payload_cost.csv',index=False);pd.DataFrame(timing).to_csv(out/'runtime.csv',index=False)
    # Pool randomized-policy seeds at the record level before source aggregation.
    keys=['flight_id','source_id','platform','budget','epsilon_Wh','mechanism','cap_scenario']
    random=f[f.policy.str.startswith('random')].groupby(keys)[['reference_violates','rejected','missed']].mean().reset_index()
    random['policy']='random_mean';pool=pd.concat([f[~f.policy.str.startswith('random')],random],ignore_index=True)
    summary=[]
    for key,a in pool.groupby(['budget','epsilon_Wh','mechanism','cap_scenario','policy'],sort=True):
        B,eps,kind,scenario,policy=key
        summary.append(dict(budget=B,epsilon_Wh=eps,mechanism=kind,cap_scenario=scenario,policy=policy,
                            missed_burden=macro(a,'missed'),rejection_fraction=macro(a,'rejected'),
                            nominal_violation_fraction=macro(a,'reference_violates')))
    pd.DataFrame(summary).to_csv(out/'noisy_summary.csv',index=False)
    seeds=[]
    for key,a in f[f.policy.str.startswith('random')].groupby(['budget','epsilon_Wh','mechanism','cap_scenario','policy'],sort=True):
        B,eps,kind,scenario,policy=key;seeds.append(dict(budget=B,epsilon_Wh=eps,mechanism=kind,cap_scenario=scenario,
                                                        policy=policy,missed_burden=macro(a,'missed')))
    pd.DataFrame(seeds).to_csv(out/'random_seed_summary.csv',index=False)
    paired=[]
    for key,a in f[f.policy.eq('audit')].groupby(['budget','epsilon_Wh','mechanism','cap_scenario'],sort=True):
        B,eps,kind,scenario=key;a=a.copy();a['reselection_change']=a.rejected.astype(int)-a.fixed_clean_audit_rejected.astype(int)
        paired.append(dict(budget=B,epsilon_Wh=eps,mechanism=kind,cap_scenario=scenario,
                           gain_from_reselecting=macro(a,'reselection_change'),changed_records=int(a.changed_indices.sum()),
                           changed_fraction=macro(a,'changed_indices')))
    pd.DataFrame(paired).to_csv(out/'reselection_effect.csv',index=False)
    deterministic=['noisy_histories.csv','noisy_cases.csv','adaptive_changed.csv','payload_cost.csv',
                   'noisy_summary.csv','random_seed_summary.csv','reselection_effect.csv']
    gate=dict(passed=True,records=len(records),source_groups=56,histories=histories,decision_cases=len(f),
              original_deployable_clean_histories_identical=matched,false_rejections_of_nominal_compatible_reference=0,
              noise_is_simulated_not_calibrated=True,selector_reexecuted_on_queried_noisy_values=True,
              new_fits=0,new_blind_cases=0,
              output_hashes=[dict(path=p,sha256=sha(out/p)) for p in deterministic],
              timing_reproducibility='runtime.csv is an execution measurement; excluded from deterministic hash gate')
    write(out/'noisy_gate.json',gate)
    write(out/'runtime_environment.json',dict(python=sys.version,platform=system_platform.platform(),
        numerical_thread_limit=2,proxy_n_jobs=1,scope='local in-memory lookup and selectors; no network/server I/O/radio energy',
        excluded_payload_costs=['kinematic acquisition','transport headers','retries','server file I/O','radio power']))
    print(json.dumps(gate,indent=2),flush=True)


def sha_string(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review3');args=parser.parse_args();main(ROOT/args.output)
