"""Registered decision-aligned candidate controls; original H1 is unchanged."""
from review4_common import ROOT,sha,write,check_registration
from review4_selectors import excess_pair
from review2_counter import repair_radius
from review3_noisy import observations,sha_string
from review3_common import macro
import numpy as np
import pandas as pd
import joblib,json,csv,time,argparse
from threadpoolctl import threadpool_limits


def validate(out):
    rng=np.random.default_rng(2026102704);checks=0
    for trial in range(400):
        n=int(rng.integers(4,45));t=np.r_[0.,np.cumsum(rng.uniform(.05,3,n-1))]
        p=rng.uniform(0,2500,n-1);cap=float(rng.uniform(100,2100));F=np.r_[0.,np.cumsum(p*np.diff(t)/3600)]
        brute=max([0.]+[float(F[j]-F[i]-cap*(t[j]-t[i])/3600) for i in range(n) for j in range(i+1,n)])/2
        for B in [4,8,16]:
            idx,d=excess_pair(t,p,B,cap,0.,F[-1],0.,False)
            assert len(idx)==min(B,n) and idx[0]==0 and idx[-1]==n-1
            assert abs(d['predicted_repair_Wh']-brute)<1e-8
            assert abs(repair_radius(t[idx],F[idx],cap)[0]-brute)<1e-8
            _,large=excess_pair(t,p,B,cap,0.,F[-1],brute+1,False)
            assert large['uniform_fallback'];checks+=1
    result=dict(passed=True,synthetic_cases=400,budget_cases=checks,proxy_pair_all_pair_optimality_checked=True,
                true_energy_optimality_claimed=False,large_tolerance_fallback_checked=True)
    write(out/'retrieval_validation.json',result)


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);validate(out)
    records=joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')
    caps=pd.read_csv(ROOT/'results/review1/all_caps.csv');caps=caps[caps.cap_scenario.isin(['q0.9','q0.999'])]
    capmap={s:dict(zip(a.cap_scenario,a.cap_W)) for s,a in caps.groupby('source_id')}
    conditions=[(0.,'clean')]+[(e,m) for e in [.01,.1] for m in ['iid0','iid1','drift','rounding']]
    rows=[];cost=[];timing=[];count=0;started=time.perf_counter()
    with (out/'candidate_histories.csv').open('w',encoding='utf-8',newline='') as stream,threadpool_limits(limits=2):
        columns=['flight_id','source_id','platform','budget','cap_scenario','policy','epsilon_Wh','mechanism','indices','observed_energy_Wh']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for k,source in enumerate(sorted({r['source_id'] for r in records})):
            model=joblib.load(ROOT/'evidence/v3/models/proxy'/(sha_string(source)[:12]+'.joblib'))
            for r in [r for r in records if r['source_id']==source]:
                fid=r['flight_id'];t=r['t'];truth=r['truth'];pp=np.maximum(model.predict((r['x'][:-1]+r['x'][1:])/2),1.)
                for eps,mechanism in conditions:
                    y=observations(r,eps,mechanism)
                    for B in [2,4,8,16]:
                        for scenario,cap in capmap[source].items():
                            V=bool((np.diff(truth)*3600/np.diff(t)>cap+1e-8).any())
                            for policy,normalize in [('excess_absolute',False),('excess_endpoint',True)]:
                                begin=time.perf_counter();idx,d=excess_pair(t,pp,B,cap,float(y[0]),float(y[-1]),eps,normalize)
                                elapsed=time.perf_counter()-begin
                                radius=repair_radius(t[idx],y[idx],cap)[0];D=bool(2*radius>2*eps+1e-8)
                                assert not D or V
                                base=dict(flight_id=fid,source_id=source,platform=r['platform'],budget=B,cap_scenario=scenario,
                                          policy=policy,epsilon_Wh=eps,mechanism=mechanism)
                                writer.writerow({**base,'indices':json.dumps(idx.tolist()),'observed_energy_Wh':json.dumps(y[idx].tolist())});count+=1
                                rows.append({**base,'reference_violates':V,'rejected':D,'missed':int(V and not D),
                                             'critical_epsilon_Wh':radius,'anchors':len(idx),**d})
                                if mechanism=='clean':
                                    E=np.diff(truth[idx]);T=np.diff(t[idx]);assert np.max(E-2000*T/3600)<1e-8
                                    cost.append({**base,'volume_Wh_s':float(np.sum(E*(T-3600*E/2000)))})
                                    timing.append({**base,'selector_s':elapsed})
            if (k+1)%5==0:print('CANDIDATE SOURCE',k+1,'/56',round(time.perf_counter()-started,1),'s',flush=True)
    f=pd.DataFrame(rows);assert len(f)==63360 and count==63360
    f.to_csv(out/'candidate_cases.csv',index=False);pd.DataFrame(cost).to_csv(out/'candidate_recovery.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'candidate_runtime.csv',index=False)
    summaries=[]
    for key,a in f.groupby(['budget','cap_scenario','policy','epsilon_Wh','mechanism'],sort=True):
        B,scenario,policy,eps,mechanism=key
        summaries.append(dict(budget=B,cap_scenario=scenario,policy=policy,epsilon_Wh=eps,mechanism=mechanism,
                              missed_burden=macro(a,'missed'),rejection_fraction=macro(a,'rejected'),
                              uniform_fallback_fraction=macro(a,'uniform_fallback'),raw_rejections=int(a.rejected.sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'candidate_summary.csv',index=False)
    old=pd.read_csv(ROOT/'results/review3/noisy_summary.csv');joined=pd.concat([old,summary],ignore_index=True)
    gains=[]
    for key,a in joined.groupby(['budget','cap_scenario','epsilon_Wh','mechanism'],sort=True):
        B,scenario,eps,mechanism=key;baseline=float(a[a.policy.eq('uniform')].iloc[0].missed_burden)
        for r in a.itertuples():gains.append(dict(budget=B,cap_scenario=scenario,epsilon_Wh=eps,mechanism=mechanism,
                                               policy=r.policy,gain_over_uniform=baseline-r.missed_burden))
    pd.DataFrame(gains).to_csv(out/'candidate_comparisons.csv',index=False)
    prior=pd.read_csv(ROOT/'results/review3/frontier_with_random.csv');v=pd.DataFrame(cost);front=[]
    clean=summary[summary.mechanism.eq('clean')]
    for row in clean.itertuples():
        a=v[v.budget.eq(row.budget)&v.cap_scenario.eq(row.cap_scenario)&v.policy.eq(row.policy)]
        front.append(dict(budget=row.budget,cap_scenario=row.cap_scenario,policy=row.policy,
                          missed_burden=row.missed_burden,volume_Wh_s=macro(a,'volume_Wh_s')))
    frontier=pd.concat([prior.drop(columns='dominated_within_budget'),pd.DataFrame(front)],ignore_index=True)
    frontier['dominated_within_budget']=False
    for i,r in frontier.iterrows():
        a=frontier[frontier.budget.eq(r.budget)&frontier.cap_scenario.eq(r.cap_scenario)]
        weak=a.missed_burden.le(r.missed_burden+1e-12)&a.volume_Wh_s.le(r.volume_Wh_s+1e-9)
        strict=a.missed_burden.lt(r.missed_burden-1e-12)|a.volume_Wh_s.lt(r.volume_Wh_s-1e-9)
        frontier.loc[i,'dominated_within_budget']=bool((weak&strict).any())
    frontier.to_csv(out/'expanded_frontier.csv',index=False)
    files=['retrieval_validation.json','candidate_histories.csv','candidate_cases.csv','candidate_recovery.csv',
           'candidate_summary.csv','candidate_comparisons.csv','expanded_frontier.csv']
    gate=dict(passed=True,new_candidate_histories=count,records=440,new_fits=0,new_blind_cases=0,
              role='registered already-seen DEV candidate comparison; original primary unchanged',
              candidate_success_required=False,false_rejections_of_nominal_compatible_reference=0,
              output_hashes=[dict(path=p,sha256=sha(out/p)) for p in files])
    write(out/'retrieval_gate.json',gate);print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review4');args=parser.parse_args();main(ROOT/args.output)
