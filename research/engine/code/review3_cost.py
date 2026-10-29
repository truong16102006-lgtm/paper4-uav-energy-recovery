"""Derived exact-counter recovery frontier including the new random controls.

This descriptive extension is registered separately before its computation.
Runtime files remain measured observations, not deterministic reproducibility gates.
"""
from review3_common import ROOT,sha,write,macro
import json,argparse
import numpy as np
import pandas as pd
import joblib


def main(out):
    reg=json.loads((ROOT/'audit/review3_cost_registration.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'code/review3_cost.py')==reg['code_sha256']
    records={r['flight_id']:r for r in joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')}
    h=pd.read_csv(out/'noisy_histories.csv');h=h[h.mechanism.eq('clean')]
    volumes=[]
    for row in h.itertuples():
        r=records[row.flight_id];idx=np.array(json.loads(row.indices),int)
        E=np.diff(r['truth'][idx]);T=np.diff(r['t'][idx])
        assert np.min(E)>=-1e-8 and np.max(E-2000*T/3600)<=1e-8
        J=float(np.sum(E*(T-3600*E/2000)))
        volumes.append(dict(flight_id=row.flight_id,source_id=row.source_id,platform=row.platform,
                            budget=row.budget,policy=row.policy,volume_Wh_s=J))
    v=pd.DataFrame(volumes);v.to_csv(out/'random_recovery_per_record.csv',index=False)
    nominal=pd.read_csv(out/'noisy_cases.csv');nominal=nominal[nominal.mechanism.eq('clean')]
    keys=['flight_id','source_id','platform','budget','policy']
    nominal=nominal.merge(v,on=keys,validate='many_to_one')
    random=nominal[nominal.policy.str.startswith('random')].groupby(['flight_id','source_id','platform','budget','cap_scenario'])[['missed','volume_Wh_s']].mean().reset_index();random['policy']='random_mean'
    pool=pd.concat([nominal[~nominal.policy.str.startswith('random')],random],ignore_index=True)
    rows=[]
    for (B,scenario,policy),a in pool.groupby(['budget','cap_scenario','policy'],sort=True):
        rows.append(dict(budget=B,cap_scenario=scenario,policy=policy,missed_burden=macro(a,'missed'),volume_Wh_s=macro(a,'volume_Wh_s')))
    frontier=pd.DataFrame(rows);frontier['dominated_within_budget']=False
    for i,row in frontier.iterrows():
        a=frontier[frontier.budget.eq(row.budget)&frontier.cap_scenario.eq(row.cap_scenario)]
        weak=a.missed_burden.le(row.missed_burden+1e-12)&a.volume_Wh_s.le(row.volume_Wh_s+1e-9)
        strict=a.missed_burden.lt(row.missed_burden-1e-12)|a.volume_Wh_s.lt(row.volume_Wh_s-1e-9)
        frontier.loc[i,'dominated_within_budget']=bool((weak&strict).any())
    frontier.to_csv(out/'frontier_with_random.csv',index=False)
    old=pd.read_csv(ROOT/'results/review2/descriptive_frontier.csv')
    z=old[old.cap_scenario.isin(['q0.9','q0.999'])].merge(frontier,on=['budget','cap_scenario','policy'],suffixes=('_old',''),validate='one_to_one')
    assert np.max(np.abs(z.volume_Wh_s_old-z.volume_Wh_s))<1e-7
    # The decision guard now uses Wh; flag any clean decision differences explicitly.
    prior=pd.read_csv(ROOT/'results/review1/query_per_record.csv')
    prior=prior[prior.policy.isin(['uniform','audit','proxy_centers','proxy_brackets'])&prior.cap_scenario.isin(['q0.9','q0.999'])]
    z=prior.merge(nominal,on=keys+['cap_scenario'],suffixes=('_old',''),validate='one_to_one')
    changed=int((z.rejected_old!=z.rejected).sum())
    paths=['random_recovery_per_record.csv','frontier_with_random.csv']
    gate=dict(passed=True,clean_rows_compared=len(z),clean_decision_guard_differences=changed,
              old_recovery_macro_volumes_reproduced=True,role='descriptive post-review random-control frontier, exact-counter 2000 W recovery class',
              output_hashes=[dict(path=p,sha256=sha(out/p)) for p in paths])
    write(out/'cost_gate.json',gate);print(json.dumps(gate,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review3');args=parser.parse_args();main(ROOT/args.output)
