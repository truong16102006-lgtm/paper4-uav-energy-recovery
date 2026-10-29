"""Replay all retained transcripts through the same noisy recovery contract."""
from review5_common import ROOT,sha,write,check_registration
from review5_envelopes_v1 import envelopes,evaluate
from review3_common import macro
import argparse,csv,json,time
import numpy as np
import pandas as pd
import joblib


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True)
    records={r['flight_id']:r for r in joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')}
    rows=[];count=0;node_checks=0;started=time.perf_counter();maxcontain=0.;maxexact=0.
    for name,expected in [('review3/noisy_histories.csv',142560),('review4/candidate_histories.csv',63360)]:
        local=0
        with (ROOT/'results'/name).open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                r=records[row['flight_id']];idx=np.array(json.loads(row['indices']),int)
                t,truth=r['t'],r['truth'];y=np.array(json.loads(row['observed_energy_Wh']),float);eps=float(row['epsilon_Wh'])
                assert np.max(np.abs(y-truth[idx]))<=eps+1e-10
                got=envelopes(t[idx],y,2000.,eps)
                base=dict(flight_id=row['flight_id'],source_id=row['source_id'],platform=row['platform'],budget=int(row['budget']),
                          cap_scenario=row.get('cap_scenario','shared'),policy=row['policy'],epsilon_Wh=eps,
                          mechanism=row['mechanism'],status=got['status'])
                if got['status']=='CONDITIONAL_ENVELOPE':
                    E,T=np.diff(truth[idx]),np.diff(t[idx]);exact=float(np.sum(E*(T-3600*E/2000)))
                    assert got['volume_Wh_s']>=exact-1e-8
                    lo,hi=evaluate(got,t-t[0]);actual=truth-truth[0]
                    residual=float(max(np.max(lo-actual),np.max(actual-hi),0.));maxcontain=max(maxcontain,residual)
                    assert residual<1e-8
                    midpoint=(lo+hi)/2;error=midpoint-actual;node_checks+=len(t)
                    assert np.max(np.abs(error))<=got['minimax_radius_Wh']+1e-8
                    if eps==0:
                        discrepancy=abs(got['volume_Wh_s']-exact);maxexact=max(maxexact,discrepancy)
                        assert discrepancy<max(1e-8,1e-9*abs(exact))
                    base.update(volume_Wh_s=got['volume_Wh_s'],minimax_radius_Wh=got['minimax_radius_Wh'],
                                exact_same_indices_volume_Wh_s=exact,
                                relative_volume_inflation=(got['volume_Wh_s']/exact-1) if exact>1e-12 else np.nan,
                                midpoint_grid_RMSE_Wh=float(np.sqrt(np.mean(error**2))),
                                midpoint_grid_max_error_Wh=float(np.max(np.abs(error))))
                rows.append(base);count+=1;local+=1
                if count%25000==0:print('RECOVERY HISTORIES',count,round(time.perf_counter()-started,1),'s',flush=True)
        assert local==expected
    frame=pd.DataFrame(rows);assert len(frame)==205920
    frame.to_csv(out/'noisy_recovery_cases.csv',index=False)
    # Original policies are cap-independent; attach each cap once for comparison.
    shared=frame[frame.cap_scenario.eq('shared')];parts=[frame[~frame.cap_scenario.eq('shared')]]
    for cap in ['q0.9','q0.999']:
        part=shared.copy();part['cap_scenario']=cap;parts.append(part)
    joined=pd.concat(parts,ignore_index=True)
    keys=['flight_id','source_id','platform','budget','cap_scenario','epsilon_Wh','mechanism']
    columns=['volume_Wh_s','minimax_radius_Wh','exact_same_indices_volume_Wh_s','relative_volume_inflation',
             'midpoint_grid_RMSE_Wh','midpoint_grid_max_error_Wh']
    usable=joined[joined.status.eq('CONDITIONAL_ENVELOPE')]
    random=usable[usable.policy.str.startswith('random')].groupby(keys)[columns].mean().reset_index();random['policy']='random_mean'
    pool=pd.concat([usable[~usable.policy.str.startswith('random')],random],ignore_index=True)
    summaries=[]
    for key,a in pool.groupby(['budget','cap_scenario','epsilon_Wh','mechanism','policy'],sort=True):
        B,cap,eps,kind,policy=key
        summaries.append(dict(budget=B,cap_scenario=cap,epsilon_Wh=eps,mechanism=kind,policy=policy,
                              records=len(a),undefined_inflation_records=int(a.relative_volume_inflation.isna().sum()),
                              **{column:macro(a,column) for column in columns}))
    summary=pd.DataFrame(summaries)
    decision=pd.concat([pd.read_csv(ROOT/'results/review3/noisy_summary.csv'),
                        pd.read_csv(ROOT/'results/review4/candidate_summary.csv')],ignore_index=True)
    group=['budget','cap_scenario','epsilon_Wh','mechanism','policy']
    summary=summary.merge(decision[group+['missed_burden','rejection_fraction']],on=group,how='left',validate='one_to_one')
    assert summary.missed_burden.notna().all();summary['dominated_within_condition']=False
    for _,a in summary.groupby(['budget','cap_scenario','epsilon_Wh','mechanism']):
        for i,r in a.iterrows():
            weak=a.missed_burden.le(r.missed_burden+1e-12)&a.volume_Wh_s.le(r.volume_Wh_s+1e-8)
            strict=a.missed_burden.lt(r.missed_burden-1e-12)|a.volume_Wh_s.lt(r.volume_Wh_s-1e-8)
            summary.loc[i,'dominated_within_condition']=bool((weak&strict).any())
    summary.to_csv(out/'noisy_recovery_summary.csv',index=False)
    files=['noisy_recovery_cases.csv','noisy_recovery_summary.csv']
    gate=dict(passed=True,saved_histories_replayed=count,original_histories=142560,candidate_histories=63360,
              status_counts={str(k):int(v) for k,v in frame.status.value_counts().items()},reference_grid_nodes_checked=node_checks,
              maximum_reference_containment_residual_Wh=maxcontain,maximum_clean_reduction_difference_Wh_s=maxexact,
              recovery_cap_W=2000.,original_queries_changed=False,new_fits=0,new_blind_cases=0,
              observation_contract='same saved noisy readings and error boxes; conditional increment recovery',
              output_hashes=[dict(path=p,sha256=sha(out/p)) for p in files])
    write(out/'recovery_gate.json',gate);print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review5');a=p.parse_args();main(ROOT/a.output)
