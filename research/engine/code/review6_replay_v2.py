"""Paired timestamp-box stress replay at B8; no changed requests or fitting."""
from review6_common import ROOT,sha,write,check_registration
from review6_envelopes_v2 import envelopes,evaluate,feasibility_residual
from review3_common import macro
import csv,json,argparse,time
import numpy as np
import pandas as pd
import joblib


def key(row,scenario):return (row.flight_id,row.policy,float(row.epsilon_Wh),row.mechanism,scenario)


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    records={r['flight_id']:r for r in joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')}
    old=pd.concat([pd.read_csv(ROOT/'results/review3/noisy_cases.csv'),pd.read_csv(ROOT/'results/review4/candidate_cases.csv')],ignore_index=True)
    old=old[old.budget.eq(8)];decisions={key(r,r.cap_scenario):(bool(r.rejected),bool(r.reference_violates)) for r in old.itertuples()}
    prior=pd.read_csv(ROOT/'results/review5/noisy_recovery_cases.csv');prior=prior[prior.budget.eq(8)]
    geometry={key(r,r.cap_scenario):(r.volume_Wh_s,r.minimax_radius_Wh) for r in prior.itertuples()}
    caps=pd.read_csv(ROOT/'results/review1/all_caps.csv');caps=caps[caps.cap_scenario.isin(['q0.9','q0.999'])]
    capmap={s:dict(zip(a.cap_scenario,a.cap_W)) for s,a in caps.groupby('source_id')}
    recovery=[];audits=[];histories=0;checked=0;matched=0;matchedD=0;maxdifference=0.;containment=0.
    for file in ['review3/noisy_histories.csv','review4/candidate_histories.csv']:
        with (ROOT/'results'/file).open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                if int(row['budget'])!=8:continue
                histories+=1;r=records[row['flight_id']];idx=np.array(json.loads(row['indices']),int);t=r['t'];truth=r['truth']-r['truth'][0]
                times=t[idx];y=np.array(json.loads(row['observed_energy_Wh']),float);eps=float(row['epsilon_Wh']);scenario=row.get('cap_scenario','shared')
                base=dict(flight_id=row['flight_id'],source_id=row['source_id'],platform=row['platform'],policy=row['policy'],
                          epsilon_Wh=eps,mechanism=row['mechanism'],cap_scenario=scenario,budget=8)
                basekey=(row['flight_id'],row['policy'],eps,row['mechanism'],scenario)
                previous_volume=0.;nominal_volume=None;previous_rejections={}
                for h in [0.,.1,.5]:
                    a=np.maximum(times-h,times[0]);b=np.minimum(times+h,times[-1]);a[0]=b[0]=times[0];a[-1]=b[-1]=times[-1]
                    got=envelopes(a,b,y,2000.,eps)
                    assert got['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE',base
                    volume=got['volume_Wh_s'];radius=got['minimax_radius_Wh'];assert volume>=previous_volume-1e-8;previous_volume=volume
                    if h==0:
                        expected=geometry[basekey];difference=max(abs(volume-expected[0]),abs(radius-expected[1]));maxdifference=max(maxdifference,difference)
                        assert difference<max(1e-8,1e-9*abs(expected[0]));matched+=1;nominal_volume=volume
                    lo,hi=evaluate(got,t-t[0]);res=float(max(0.,np.max(lo-truth),np.max(truth-hi)));containment=max(containment,res)
                    assert res<1e-8;checked+=len(t);error=(lo+hi)/2-truth
                    recovery.append({**base,'time_halfwidth_s':h,'status':got['status'],'volume_Wh_s':volume,
                                     'minimax_radius_Wh':radius,'midpoint_grid_RMSE_Wh':float(np.sqrt(np.mean(error**2))),
                                     'volume_added_by_time_Wh_s':volume-nominal_volume,
                                     'midpoint_clock_min_gap_s':got['minimum_midpoint_clock_gap_s']})
                    for capname,cap in capmap[row['source_id']].items():
                        if scenario!='shared' and scenario!=capname:continue
                        oldD,V=decisions[(row['flight_id'],row['policy'],eps,row['mechanism'],capname)]
                        residual=feasibility_residual(a,b,y,cap,eps);D=bool(residual>1e-8)
                        assert not D or V
                        if h==0:assert D==oldD;matchedD+=1
                        else:assert not D or previous_rejections[capname]
                        previous_rejections[capname]=D
                        audits.append({**base,'cap_scenario':capname,'time_halfwidth_s':h,'reference_violates':V,
                                       'rejected':D,'missed':int(V and not D),'fixed_time_rejected':oldD,
                                       'numeric_boundary':bool(0<residual<=1e-8)})
                if histories%10000==0:print('CLOCK REPLAY',histories,'/51480',round(time.perf_counter()-start,1),'s',flush=True)
    frame=pd.DataFrame(recovery);decision=pd.DataFrame(audits)
    assert histories==51480 and len(frame)==154440 and len(decision)==261360
    frame.to_csv(out/'clock_recovery_cases.csv',index=False);decision.to_csv(out/'clock_decision_cases.csv',index=False)
    shared=frame[frame.cap_scenario.eq('shared')];parts=[frame[~frame.cap_scenario.eq('shared')]]
    for cap in ['q0.9','q0.999']:
        a=shared.copy();a['cap_scenario']=cap;parts.append(a)
    both=pd.concat(parts,ignore_index=True)
    keys=['flight_id','source_id','platform','cap_scenario','epsilon_Wh','mechanism','time_halfwidth_s','policy']
    combined=both.merge(decision[keys+['rejected','missed','numeric_boundary']],on=keys,validate='one_to_one')
    metrics=['volume_Wh_s','minimax_radius_Wh','midpoint_grid_RMSE_Wh','volume_added_by_time_Wh_s','rejected','missed','numeric_boundary']
    random=combined[combined.policy.str.startswith('random')].groupby(keys[:-1])[metrics].mean().reset_index();random['policy']='random_mean'
    pool=pd.concat([combined[~combined.policy.str.startswith('random')],random],ignore_index=True)
    summaries=[]
    for k,a in pool.groupby(['cap_scenario','epsilon_Wh','mechanism','time_halfwidth_s','policy'],sort=True):
        cap,eps,kind,h,p=k;summaries.append(dict(cap_scenario=cap,epsilon_Wh=eps,mechanism=kind,time_halfwidth_s=h,policy=p,
                                              records=len(a),**{col:macro(a,col) for col in metrics}))
    pd.DataFrame(summaries).to_csv(out/'clock_summary.csv',index=False)
    files=['clock_recovery_cases.csv','clock_decision_cases.csv','clock_summary.csv']
    gate=dict(passed=True,saved_B8_histories=histories,recovery_histories=len(frame),detection_cases=len(decision),
              zero_width_histories_matched=matched,zero_width_decisions_matched=matchedD,
              max_fixed_time_geometry_difference=maxdifference,reference_nodes_checked=checked,max_containment_residual_Wh=containment,
              recovery_status_counts={str(k):int(v) for k,v in frame.status.value_counts().items()},
              no_new_rejections_with_wider_time_boxes=True,no_false_rejection_of_nominal_compatible_reference=True,
              time_halfwidths_s=[0,.1,.5],exact_horizon_endpoints=True,new_fits=0,new_queries=0,new_blind_cases=0,
              output_hashes=[dict(path=p,sha256=sha(out/p)) for p in files])
    write(out/'clock_replay_gate.json',gate);print(json.dumps(gate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review6');a=p.parse_args();main(ROOT/a.output)
