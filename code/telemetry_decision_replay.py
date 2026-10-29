"""Fixed, exploratory telemetry-design replay; no fitting or new flight data."""
from common4 import ROOT, sha, write
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from fast_center import recover_clipped, norm
from calibration_enclosure import enclosure
from quadratic_reference import Reference
from review6_envelopes_v2 import envelopes, evaluate
from threshold_audit import audit_thresholds, first_reach
from threadpoolctl import threadpool_limits

OUT=ROOT/'results/telemetry_decisions'
BUDGETS=[5,9,17]
MECHANISMS=['ideal','bounded','alternating_loss','burst_loss']
THRESHOLDS=[5.,10.,20.]


class LinearEnergyReference:
    def __init__(self,t,e):
        self.t=np.asarray(t,float);self.energy=np.asarray(e,float)
    def value(self,t):return np.interp(t,self.t,self.energy)
    def error(self,t,curve):
        z=np.unique(np.r_[t,self.t]);d=np.interp(z,t,curve)-self.value(z)
        return norm(z,d),float(np.max(abs(d)))


def truth_crossing(ref,threshold):
    if ref.energy[-1]<threshold:return float('inf')
    j=int(np.searchsorted(ref.energy,threshold,side='left'))
    if j==0:return float(ref.t[0])
    left,right=ref.t[j-1],ref.t[j]
    for _ in range(48):
        mid=(left+right)/2
        if ref.value(mid)>=threshold:right=mid
        else:left=mid
    return float(right)


def signed_containment(ref,t,lower,upper):
    """Signed extrema on each numerical reference segment, including critical points."""
    z=np.unique(np.r_[ref.t,t]);probes=[z]
    if isinstance(ref,Reference):
        dt=np.diff(z);p0=np.interp(z[:-1],ref.t,ref.p);p1=np.interp(z[1:],ref.t,ref.p)
        delta=p1-p0
        for curve in [lower,upper]:
            slope=np.diff(np.interp(z,t,curve))/dt
            offset=np.zeros_like(delta)
            np.divide((slope*3600-p0)*dt,delta,out=offset,where=delta!=0)
            inside=(delta!=0)&(offset>0)&(offset<dt)
            probes.append(z[:-1][inside]+offset[inside])
    x=np.unique(np.concatenate(probes));truth=ref.value(x)
    return max(0.,float(np.max(np.interp(x,t,lower)-truth)),
               float(np.max(truth-np.interp(x,t,upper))))


def load_records():
    ledger=pd.read_csv(ROOT/'results/robust_transfer/evaluation_ledger.csv',keep_default_na=False)
    ledger=ledger[ledger.role.eq('TEST')&ledger.evaluation_status.eq('ELIGIBLE')]
    native=pd.read_parquet(ROOT/'data/external_amovfly/native_power.parquet')
    byid={key:z for key,z in native.groupby('record_id',sort=False)}
    records=[]
    for row in ledger.sort_values('record_id').itertuples():
        z=byid[row.record_id];ref=Reference(z.time_s.to_numpy(),z.power_W.to_numpy())
        records.append(('AMOVFLY',row.record_id,row.aircraft,row.date,4000.,.02,ref))
    solar=pd.read_parquet(ROOT/'results/external_solar/native_reference_windows.parquet')
    for wid,z in solar.groupby('window_id',sort=True):
        records.append(('AtlantikSolar',f'window_{wid:03d}','AtlantikSolar','single_flight',500.,.1,
                        LinearEnergyReference(z.time_s.to_numpy(),z.energy_Wh.to_numpy())))
    assert sum(r[0]=='AMOVFLY' for r in records)==247
    assert sum(r[0]=='AtlantikSolar' for r in records)==161
    return records


def register():
    files=['PROTOCOL_TELEMETRY_DECISIONS.md','code/threshold_audit.py',
           'code/telemetry_decision_replay.py','code/fast_center.py','code/calibration_enclosure.py',
           'code/quadratic_reference.py','research/engine/code/review6_envelopes_v2.py',
           'results/robust_transfer/evaluation_ledger.csv','data/external_amovfly/native_power.parquet',
           'results/external_solar/native_reference_windows.parquet']
    target=ROOT/'audit/telemetry_decision_registration.json'
    assert not target.exists(),'Preserve the original registration; use an explicit amendment for changed inputs/code.'
    write(target,dict(date='2026-10-29',status='exploratory reused-data extension',
                     independent_preregistration=False,files=[dict(path=x,sha256=sha(ROOT/x)) for x in files]))


def run():
    reg=json.loads((ROOT/'audit/telemetry_decision_registration.json').read_text())
    amendment=ROOT/'audit/telemetry_decision_amendment.json'
    corrected=json.loads(amendment.read_text())['corrected_sha256'] if amendment.exists() else {}
    for row in reg['files']:assert sha(ROOT/row['path'])==corrected.get(row['path'],row['sha256']),row['path']
    OUT.mkdir(exist_ok=True,parents=True);records=load_records();rows=[];nested=[];start=time.perf_counter()
    maximum_containment=0.;max_nested=0.;examples={}
    # Illustration selected by median duration before any performance is computed.
    solar=[r for r in records if r[0]=='AtlantikSolar']
    example_id=sorted(solar,key=lambda r:(r[-1].t[-1],r[1]))[len(solar)//2][1]
    with threadpool_limits(limits=2):
        for count,(cohort,rid,aircraft,date,cap,reading_eps,ref) in enumerate(records,1):
            T=float(ref.t[-1]);truth_times={theta:truth_crossing(ref,theta) for theta in THRESHOLDS}
            previous={}
            for budget in BUDGETS:
                nominal=np.linspace(0,T,budget)
                for mechanism in MECHANISMS:
                    ideal=mechanism=='ideal';h=0. if ideal else .5;eps=0. if ideal else reading_eps
                    gamma=0. if ideal else .05;bias=0. if ideal else 5.
                    keep=np.ones(budget,dtype=bool)
                    if mechanism=='alternating_loss':keep[1:-1:2]=False
                    if mechanism=='burst_loss':
                        j=(budget-1)//2;keep[j:j+2]=False
                    anchors=nominal[keep];a=np.maximum(0,anchors-h);b=np.minimum(T,anchors+h)
                    a[0]=b[0]=0.;a[-1]=b[-1]=T
                    actual=anchors+h*np.sin(2*np.pi*anchors/T);actual[0]=0.;actual[-1]=T
                    y=(1+gamma)*ref.value(actual)+bias*actual/3600+eps*(2*actual/T-1)
                    enc=enclosure(a,b,y,cap,eps,gamma,bias)
                    assert enc['status']=='CONDITIONAL_CALIBRATION_RELAXATION'
                    env=envelopes(a,b,enc['centers'],cap,enc['epsilon'])
                    assert env['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE'
                    grid=np.unique(np.r_[np.linspace(0,T,513),anchors])
                    prior=PchipInterpolator(anchors,enc['centers'])(grid)
                    got=recover_clipped(grid,prior,a,b,enc['centers'],cap,enc['epsilon'])
                    assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER'
                    t=got['grid'];lo=got['lower'];hi=got['upper'];curve=got['increment'];radius=got['radius_Wh']
                    residual=signed_containment(ref,t,lo,hi)
                    assert residual<1e-8,(rid,budget,mechanism,residual)
                    maximum_containment=max(maximum_containment,residual)
                    if mechanism in ['ideal','bounded']:
                        if mechanism in previous:
                            old=previous[mechanism];check=np.unique(np.r_[old['knots_elapsed_s'],env['knots_elapsed_s']])
                            L0,U0=evaluate(old,check);L1,U1=evaluate(env,check)
                            violation=max(0.,float(np.max(L0-L1)),float(np.max(U1-U0)))
                            max_nested=max(max_nested,violation)
                            assert violation<1e-8,(rid,budget,mechanism,violation)
                            nested.append(dict(cohort=cohort,record_id=rid,budget=budget,mechanism=mechanism,
                                               maximum_nesting_violation_Wh=violation))
                        previous[mechanism]=env
                    audits={'envelope':audit_thresholds(t,lo,hi,THRESHOLDS),
                            'radius_tube':audit_thresholds(t,np.maximum(0,curve-radius),curve+radius,THRESHOLDS)}
                    raw=np.asarray(prior)-prior[0]
                    assert np.min(np.diff(raw))>=-1e-9
                    raw_rmse,_=ref.error(grid,raw);clip_rmse,_=ref.error(t,curve)
                    for method,audit in audits.items():
                        for outcome in audit['thresholds']:
                            theta=outcome['threshold_Wh'];tau=truth_times[theta]
                            early=outcome['earliest_crossing_s'];late=outcome['latest_crossing_s']
                            e=float('inf') if early is None else early;l=float('inf') if late is None else late
                            below_wrong=max(0.,min(T,e)-min(T,tau));reached_wrong=max(0.,min(T,tau)-min(T,l))
                            assert max(below_wrong,reached_wrong)<1e-6,(rid,method,theta,below_wrong,reached_wrong)
                            if np.isfinite(tau):assert e-1e-6<=tau<=l+1e-6
                            naive=first_reach(grid,raw,theta)
                            naive_wrong=abs(min(T,naive)-min(T,tau))
                            rows.append(dict(cohort=cohort,record_id=rid,aircraft=aircraft,date=date,
                                duration_s=T,budget=budget,received=int(keep.sum()),mechanism=mechanism,
                                threshold_Wh=theta,method=method,reference_reached=bool(np.isfinite(tau)),
                                reference_crossing_s=tau if np.isfinite(tau) else None,
                                earliest_crossing_s=early,latest_crossing_s=late,
                                bracket_width_s=(l-e) if np.isfinite(l) and np.isfinite(e) else None,
                                end_decision=outcome['decisions'][0],
                                unresolved_duration_fraction=outcome['unresolved_duration_fraction'],
                                false_below_duration_s=below_wrong,false_reached_duration_s=reached_wrong,
                                naive_wrong_duration_s=naive_wrong,raw_RMSE_Wh=raw_rmse,clip_RMSE_Wh=clip_rmse,
                                containment_residual_Wh=residual,radius_Wh=radius))
                    if cohort=='AtlantikSolar' and rid==example_id and budget==9 and mechanism in ['bounded','burst_loss']:
                        examples[mechanism]=dict(t=t,lower=lo,upper=hi,center=curve,
                            tube_lower=np.maximum(0,curve-radius),tube_upper=curve+radius,
                            reference=ref.value(t),anchors=anchors,returned=y)
            if count%40==0:print('TELEMETRY',count,'of',len(records),round(time.perf_counter()-start,1),'s',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'cases.csv',index=False)
    pd.DataFrame(nested).to_csv(OUT/'nested_checks.csv',index=False)
    assert len(f)==408*3*4*3*2
    keys=['cohort','budget','mechanism','threshold_Wh','method']
    summaries=[]
    for key,g in f.groupby(keys):
        finite=g.bracket_width_s.notna();reached=g.reference_reached
        summaries.append(dict(zip(keys,key),records=len(g),received_mean=float(g.received.mean()),
            reference_reached=int(reached.sum()),finite_brackets=int(finite.sum()),
            certainly_below_at_end=int(g.end_decision.eq('CERTAINLY_BELOW').sum()),
            certainly_reached_at_end=int(g.end_decision.eq('CERTAINLY_REACHED').sum()),
            unresolved_at_end=int(g.end_decision.eq('UNRESOLVED').sum()),
            mean_unresolved_fraction=float(g.unresolved_duration_fraction.mean()),
            mean_finite_bracket_width_s=float(g.loc[finite,'bracket_width_s'].mean()) if finite.any() else None,
            false_below_duration_s=float(g.false_below_duration_s.sum()),
            false_reached_duration_s=float(g.false_reached_duration_s.sum()),
            naive_mean_wrong_duration_s=float(g.naive_wrong_duration_s.mean()),
            raw_RMSE_Wh=float(g.raw_RMSE_Wh.mean()),clip_RMSE_Wh=float(g.clip_RMSE_Wh.mean())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    stratum=f[f.threshold_Wh.eq(10)&f.method.eq('envelope')].copy()
    stratum['stratum']=np.where(stratum.cohort.eq('AMOVFLY'),stratum.aircraft+' / '+stratum.date,
                              'chronological_block_'+(pd.to_numeric(stratum.record_id.str.extract(r'(\d+)$')[0],errors='coerce').fillna(0).astype(int)//48).astype(str))
    stratum.groupby(['cohort','stratum','budget','mechanism']).agg(records=('record_id','size'),
        mean_unresolved_fraction=('unresolved_duration_fraction','mean'),
        raw_RMSE_Wh=('raw_RMSE_Wh','mean'),clip_RMSE_Wh=('clip_RMSE_Wh','mean')).reset_index().to_csv(OUT/'strata.csv',index=False)
    np.savez_compressed(OUT/'example.npz',**{m+'__'+k:v for m,item in examples.items() for k,v in item.items()})
    gate=dict(passed=True,analysis_status='exploratory previously inspected records',
        records=408,AMOVFLY_records=247,solar_windows=161,solar_flights=1,
        reconstruction_cases=4896,threshold_method_rows=len(f),thresholds_Wh=THRESHOLDS,
        primary_threshold_Wh=10.,nested_comparisons=len(nested),
        maximum_reference_containment_residual_Wh=maximum_containment,
        maximum_nesting_violation_Wh=max_nested,
        maximum_false_decision_duration_s=float(f[['false_below_duration_s','false_reached_duration_s']].max().max()),
        example_selection='median-duration solar window, tie by ID, before outcomes',example_id=example_id,
        physical_calibration=False,onboard_measurements=False,new_flights=0,models_fit=0,
        deterministic_stresses_not_independent_trials=True,seconds=time.perf_counter()-start,
        outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='gate.json'])
    write(OUT/'gate.json',gate);print(json.dumps(gate,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    register() if args.register else run()
