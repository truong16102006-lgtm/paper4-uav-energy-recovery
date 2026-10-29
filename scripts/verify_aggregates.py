"""Independent regrouping of saved case-level outcomes, with no source waveforms."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=R/'REPOSITORY_MANIFEST.json'
if manifest.exists():
    for x in json.loads(manifest.read_text())['files']:assert digest(R/x['path'])==x['sha256'],x['path']
D=R/'results/concise_20261029';f=pd.read_csv(D/'primary_records.csv');s=json.loads((D/'primary_statistics.json').read_text());checks=[]
for co,n in [('AMOVFLY',247),('AtlantikSolar',161)]:
    z=f[f.cohort.eq(co)];assert len(z)==n and z.record_id.nunique()==n
    assert (z.delta_pp>=-1e-8).all()
    row=next(x for x in s['analyses'] if x['cohort']==co);assert abs(z.delta_pp.mean()-row['mean_delta_pp'])<1e-10
    assert abs(z.envelope_percent.mean()-row['mean_envelope_percent'])<1e-10
    assert abs(z.tube_percent.mean()-row['mean_tube_percent'])<1e-10
    base=pd.read_csv(D/'baseline_cases.csv');base=base[base.cohort.eq(co)]
    assert abs(100*base[base.method.eq('envelope')].unresolved_fraction.mean()-z.envelope_percent.mean())<1e-8
    ab=pd.read_csv(D/'prior_ablation.csv');ab=ab[ab.cohort.eq(co)]
    assert len(ab)==3*n and ab.groupby('record_id').radius_Wh.nunique().eq(1).all()
    checks.append({'cohort':co,'records':n,'mean_delta_pp':z.delta_pp.mean()})
lp=pd.read_csv(D/'direct_LP_checks.csv');assert len(lp)==2040 and lp.max_difference_Wh.max()<1e-8
burst=pd.read_csv(D/'burst_cases.csv');assert len(burst)==12240 and burst['seed'].nunique()==5
pivot=burst.pivot(index=['cohort','record_id','mean_bad_run','seed'],columns='method',values='unresolved_fraction');assert (pivot.envelope<=pivot.center_radius+1e-10).all()
screen=pd.read_csv(D/'sensor_range_ledger.csv');assert len(screen)==161 and screen.range_eligible.sum()==140
solar=pd.read_csv(R/'results/external_solar/external_cases.csv');solar=solar[(solar.cap_W==500)&(solar.budget==8)&(solar.epsilon_Wh==0)&(solar.halfwidth_s==.5)&(solar.risk_factor==1)&solar.prior.eq('PCHIP')]
assert len(solar)==161 and abs(100*(1-solar.continuous_RMSE_Wh.mean()/solar.midpoint_RMSE_Wh.mean())-29.108161978822288)<1e-9
def macro(frame, columns):
    return frame.groupby(['platform','source_id'])[columns].mean().groupby('platform').mean().mean()

# Independently recompute the retained Table 4 development results.
fc=pd.read_csv(R/'results/fast_centers/fast_cases.csv')
nc=pd.read_csv(R/'results/new_centers/center_cases.csv')
def b8clean(frame):
    return frame[(frame.budget==8)&frame.policy.eq('uniform')&(frame.epsilon_Wh==0)&(frame.halfwidth_s==.5)&(frame.risk_factor==1)]
fc=b8clean(fc);nc=b8clean(nc)
development={}
for name,expected in [('PCHIP',0.3532340864259209),('HGB',0.3945545160030367)]:
    z=fc[fc.prior.eq(name)];assert len(z)==440
    m=macro(z,['continuous_RMSE_Wh','midpoint_RMSE_Wh','radius_Wh'])
    assert abs(m.continuous_RMSE_Wh-expected)<1e-12
    raw=macro(nc[nc.prior.eq(name)],['raw_RMSE_Wh']).raw_RMSE_Wh
    development[name]=dict(clip_RMSE_Wh=float(m.continuous_RMSE_Wh),raw_RMSE_Wh=float(raw))
    assert abs(m.midpoint_RMSE_Wh-0.5304711977949378)<1e-12

joint=pd.read_csv(R/'results/joint_refinement/cases.csv')
jn=json.loads((R/'results/article/joint_numbers.json').read_text())
for model in ['coupled_calibration','shared_affine']:
    for h in [0.,.5]:
        z=joint[joint.model.eq(model)&joint.group.eq('full_primary')&joint.halfwidth_s.eq(h)]
        assert len(z)==247
        expected=jn[model][str(h)]
        for col in ['RMSE_Wh','old_RMSE_Wh','old_radius_Wh','radius_upper_Wh','radius_lower_Wh','uniform_temporal_floor_Wh']:
            assert abs(z[col].mean()-expected[col])<1e-10,(model,h,col)
assert joint.shape_solver_mode.value_counts().to_dict()=={'continuous_L2_QP':1740,'weighted_node_L1_LP':16}
cert=pd.read_csv(R/'results/joint_certificates/cases.csv')
cs=pd.read_csv(R/'results/joint_certificates/summary.csv')
for _,row in cs.iterrows():
    z=cert[cert.model.eq(row.model)&cert.group.eq('full_primary')&cert.halfwidth_s.eq(row.halfwidth_s)]
    assert len(z)==247 and z.candidate_joint_member.all()
    assert abs(z.relative_suboptimality_upper.mean()-row.mean_relative_excess)<1e-12
    assert abs(z.relative_suboptimality_upper.max()-row.maximum_relative_excess)<1e-12
timing=pd.read_csv(R/'results/fast_timing/paired_timing.csv')
tg=json.loads((R/'results/fast_timing/timing_gate.json').read_text())
assert len(timing)==392 and timing.source_id.nunique()==56
tm=timing.groupby('source_id')[['clip_seconds','QP_seconds']].median()
speed=float((tm.QP_seconds/tm.clip_seconds).median())
assert abs(speed-tg['median_source_speedup'])<1e-9
report=dict(passed=True,checks=checks,LP_solves=4080,LP_max_difference_Wh=lp.max_difference_Wh.max(),original_solar_reduction_percent=29.108161978822288,source_waveforms_required=False,
            development_controls=development,joint_primary_groups_verified=4,joint_solver_modes=joint.shape_solver_mode.value_counts().to_dict(),joint_certificate_groups_verified=len(cs),
            timing_source_median_speedup=speed)
(R/'AGGREGATE_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
