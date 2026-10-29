"""Reproduce the registered fixed-transcript counter-tolerance sensitivity."""
from common import ROOT, sha, write
from review2_counter import repair_radius, adjacent_radius, feasible_witness, lp_radius
import json, time
import numpy as np
import pandas as pd
import joblib

OUT = ROOT / 'results/review2'
EPS = [0., .001, .01, .1, .5, 1.]
TOL = 1e-10


def check_registration():
    registration = json.loads((ROOT / 'audit/review2_registration.json').read_text(encoding='utf-8'))
    for row in registration['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']


def macro(frame, column):
    return float(frame.groupby(['platform', 'source_id'])[column].mean().groupby(level=0).mean().mean())


def validate():
    rng = np.random.default_rng(20261027)
    max_error = 0.
    for k in range(600):
        n = int(rng.integers(2, 35))
        t = np.r_[0., np.cumsum(rng.uniform(.1, 4, n - 1))]
        cap = float(rng.choice([0., 100., 1000., 2000.]))
        y = rng.normal(0, .2, n) if k % 3 == 0 else np.r_[0., np.cumsum(rng.uniform(0, 3000, n - 1) * np.diff(t) / 3600)]
        eps, pair, kind = repair_radius(t, y, cap)
        brute = max([0.] + [float(max(y[i]-y[j], y[j]-y[i]-cap*(t[j]-t[i])/3600)) / 2
                            for i in range(n) for j in range(i+1,n)])
        lp = lp_radius(t, y, cap)
        error = max(abs(eps-brute), abs(eps-lp)); max_error = max(max_error, error)
        assert error < 1e-8
        z = feasible_witness(t, y, cap, eps)
        assert np.max(np.abs(z-y)) <= eps+1e-8
        assert np.min(np.diff(z)) >= -1e-8
        assert np.max(np.diff(z)-cap*np.diff(t)/3600) <= 1e-8
        assert abs(repair_radius(t, y + 17., cap)[0]-eps) < 1e-8
        assert abs(repair_radius(t, y*3., cap*3.)[0]-eps*3.) < 1e-8
        if eps > 1e-8:
            i,j = pair
            violation = y[i]-y[j] if kind == 'decrease' else y[j]-y[i]-cap*(t[j]-t[i])/3600
            assert abs(violation-2*eps) < 1e-8
    # Every adjacent cap excess fits a 0.6 Wh repair, but the shared history does not.
    t = np.array([0., 3600., 7200.]); y = np.array([0., 2., 4.]); cap = 1.
    assert adjacent_radius(t,y,cap) == .5 and repair_radius(t,y,cap)[0] == 1.
    result = dict(passed=True, synthetic_cases=600, independent_LP_solves=600,
                  max_radius_error_Wh=max_error, adjacent_only_counterexample=True,
                  threshold_equality_feasible=True, arbitrary_reading_order_values=True,
                  numerical_validation_tolerance_Wh=1e-8, decision_tolerance_Wh=TOL)
    write(OUT / 'validation.json', result)
    return result


def main():
    check_registration(); OUT.mkdir(parents=True,exist_ok=True)
    validation = validate(); print('VALIDATION', validation, flush=True)
    records = {r['flight_id']: r for r in joblib.load(ROOT/'evidence/v3/data/dev_records.joblib')}
    histories = pd.read_csv(ROOT/'results/review1/query_histories.csv')
    old = pd.read_csv(ROOT/'results/review1/query_per_record.csv')
    caps = pd.read_csv(ROOT/'results/review1/all_caps.csv')
    cap_map = {s:list(z[['cap_scenario','cap_W']].itertuples(index=False,name=None)) for s,z in caps.groupby('source_id')}
    dense = {}
    start = time.perf_counter()
    for fid,r in records.items():
        for scenario,cap in cap_map[r['source_id']]:
            dense[fid,scenario] = repair_radius(r['t'],r['truth'],cap)
    rows=[]
    for h in histories.itertuples():
        r=records[h.flight_id]; idx=np.array(json.loads(h.indices),int)
        y=np.array(json.loads(h.queried_energy_Wh),float); t=r['t'][idx]
        assert np.array_equal(y,r['truth'][idx])
        for scenario,cap in cap_map[h.source_id]:
            eps,pair,kind=repair_radius(t,y,cap); adj=adjacent_radius(t,y,cap)
            full,full_pair,_=dense[h.flight_id,scenario]
            assert eps <= full+1e-8 and adj <= eps+1e-8
            # Four anchors can expose the all-pair tolerance ceiling, not merely a slope peak.
            if h.budget>=4:
                witness=np.array(sorted({0,len(r['t'])-1,*full_pair}),int)
                assert abs(repair_radius(r['t'][witness],r['truth'][witness],cap)[0]-full)<1e-8
            rows.append(dict(flight_id=h.flight_id,source_id=h.source_id,platform=h.platform,
                budget=h.budget,policy=h.policy,cap_scenario=scenario,cap_W=cap,
                critical_epsilon_Wh=eps,adjacent_epsilon_Wh=adj,dense_epsilon_Wh=full,
                witness_left_node=int(idx[pair[0]]),witness_right_node=int(idx[pair[1]]),
                witness_kind=kind,actual_anchors=len(idx)))
    f=pd.DataFrame(rows)
    keys=['flight_id','source_id','platform','budget','policy','cap_scenario']
    f=f.merge(old[keys+['reference_violates','rejected']],on=keys,validate='one_to_one')
    f['zero_epsilon_rejects']=2*f.critical_epsilon_Wh>TOL
    assert len(f)==71280 and (f.zero_epsilon_rejects==f.rejected).all()
    f.to_csv(OUT/'critical_tolerances.csv',index=False)
    summaries=[]; contrasts=[]
    for epsilon in EPS:
        z=f.copy();z['robust_rejects']=2*z.critical_epsilon_Wh>2*epsilon+TOL
        z['adjacent_rejects']=2*z.adjacent_epsilon_Wh>2*epsilon+TOL
        z['ceiling_rejects']=np.where(z.budget.ge(4),2*z.dense_epsilon_Wh>2*epsilon+TOL,z.robust_rejects)
        z['missed']=z.reference_violates.astype(int)-z.robust_rejects.astype(int)
        z['additional']=z.robust_rejects.astype(int)-z.adjacent_rejects.astype(int)
        assert not (z.robust_rejects & ~z.reference_violates).any()
        for (scenario,B,policy),a in z.groupby(['cap_scenario','budget','policy'],sort=True):
            summaries.append(dict(epsilon_Wh=epsilon,cap_scenario=scenario,budget=B,policy=policy,
                records=len(a),sources=a.source_id.nunique(),nominal_reference_violators=int(a.reference_violates.sum()),
                robust_rejections=int(a.robust_rejects.sum()),additional_shared_history_rejections=int(a.additional.sum()),
                missed_burden=macro(a,'missed'),rejection_fraction=macro(a,'robust_rejects'),
                reachable_ceiling_fraction=macro(a,'ceiling_rejects')))
        pivot=z.pivot(index=['flight_id','source_id','platform','budget','cap_scenario'],columns='policy',values='missed').reset_index()
        for (scenario,B),a in pivot.groupby(['cap_scenario','budget'],sort=True):
            for policy in ['audit','proxy_brackets','proxy_centers','oracle']:
                b=a.copy();b['gain']=b.uniform-b[policy]
                contrasts.append(dict(epsilon_Wh=epsilon,cap_scenario=scenario,budget=B,policy=policy,gain_over_uniform=macro(b,'gain')))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'tolerance_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'tolerance_contrasts.csv',index=False)
    for _,a in summary.groupby(['cap_scenario','budget','policy']):
        assert np.all(np.diff(a.sort_values('epsilon_Wh').robust_rejections)<=0)
    # Separate exact-counter recovery objective, same original rows and weights.
    recovery=pd.read_csv(ROOT/'results/review1/recovery_cost.csv'); front=[]
    for (scenario,B,policy),a in old[~old.policy.eq('oracle')].groupby(['cap_scenario','budget','policy']):
        c=recovery[recovery.budget.eq(B)&recovery.policy.eq(policy)]
        front.append(dict(cap_scenario=scenario,budget=B,policy=policy,
            missed_burden=macro(a,'missed_violation'),volume_Wh_s=macro(c,'continuous_volume_Wh_s'),
            relative_volume_increase=macro(c,'relative_volume_increase')))
    front=pd.DataFrame(front);front['dominated_within_budget']=False;front['dominated_with_budget_as_cost']=False
    for i,row in front.iterrows():
        z=front[front.cap_scenario.eq(row.cap_scenario)]
        weak=(z.missed_burden<=row.missed_burden+1e-12)&(z.volume_Wh_s<=row.volume_Wh_s+1e-9)
        strict=(z.missed_burden<row.missed_burden-1e-12)|(z.volume_Wh_s<row.volume_Wh_s-1e-9)
        front.loc[i,'dominated_within_budget']=bool((weak&strict&z.budget.eq(row.budget)).any())
        front.loc[i,'dominated_with_budget_as_cost']=bool((weak&z.budget.le(row.budget)&(strict|z.budget.lt(row.budget))).any())
    front.to_csv(OUT/'descriptive_frontier.csv',index=False)
    files=['critical_tolerances.csv','tolerance_summary.csv','tolerance_contrasts.csv','descriptive_frontier.csv','validation.json']
    gate=dict(passed=True,records=len(records),histories=len(histories),policy_cap_budget_cases=len(f),
        tolerance_scenarios=len(EPS),case_tolerance_evaluations=len(f)*len(EPS),
        original_zero_tolerance_decisions_identical=len(f),independent_LP_solves=600,
        false_alarms_against_nominal_reference=0,monotone_tolerance_sensitivity=True,
        robust_ceiling_witnesses_checked=True,new_fits=0,new_blind_cases=0,
        role='already-seen DEV; fixed nominal transcripts; assumed deterministic tolerance; no noisy-selector trial',
        elapsed_s=round(time.perf_counter()-start,3),
        output_hashes=[dict(path='results/review2/'+p,sha256=sha(OUT/p)) for p in files])
    write(OUT/'gate.json',gate); print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    main()
