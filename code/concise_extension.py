"""Approved, fixed-setting concise-paper analyses on retained public-data references."""
from common4 import ROOT,sha,write
import argparse,hashlib,io,json,time,zipfile,warnings
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog,OptimizeWarning
from threadpoolctl import threadpool_limits
from telemetry_decision_replay import load_records,truth_crossing,signed_containment
from fast_center import recover_clipped
from calibration_enclosure import enclosure
from calibration_spec import power_bounds
from review6_envelopes_v2 import envelopes,evaluate
from threshold_audit import audit_thresholds

OUT=ROOT/'results/concise_20261029'
SEED=20261029
REPS=5000


def interval(values):return [float(x) for x in np.quantile(values,[.025,.975])]


def cluster_indices(groups,rng,reps=REPS):
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    for _ in range(reps):yield np.concatenate([members[j] for j in rng.integers(0,len(unique),len(unique))])


def block_indices(ids,length,rng,reps=REPS):
    ids=np.asarray(ids,int);starts=[i for i in range(len(ids)-length+1) if np.all(np.diff(ids[i:i+length])==1)]
    assert starts
    nblocks=int(np.ceil(len(ids)/length))
    for _ in range(reps):
        yield np.concatenate([np.arange(starts[j],starts[j]+length) for j in rng.integers(0,len(starts),nblocks)])[:len(ids)]


def statistics():
    original=pd.read_csv(ROOT/'results/telemetry_decisions/cases.csv')
    original=original[(original.threshold_Wh==10)&(original.budget==9)&original.mechanism.eq('bounded')]
    rows=[];strata=[];record_values=[]
    for cohort,g in original.groupby('cohort'):
        a=g[g.method.eq('envelope')].set_index('record_id').sort_index()
        b=g[g.method.eq('radius_tube')].set_index('record_id').reindex(a.index)
        delta=100*(b.unresolved_duration_fraction-a.unresolved_duration_fraction).to_numpy()
        z=a.reset_index();z['delta_pp']=delta;z['envelope_percent']=100*z.unresolved_duration_fraction
        z['tube_percent']=100*b.unresolved_duration_fraction.to_numpy();record_values.append(z)
        common=dict(cohort=cohort,records=len(z),mean_envelope_percent=float(z.envelope_percent.mean()),
            mean_tube_percent=float(z.tube_percent.mean()),mean_delta_pp=float(delta.mean()),
            median_delta_pp=float(np.median(delta)),Q1_delta_pp=float(np.quantile(delta,.25)),Q3_delta_pp=float(np.quantile(delta,.75)),
            strictly_tighter_fraction=float(np.mean(delta>1e-6)),ties_fraction=float(np.mean(abs(delta)<=1e-6)),
            finite_brackets=int(z.bracket_width_s.notna().sum()),reference_reached=int(z.reference_reached.sum()))
        if cohort=='AMOVFLY':
            draws=cluster_indices(z.date.to_numpy(),np.random.default_rng(SEED))
            common.update(interval_type='paired date-cluster percentile; conditional retained collection',clusters=z.date.nunique(),
                          bootstrap_CI95_delta_pp=interval([float(delta[idx].mean()) for idx in draws]))
            rows.append(common)
            for aircraft,h in z.groupby('aircraft'):
                strata.append(dict(cohort=cohort,aircraft=aircraft,records=len(h),dates=h.date.nunique(),
                    mean_delta_pp=float(h.delta_pp.mean()),median_delta_pp=float(h.delta_pp.median()),
                    Q1_delta_pp=float(h.delta_pp.quantile(.25)),Q3_delta_pp=float(h.delta_pp.quantile(.75)),
                    strictly_tighter_fraction=float(h.delta_pp.gt(1e-6).mean())))
        else:
            z['wid']=z.record_id.str.extract(r'(\d+)$')[0].astype(int);z=z.sort_values('wid');d=z.delta_pp.to_numpy()
            for length in [4,8,16]:
                draws=block_indices(z.wid.to_numpy(),length,np.random.default_rng(SEED+length))
                rows.append(dict(**common,interval_type='moving-block single-flight variability; not population CI',block_length=length,
                    bootstrap_CI95_delta_pp=interval([float(d[idx].mean()) for idx in draws])))
    pd.concat(record_values,ignore_index=True).to_csv(OUT/'primary_records.csv',index=False)
    write(OUT/'primary_statistics.json',dict(primary_endpoint='mean paired A_tube-A_envelope, pp; 10Wh B9 bounded/full',
        sign_follows_set_inclusion=True,resamples=REPS,seed=SEED,analyses=rows,aircraft_strata=strata))
    pd.DataFrame(strata).to_csv(OUT/'aircraft_statistics.csv',index=False)
    solar=pd.read_csv(ROOT/'results/external_solar/external_cases.csv')
    solar=solar[(solar.cap_W==500)&(solar.budget==8)&(solar.epsilon_Wh==0)&(solar.halfwidth_s==.5)&
                (solar.risk_factor==1)&solar.prior.eq('PCHIP')].sort_values('window_id')
    ci=[]
    for length in [4,8,16]:
        estimates=[100*(1-solar.continuous_RMSE_Wh.to_numpy()[i].mean()/solar.midpoint_RMSE_Wh.to_numpy()[i].mean())
                   for i in block_indices(solar.window_id.to_numpy(),length,np.random.default_rng(SEED+length))]
        ci.append(dict(block_length=length,CI95_percent=interval(estimates)))
    robust=pd.read_csv(ROOT/'results/robust_transfer/cases.csv')
    robust=robust[(robust.budget==8)&robust.delivery.eq('full')&(robust.halfwidth_s==.5)&robust.sensor.eq('ideal')]
    p=robust[robust.method.eq('PCHIP')&robust.prior_condition.eq('measurement')].set_index('record_id').sort_index()
    s=robust[robust.method.eq('validated')&robust.prior_condition.eq('original')].set_index('record_id').reindex(p.index)
    difference=(s.RMSE_Wh-p.RMSE_Wh).to_numpy()
    paired=[float(difference[i].mean()) for i in cluster_indices(p.date.to_numpy(),np.random.default_rng(SEED))]
    write(OUT/'legacy_uncertainty.json',dict(solar_records=len(solar),solar_mean_reduction_percent=100*(1-solar.continuous_RMSE_Wh.mean()/solar.midpoint_RMSE_Wh.mean()),
        solar_single_flight_block_intervals=ci,AMOVFLY_records=len(p),AMOVFLY_selector_minus_PCHIP_clip_mean_Wh=float(difference.mean()),
        AMOVFLY_paired_date_cluster_CI95_Wh=interval(paired),AMOVFLY_selector_comparator_is_clipped=True,
        no_new_independent_campaigns=True))


def transcript(record,budget=9,keep=None,gamma=.05,bias=5.,epsilon=None,cap=None,quantize=False):
    cohort,rid,aircraft,date,original_cap,eps,ref=record
    eps=eps if epsilon is None else epsilon;cap=original_cap if cap is None else cap
    T=float(ref.t[-1]);nominal=np.linspace(0,T,budget)
    if keep is None:keep=np.ones(budget,dtype=bool)
    anchors=nominal[keep];h=.5;a=np.maximum(0,anchors-h);b=np.minimum(T,anchors+h);a[0]=b[0]=0;a[-1]=b[-1]=T
    actual=anchors+h*np.sin(2*np.pi*anchors/T);actual[0]=0;actual[-1]=T
    y=(1+gamma)*ref.value(actual)+bias*actual/3600
    if quantize:y=np.floor(y*36)/36
    else:y+=eps*(2*actual/T-1)
    enc=enclosure(a,b,y,cap,eps,gamma,bias);assert enc['status']=='CONDITIONAL_CALIBRATION_RELAXATION'
    grid=np.unique(np.r_[np.linspace(0,T,513),anchors]);prior=PchipInterpolator(anchors,enc['centers'])(grid)
    got=recover_clipped(grid,prior,a,b,enc['centers'],cap,enc['epsilon'])
    assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER'
    assert signed_containment(ref,got['grid'],got['lower'],got['upper'])<1e-8
    return dict(anchors=anchors,a=a,b=b,y=y,enc=enc,grid=grid,prior=prior,got=got,cap=cap)


def bound_audit(t,lower,upper):return audit_thresholds(t,lower,upper,[10.])['thresholds'][0]


def hold_audit(tr,ref):
    a,b=tr['a'],tr['b'];enc=tr['enc'];lower=enc['lower'];upper=enc['upper'];m=tr['cap']/3600;T=ref.t[-1]
    tau=truth_crossing(ref,10.);unresolved=0.;false_below=0.;false_reached=0.
    for i in range(len(a)-1):
        left,right=float(b[i]),float(b[i+1]);lo=float(lower[i]);offset=min(0.,float(upper[i])-m*a[i])
        if lo>=10+1e-8:false_reached+=max(0.,min(right,tau)-left);continue
        crossing=(10-1e-8-offset)/m
        certainly_below_end=min(right,max(left,crossing))
        unresolved+=right-certainly_below_end
        false_below+=max(0.,certainly_below_end-max(left,tau))
    assert max(false_below,false_reached)<1e-6
    return dict(unresolved_duration_fraction=float(unresolved/T),false_below_s=false_below,false_reached_s=false_reached)


def independent_LP(tr,probe):
    a,b=tr['a'],tr['b'];enc=tr['enc'];grid=np.unique(np.r_[a,b,probe]);n=len(grid);A=[];rhs=[];m=tr['cap']/3600
    for j in range(n-1):
        row=np.zeros(n);row[j+1]=1;row[j]=-1;A.extend([row,-row]);rhs.extend([m*(grid[j+1]-grid[j]),0.])
    for ai,bi,l,u in zip(a,b,enc['lower'],enc['upper']):
        row=np.zeros(n);row[np.searchsorted(grid,ai)]=1;A.append(row);rhs.append(u)
        row=np.zeros(n);row[np.searchsorted(grid,bi)]=-1;A.append(row);rhs.append(-l)
    bounds=[(None,None)]*n;bounds[0]=(0.,0.);c=np.zeros(n);c[np.searchsorted(grid,probe)]=1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        lo=linprog(c,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'threads':1})
        hi=linprog(-c,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'threads':1})
    assert lo.success and hi.success
    return float(lo.fun),float(-hi.fun)


def select_spatial(records):
    ledger=pd.read_csv(ROOT/'results/robust_transfer/evaluation_ledger.csv',keep_default_na=False)
    ledger=ledger[ledger.role.eq('TEST')&ledger.evaluation_status.eq('ELIGIBLE')].copy()
    median=float(ledger.duration_s.median());ledger['distance']=abs(ledger.duration_s-median)
    candidates=ledger.sort_values(['distance','record_id'])
    byid={x[1]:x for x in records if x[0]=='AMOVFLY'}
    with zipfile.ZipFile(ROOT/'data/external_amovfly/AMOVFLY_pinned.zip') as z:
        prefix=z.namelist()[0]
        for row in candidates.itertuples():
            f=pd.read_csv(z.open(prefix+row.path),usecols=['time','battery_voltage','battery_current','gps_x','gps_y','gps_z']).apply(pd.to_numeric,errors='coerce')
            valid=np.isfinite(f[['time','battery_voltage','battery_current']]).all(axis=1)&f.battery_voltage.gt(0)&f.battery_current.ge(0)
            f=f[valid].sort_values('time',kind='stable').drop_duplicates('time',keep='first')
            if not np.isfinite(f[['gps_x','gps_y','gps_z']]).all().all():continue
            record=byid[row.record_id];ref=record[-1];t=f.time.to_numpy()-f.time.iloc[0];power=(f.battery_voltage*f.battery_current).to_numpy()
            assert t.shape==ref.t.shape and np.max(abs(t-ref.t))<1e-8 and np.max(abs(power-ref.p))<1e-8
            tr=transcript(record);got=tr['got'];coords=f[['gps_x','gps_y','gps_z']].to_numpy().copy();coords-=coords[0]
            np.savez_compressed(OUT/'spatial_example_private.npz',t=t,xyz=coords,truth=ref.value(t),grid=got['grid'],
                lower=got['lower'],upper=got['upper'],center=got['increment'],prior=got['prior_raw'],
                prior_grid=got['grid'],anchors=tr['anchors'],returned=tr['y'])
            write(OUT/'spatial_gate.json',dict(passed=True,record_id=row.record_id,source_path=row.path,
                selection='nearest median duration among electrically eligible records with finite coordinates; ties by ID',
                cohort_median_duration_s=median,duration_s=float(ref.t[-1]),source_local_displacement_m=True,
                axes_not_claimed_ENU_or_AGL=True,coordinate_rows=len(t),normalized_times_and_power_match_retained_adapter=True,
                GPS_used_as_recovery_input=False,raw_coordinates_excluded_from_shareable_repo=True))
            return
    raise AssertionError('No coordinate-consistent example; use approved time-budget-energy fallback.')


def experiments(records):
    baseline=[];ablation=[];LPs=[];start=time.perf_counter()
    for k,record in enumerate(records,1):
        cohort,rid,aircraft,date,cap,eps,ref=record;tr=transcript(record);got=tr['got'];t=got['grid'];R=got['radius_Wh']
        meta=dict(cohort=cohort,record_id=rid,aircraft=aircraft,date=date)
        raw=np.interp(t,tr['grid'],tr['prior']);rho=float(np.max(np.maximum(raw-got['lower'],got['upper']-raw)))
        assert rho>=R-1e-8
        variants={'envelope':(got['lower'],got['upper']),
                  'center_radius':(np.maximum(0,got['increment']-R),got['increment']+R),
                  'PCHIP_own_risk':(np.maximum(0,raw-rho),raw+rho)}
        for name,(lo,hi) in variants.items():
            audit=bound_audit(t,lo,hi)
            baseline.append(dict(**meta,method=name,unresolved_fraction=audit['unresolved_duration_fraction'],
                latest_crossing_s=audit['latest_crossing_s'],information='all delivered horizon readings',risk_Wh=R if name=='center_radius' else rho if name=='PCHIP_own_risk' else None))
        hold=hold_audit(tr,ref)
        baseline.append(dict(**meta,method='causal_hold_cap',unresolved_fraction=hold['unresolved_duration_fraction'],
            latest_crossing_s=None,information='latest reading whose upper time bound has elapsed',risk_Wh=None))
        for name,prior in [('PCHIP',tr['prior']),('Linear',np.interp(tr['grid'],tr['anchors'],tr['enc']['centers'])),('Flat',np.zeros_like(tr['grid']))]:
            result=got if name=='PCHIP' else recover_clipped(tr['grid'],prior,tr['a'],tr['b'],tr['enc']['centers'],cap,tr['enc']['epsilon'])
            rmse,peak=ref.error(result['grid'],result['increment'])
            assert peak<=R+1e-8
            raw_rmse,_=ref.error(tr['grid'],prior-prior[0])
            ablation.append(dict(**meta,prior=name,RMSE_Wh=rmse,raw_RMSE_Wh=raw_rmse,radius_Wh=R,raw_conditional_risk_Wh=float(np.max(np.maximum(result['prior_raw']-result['lower'],result['upper']-result['prior_raw'])))))
        for fraction in [.1,.3,.5,.7,.9]:
            probe=fraction*ref.t[-1];lo,hi=independent_LP(tr,probe);envlo=float(np.interp(probe,t,got['lower']));envhi=float(np.interp(probe,t,got['upper']))
            error=max(abs(lo-envlo),abs(hi-envhi));assert error<1e-8
            LPs.append(dict(**meta,fraction=fraction,LP_lower_Wh=lo,LP_upper_Wh=hi,envelope_lower_Wh=envlo,envelope_upper_Wh=envhi,max_difference_Wh=error))
        if k%80==0:print('BASELINE/ABLATION/LP',k,'/408',round(time.perf_counter()-start,1),'s',flush=True)
    pd.DataFrame(baseline).to_csv(OUT/'baseline_cases.csv',index=False);pd.DataFrame(ablation).to_csv(OUT/'prior_ablation.csv',index=False)
    pd.DataFrame(LPs).to_csv(OUT/'direct_LP_checks.csv',index=False)
    write(OUT/'direct_LP_gate.json',dict(passed=True,transcripts=len(records),solves=2*len(LPs),
        maximum_difference_Wh=max(r['max_difference_Wh'] for r in LPs),feasible_class='same uncertain-time rectangular class with fixed increment origin'))


def burst(records):
    rows=[]
    for k,record in enumerate(records,1):
        cohort,rid,aircraft,date,cap,eps,ref=record
        salt=int(hashlib.sha256(rid.encode()).hexdigest()[:8],16)
        for length in [2,4,8]:
            for seed in range(2026102901,2026102906):
                rng=np.random.default_rng(seed+salt);bad=bool(rng.random()<.1);keep=np.ones(17,dtype=bool)
                for j in range(1,16):
                    keep[j]=not bad
                    bad=bool(rng.random()>=1/length) if bad else bool(rng.random()<.1/(.9*length))
                tr=transcript(record,17,keep);got=tr['got'];t=got['grid'];R=got['radius_Wh']
                for name,lo,hi in [('envelope',got['lower'],got['upper']),('center_radius',np.maximum(0,got['increment']-R),got['increment']+R)]:
                    audit=bound_audit(t,lo,hi)
                    rows.append(dict(cohort=cohort,record_id=rid,aircraft=aircraft,date=date,mean_bad_run=length,seed=seed,
                        received=int(keep.sum()),method=name,unresolved_fraction=audit['unresolved_duration_fraction']))
        if k%100==0:print('BURST',k,'/408',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'burst_cases.csv',index=False)
    averaged=f.groupby(['cohort','record_id','aircraft','date','mean_bad_run','method']).agg(
        unresolved_fraction=('unresolved_fraction','mean'),received=('received','mean')).reset_index()
    averaged.to_csv(OUT/'burst_record_means.csv',index=False)


def sensor(records):
    components=dict(voltage_relative_error=.001,current_relative_error=.0015,voltage_bias_V=.00875,
                    current_bias_A=.00985,true_voltage_upper_V=36.,true_current_upper_A=10.)
    derived=power_bounds(**components);gamma=derived['relative_power_gain_bound'];bias=derived['additive_power_bias_bound_W']
    native=pd.read_parquet(ROOT/'data/external_solar/native_power.parquet');ledger=pd.read_csv(ROOT/'results/external_solar/window_ledger.csv').set_index('window_id')
    rows=[];screen=[]
    for record in records:
        if record[0]!='AtlantikSolar':continue
        wid=int(record[1].split('_')[-1]);window=ledger.loc[wid]
        times=native.time_s.to_numpy()
        left=max(0,int(np.searchsorted(times,window.left_s,side='right'))-1)
        right=min(len(native),int(np.searchsorted(times,window.right_s,side='left'))+1)
        local=native.iloc[left:right]
        vmax=float(local.voltage_V.max());imax=float(local.current_A.max())
        eligible=bool(vmax<=36 and imax<=10 and local.voltage_V.min()>=0 and local.current_A.min()>=0)
        screen.append(dict(window_id=wid,voltage_max_V=vmax,current_max_A=imax,range_eligible=eligible))
        if not eligible:continue
        tr=transcript(record,gamma=gamma,bias=bias,epsilon=1/36,cap=360.,quantize=True);got=tr['got'];R=got['radius_Wh'];t=got['grid']
        for name,lo,hi in [('envelope',got['lower'],got['upper']),('center_radius',np.maximum(0,got['increment']-R),got['increment']+R)]:
            audit=bound_audit(t,lo,hi);rows.append(dict(window_id=wid,method=name,unresolved_fraction=audit['unresolved_duration_fraction'],radius_Wh=R))
    pd.DataFrame(screen).to_csv(OUT/'sensor_range_ledger.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'sensor_cases.csv',index=False)
    write(OUT/'sensor_specification.json',dict(module='TI INA260EVM',datasheet='SBOS656C, December 2016, Table 7.5, 25 C, 3.3 V',
        module_range_source='https://www.ti.com/tool/INA260EVM',datasheet_url='https://www.ti.com/lit/ds/symlink/ina260.pdf',
        module_current_limit_A=10,components=components,derived=derived,cap_W=360,epsilon_Wh=1/36,halfwidth_s=.5,
        clock_bound_is_separately_supplied=True,synchronous_paired_component_abstraction=True,
        eligible_solar_windows=sum(r['range_eligible'] for r in screen),screened_solar_windows=len(screen),
        physical_calibration_established=False,actual_INA260_hardware_used=False))


def register():
    names=['PROTOCOL_CONCISE_20261029.md','code/concise_extension.py','code/telemetry_decision_replay.py','code/fast_center.py','code/threshold_audit.py',
           'code/calibration_enclosure.py','code/calibration_spec.py','results/telemetry_decisions/cases.csv']
    target=ROOT/'audit/concise_registration.json';assert not target.exists()
    write(target,dict(status='exploratory reused data; approved outline',files=[dict(path=x,sha256=sha(ROOT/x)) for x in names]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    if args.register:register();return
    registration=json.loads((ROOT/'audit/concise_registration.json').read_text())
    amendments=ROOT/'audit/concise_amendments.json';corrected=json.loads(amendments.read_text())['corrected_sha256'] if amendments.exists() else {}
    for row in registration['files']:assert sha(ROOT/row['path'])==corrected.get(row['path'],row['sha256'])
    OUT.mkdir(exist_ok=True,parents=True);start=time.perf_counter()
    with threadpool_limits(limits=2):
        records=load_records();select_spatial(records);statistics();experiments(records);burst(records);sensor(records)
    write(OUT/'gate.json',dict(passed=True,records=408,AMOVFLY_records=247,solar_windows=161,solar_flights=1,
        analysis_status='exploratory reused data',prior_ablation_curves=1224,direct_LP_solves=4080,
        burst_reconstructions=6120,bootstrap_resamples=REPS,new_model_fits=0,new_flights=0,seconds=time.perf_counter()-start,
        original_result_files_modified=False,outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(OUT.glob('*')) if p.name!='gate.json']))
    print(json.dumps(json.loads((OUT/'gate.json').read_text()),indent=2))


if __name__=='__main__':main()
