"""Fixed post-development shared-affine calibration and drift study."""
from common4 import *
from shared_affine import SharedAffineClass
from calibration_enclosure import enclosure
from quadratic_reference import Reference
from integrated_core import boxes
from query_validated_prior import candidates
from fast_center import recover_clipped
from coupled_experiment import truncated
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,sqlite3,io,time,argparse

def counter(ref,x,actual_drift,epsilon=.02):
    dt=np.diff(ref.t);mid=(ref.t[:-1]+ref.t[1:])/2;xi=actual_drift*np.cos(2*np.pi*mid/120);p0=1.03*ref.p[:-1]+3+xi;p1=1.03*ref.p[1:]+3+xi;prefix=np.r_[0,np.cumsum((p0+p1)*dt/7200)];x=np.asarray(x,float);j=np.clip(np.searchsorted(ref.t,x,side='right')-1,0,len(dt)-1);u=x-ref.t[j];return 31+prefix[j]+(p0[j]*u+(p1[j]-p0[j])*u*u/(2*dt[j]))/3600+epsilon*np.sin(2*np.pi*x/60)

def main(output):
    for item in json.loads((ROOT/'audit/shared_affine_registration.json').read_text())['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert json.loads((ROOT/'results/shared_affine_validation/validation_gate.json').read_text())['passed'];ledger=pd.read_csv(ROOT/'results/robust_transfer/evaluation_ledger.csv');test=ledger[ledger.role.eq('TEST')&ledger.evaluation_status.eq('ELIGIBLE')];subset=[]
    for aircraft,z in test.groupby('aircraft'):subset+=sorted(z.record_id,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:8]
    selected=set(subset);assert len(selected)==24;refs={fid:Reference(z.time_s.to_numpy(),z.power_W.to_numpy()) for fid,z in pd.read_parquet(ROOT/'data/external_amovfly/native_power.parquet').groupby('record_id')};output.mkdir(parents=True,exist_ok=True);write(output/'subset.json',subset);rows=[];con=sqlite3.connect(output/'curves.sqlite3');con.execute('CREATE TABLE IF NOT EXISTS curves(case_id INTEGER PRIMARY KEY,metadata TEXT,arrays BLOB)');con.execute('DELETE FROM curves');start=time.perf_counter()
    with threadpool_limits(limits=2):
        for number,row in enumerate(test.sort_values('record_id').itertuples(),1):
            ref=refs[row.record_id];jobs=[('full_primary',1.,h,2.,2.) for h in [0.,.5]]
            if row.record_id in selected:
                jobs += [('drift_sensitivity',1.,h,actual,declared) for actual,declared in [(0.,0.),(2.,0.),(2.,5.),(2.,10.)] for h in [0.,.5]]
                jobs += [('prefix',f,h,2.,2.) for f in [.25,.5,.75] for h in [0.,.5]]
            for group,fraction,h,actual_drift,declared in jobs:
                T=ref.t[-1]*fraction;prefix=truncated(ref,T);anchors=np.linspace(0,T,8);a,b=boxes(anchors,h);actual=anchors+h*np.sin(2*np.pi*anchors/T);actual[0]=0;actual[-1]=T;y=counter(ref,actual,actual_drift);half=np.maximum(anchors-a,b-anchors);model=SharedAffineClass(anchors,y,4000,.02,.05,3,declared,half);initial=np.unique(np.r_[np.linspace(0,T,257),anchors]);enc=enclosure(a,b,y,4000,.02,.05,3+actual_drift);pt,p,_=candidates(initial,initial/T,anchors,enc['centers'],4000);old=recover_clipped(pt,p,a,b,enc['centers'],4000,enc['epsilon']);got=model.recover(pt,p,33);info=dict(record_id=row.record_id,aircraft=row.aircraft,date=row.date,group=group,horizon_fraction=fraction,halfwidth_s=h,actual_drift_W=actual_drift,declared_drift_W=declared,status=got['status'],post_development=True)
                if got['status']!='CHECKED_SHARED_AFFINE_CENTER':rows.append(info);continue
                t=got['t'];E=got['E'];error,peak=prefix.error(t,E);olderror,oldpeak=prefix.error(old['grid'],old['increment']);truth_member=model.membership(t,prefix.value(t));old_member=model.membership(old['grid'],old['increment']);specified=declared>=actual_drift
                if specified:assert truth_member and peak<=got['radius_upper_Wh']+1e-7
                info.update(RMSE_Wh=error,peak_error_Wh=peak,old_RMSE_Wh=olderror,old_peak_error_Wh=oldpeak,old_radius_Wh=old['radius_Wh'],radius_upper_Wh=got['radius_upper_Wh'],radius_lower_Wh=got['radius_lower_Wh'],optimality_gap_Wh=got['optimality_gap_Wh'],reference_joint_member=truth_member,old_curve_joint_member=old_member,correctly_supplied_drift_bound=specified,shape_QP_optimized=got['shape_QP_optimized'],shape_QP_status=got['shape_QP_status'],scaled_primal_residual=got['witness']['primal_residual_scaled'],LP_calls=model.calls,seconds=got['seconds'],clock_scope=got['clock_scope'],fitted_gain=got['witness']['gain'],fitted_bias_W=got['witness']['bias_W'],inverse_gain=got['witness']['inverse_gain'],inverse_bias_W=got['witness']['inverse_bias_W'],origin_parameter_Wh=got['witness']['origin_parameter_Wh']);buffer=io.BytesIO();np.savez_compressed(buffer,t=t,E=E,Q=got['Q'],R=got['R'],lower=got['lower'],upper=got['upper'],coarse_t=got['coarse_t'],coarse_lower=got['coarse_lower'],coarse_upper=got['coarse_upper'],anchors=anchors,y=y,epsilon=model.eps,old_t=old['grid'],old_E=old['increment']);con.execute('INSERT INTO curves VALUES(?,?,?)',(len(rows),json.dumps(info),buffer.getvalue()));rows.append(info)
            if number%10==0:con.commit();print('SHARED AFFINE',number,'/247',len(rows),'cases',round(time.perf_counter()-start,1),'s',flush=True)
    con.commit();stored=con.execute('SELECT COUNT(*) FROM curves').fetchone()[0];con.close();f=pd.DataFrame(rows);f.to_csv(output/'cases.csv',index=False);keys=['group','horizon_fraction','halfwidth_s','actual_drift_W','declared_drift_W'];cols=['RMSE_Wh','peak_error_Wh','old_RMSE_Wh','old_radius_Wh','radius_upper_Wh','radius_lower_Wh','optimality_gap_Wh','reference_joint_member','old_curve_joint_member','shape_QP_optimized','seconds'];summary=[]
    for key,z in f.groupby(keys):summary.append(dict(zip(keys,key),records=len(z),successful=int(z.status.eq('CHECKED_SHARED_AFFINE_CENTER').sum()),**{c:float(z[c].mean()) for c in cols}))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False);assert len(f)==830;valid=f[f.correctly_supplied_drift_bound];primary=f[f.group.eq('full_primary')&f.halfwidth_s.eq(0)];report=dict(passed=bool(f.status.eq('CHECKED_SHARED_AFFINE_CENTER').all()),cases=len(f),stored_curves=stored,primary_records=len(primary),subset_records=24,new_external_collection=False,post_development=True,shape_QP_fallbacks=int((~f.shape_QP_optimized).sum()),max_scaled_primal_residual=float(f.scaled_primal_residual.max()),max_true_risk_excess_correctly_supplied_Wh=float(np.maximum(valid.peak_error_Wh-valid.radius_upper_Wh,0).max()),total_LP_calls=int(f.LP_calls.sum()),seconds=time.perf_counter()-start);write(output/'gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/shared_affine');a=p.parse_args();main(ROOT/a.output)
