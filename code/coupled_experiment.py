"""Post-development paired study of joint calibration and growing prefixes."""
from common4 import *
from coupled_calibration import CoupledClass
from calibration_enclosure import enclosure
from quadratic_reference import Reference
from integrated_core import boxes
from query_validated_prior import candidates
from fast_center import recover_clipped
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,sqlite3,io,time,argparse

def counter(ref,x,epsilon=.02):
    dt=np.diff(ref.t);mid=(ref.t[:-1]+ref.t[1:])/2;gain=.05*np.sin(2*np.pi*mid/120);bias=5*np.cos(2*np.pi*mid/120);p0=(1+gain)*ref.p[:-1]+bias;p1=(1+gain)*ref.p[1:]+bias;prefix=np.r_[0,np.cumsum((p0+p1)*dt/7200)];x=np.asarray(x,float);j=np.clip(np.searchsorted(ref.t,x,side='right')-1,0,len(dt)-1);u=x-ref.t[j]
    return 23+prefix[j]+(p0[j]*u+(p1[j]-p0[j])*u*u/(2*dt[j]))/3600+epsilon*np.sin(2*np.pi*x/60)

def truncated(ref,end):
    t=np.unique(np.r_[ref.t[ref.t<end],end]);return Reference(t,np.interp(t,ref.t,ref.p))

def main(output):
    registration=json.loads((ROOT/'audit/coupled_registration_v2.json').read_text())
    for item in registration['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert json.loads((ROOT/'results/coupled_validation/validation_gate.json').read_text())['passed']
    ledger=pd.read_csv(ROOT/'results/robust_transfer/evaluation_ledger.csv');test=ledger[ledger.role.eq('TEST')&ledger.evaluation_status.eq('ELIGIBLE')];subset=[]
    for aircraft,z in test.groupby('aircraft'):subset+=sorted(z.record_id,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:8]
    assert len(test)==247 and len(subset)==24;selected=set(subset);native={fid:Reference(z.time_s.to_numpy(),z.power_W.to_numpy()) for fid,z in pd.read_parquet(ROOT/'data/external_amovfly/native_power.parquet').groupby('record_id')};output.mkdir(parents=True,exist_ok=True);write(output/'subset.json',subset);con=sqlite3.connect(output/'curves.sqlite3');con.execute('CREATE TABLE IF NOT EXISTS curves(case_id INTEGER PRIMARY KEY,metadata TEXT,arrays BLOB)');con.execute('DELETE FROM curves');rows=[];start=time.perf_counter();previous={}
    with threadpool_limits(limits=2):
        for number,row in enumerate(test.sort_values('record_id').itertuples(),1):
            ref=native[row.record_id];jobs=[('full_primary',1.,8,'full',h,33) for h in [0.,.5]]
            if row.record_id in selected:
                jobs += [('budget_loss',1.,B,delivery,h,33) for B in [4,16] for delivery in ['full','half_loss'] for h in [0.,.5]]
                jobs += [('prefix',fraction,8,'full',h,33) for fraction in [.25,.5,.75] for h in [0.,.5]]
                jobs += [('refinement',1.,8,'full',h,G) for G in [65,129] for h in [0.,.5]]
            for group,fraction,B,delivery,h,G in jobs:
                T=ref.t[-1]*fraction;prefix=truncated(ref,T);all_anchors=np.linspace(0,T,B);keep=np.ones(B,dtype=bool)
                if delivery=='half_loss':keep[1:-1:2]=False
                anchors=all_anchors[keep];a,b=boxes(anchors,h);actual=anchors+h*np.sin(2*np.pi*anchors/T);actual[0]=0;actual[-1]=T;y=counter(ref,actual);half=np.minimum(anchors-a,b-anchors);model=CoupledClass(anchors,y,4000,.02,.05,5,half)
                initial=np.unique(np.r_[np.linspace(0,T,257),anchors]);en=enclosure(a,b,y,4000,.02,.05,5);pt,p,_=candidates(initial,initial/T,anchors,en['centers'],4000);old=recover_clipped(pt,p,a,b,en['centers'],4000,en['epsilon']);key=(row.record_id,h);previous_knots=previous.get(key,()) if group=='refinement' else ();got=model.recover(pt,p,G,previous_knots)
                info=dict(record_id=row.record_id,aircraft=row.aircraft,date=row.date,group=group,horizon_fraction=fraction,budget=B,received_queries=len(anchors),delivery=delivery,halfwidth_s=h,grid_nodes=G,status=got['status'],all_sources_previously_inspected=True)
                if got['status']!='CHECKED_COUPLED_CENTER':rows.append(info);continue
                E=got['E'];t=got['t'];error,peak=prefix.error(t,E);olderror,oldpeak=prefix.error(old['grid'],old['increment']);assert peak<=got['continuous_radius_upper_Wh']+1e-7
                # Verify known generating energy and noiseless counter at the
                # production grid, including the nominal-time error inflation.
                gen=model.witness(t,prefix.value(t),counter(ref,t,0)-y[0]);assert gen['primal_residual_Wh']<1e-7
                old_member=model.membership(old['grid'],old['increment']);info.update(RMSE_Wh=error,peak_error_Wh=peak,old_RMSE_Wh=olderror,old_peak_error_Wh=oldpeak,old_radius_Wh=old['radius_Wh'],radius_upper_Wh=got['continuous_radius_upper_Wh'],radius_lower_Wh=got['discrete_radius_lower_Wh'],optimality_gap_Wh=got['optimality_gap_Wh'],analytic_gap_bound_Wh=got['analytic_grid_gap_bound_Wh'],old_curve_joint_member=old_member,shape_QP_optimized=got['shape_QP_optimized'],shape_QP_status=got['shape_QP_status'],LP_calls=model.calls,seconds=got['seconds'],primal_residual_Wh=got['joint_witness']['primal_residual_Wh'],clock_scope=got['clock_scope'])
                if fraction==1 and B==8 and delivery=='full':previous[key]=t.copy()
                buffer=io.BytesIO();np.savez_compressed(buffer,t=t,E=E,Q=got['Q'],lower=got['lower'],upper=got['upper'],coarse_t=got['coarse_t'],coarse_lower=got['coarse_lower'],coarse_upper=got['coarse_upper'],anchors=anchors,y=y,epsilon=model.eps,old_t=old['grid'],old_E=old['increment']);con.execute('INSERT INTO curves VALUES(?,?,?)',(len(rows),json.dumps(info),buffer.getvalue()));rows.append(info)
            if number%10==0:con.commit();print('COUPLED',number,'/247',len(rows),'cases',round(time.perf_counter()-start,1),'s',flush=True)
    con.commit();stored=con.execute('SELECT COUNT(*) FROM curves').fetchone()[0];con.close();f=pd.DataFrame(rows);f.to_csv(output/'cases.csv',index=False);keys=['group','horizon_fraction','budget','received_queries','delivery','halfwidth_s','grid_nodes'];columns=['RMSE_Wh','peak_error_Wh','old_RMSE_Wh','old_radius_Wh','radius_upper_Wh','radius_lower_Wh','optimality_gap_Wh','old_curve_joint_member','shape_QP_optimized','seconds'];summary=[]
    for key,z in f.groupby(keys):summary.append(dict(zip(keys,key),records=len(z),successful=int(z.status.eq('CHECKED_COUPLED_CENTER').sum()),**{c:float(z[c].mean()) for c in columns}))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False);assert len(f)==926
    primary=f[f.group.eq('full_primary')&f.halfwidth_s.eq(0)];report=dict(passed=bool(f.status.eq('CHECKED_COUPLED_CENTER').all()),cases=len(f),stored_curves=stored,full_primary_records=len(primary),subset_records=24,full_history_records=247,post_development=True,new_external_collection=False,shape_QP_fallbacks=int((~f.shape_QP_optimized).sum()),max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_true_risk_excess_Wh=float(np.maximum(f.peak_error_Wh-f.radius_upper_Wh,0).max()),total_LP_calls=int(f.LP_calls.sum()),seconds=time.perf_counter()-start);write(output/'gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/coupled_calibration');a=p.parse_args();main(ROOT/a.output)
