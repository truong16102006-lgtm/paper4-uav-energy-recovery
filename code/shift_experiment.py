"""Complete registered query-adaptation grid; no outcome-selected tuning."""
from common4 import *
from local_mass import adapt_local,distorted_prior
from quadratic_reference import Reference
from fast_center import recover_clipped,norm
from integrated_core import boxes
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,joblib,sqlite3,io,time

SHIFTS=['identity','global_2','interval_gain','ramp','smooth']

def histories():
    records={r['flight_id']:r for r in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib')}
    con=sqlite3.connect(ROOT/'results/new_centers/center_curves.sqlite3')
    for pid,metadata,blob in con.execute('SELECT * FROM curves'):
        meta=json.loads(metadata)
        if meta['policy']!='uniform':continue
        with np.load(io.BytesIO(blob)) as z:t=z['t'];truth=z['truth'];prior=z['prior_HGB']
        anchors=records[meta['flight_id']]['t'][np.asarray(meta['indices'],int)]
        info=dict(dataset='development',record_id=meta['flight_id'],source_id=meta['source_id'],platform=meta['platform'],budget=meta['budget'],epsilon_Wh=meta['epsilon_Wh'],halfwidth_s=meta['halfwidth_s'],cap_W=2000.,lagged_fallback=False)
        yield info,t,truth,prior,anchors,np.asarray(meta['observed'],float),None
    con.close();del records
    native=pd.read_parquet(ROOT/'data/external_solar/native_power.parquet');nt=native.time_s.to_numpy();npower=native.power_W.to_numpy();ledger=pd.read_csv(ROOT/'results/external_solar/window_ledger.csv');previous={};prepared={}
    for row in ledger.itertuples():
        if row.status!='ELIGIBLE':continue
        x=np.unique(np.r_[row.left_s,nt[(nt>row.left_s)&(nt<row.right_s)],row.right_s]);ref=Reference(x-row.left_s,np.interp(x,nt,npower));prepared[row.window_id]=ref
    # Counter values below come from the exact quadratic numerical reference.
    for wid,ref in prepared.items():
        t=ref.t;truth=ref.energy
        for B in [4,8,16]:
            idx=np.unique(np.rint(np.linspace(0,len(t)-1,min(B,len(t)))).astype(int));anchors=t[idx]
            for eps in [0.,.1]:
                y=ref.value(anchors)+eps*(2*anchors/t[-1]-1);key=(wid-1,B,eps);fallback=key not in previous
                if fallback:prior=t/t[-1]
                else:
                    old_t,old_y=previous[key];prior=PchipInterpolator(old_t/old_t[-1],old_y-old_y[0])(t/t[-1])
                for h in [0.,.5]:
                    info=dict(dataset='external_followup',record_id=str(wid),source_id='AtlantikSolar_2015-07-14_17',platform='solar_fixed_wing',budget=B,epsilon_Wh=eps,halfwidth_s=h,cap_W=500.,lagged_fallback=fallback)
                    yield info,t,truth,prior,anchors,y,ref
                previous[(wid,B,eps)]=(anchors,y)

def main():
    reg=json.loads((ROOT/'audit/shift_registration.json').read_text())
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert json.loads((ROOT/'results/shift_validation/validation_gate.json').read_text())['passed']
    out=ROOT/'results/shift';out.mkdir(exist_ok=True,parents=True);start=time.perf_counter();rows=[];count=0;budgets=[];con=sqlite3.connect(out/'primary_curves.sqlite3');con.execute('CREATE TABLE IF NOT EXISTS curves(case_id INTEGER PRIMARY KEY,metadata TEXT NOT NULL,arrays BLOB NOT NULL)');con.execute('DELETE FROM curves')
    with threadpool_limits(limits=2):
        for info,t,truth,prior,anchors,y,ref in histories():
            a,b=boxes(anchors,info['halfwidth_s']);primary=info['budget']==8 and info['epsilon_Wh']==0 and info['halfwidth_s']==.5
            jobs=[]
            for shift in SHIFTS:
                grid,p=distorted_prior(t,prior,anchors,shift);mass=max(float(y[-1]-y[0]),0.);global_prior=p*mass/p[-1] if p[-1]>0 else mass*(grid-grid[0])/(grid[-1]-grid[0]);local_t,local,diag=adapt_local(grid,p,anchors,y,info['cap_W'])
                jobs.extend([(shift,'unchanged',grid,p,{}),(shift,'global_endpoint',grid,global_prior,{}),(shift,'bounded_local',local_t,local,diag)])
            jobs.append(('measurement_prior','PCHIP_control',t,PchipInterpolator(anchors,y-y[0])(t),{}))
            for shift,method,grid,p,diag in jobs:
                got=recover_clipped(grid,p,a,b,y,info['cap_W'],info['epsilon_Wh'],1)
                assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER',(info,shift,method)
                g=got['grid'];E=got['increment'];mid=got['midpoint']
                if ref:error,peak=ref.error(g,E);miderror,_=ref.error(g,mid)
                else:
                    actual=np.interp(g,t,truth);error=norm(g,E-actual);peak=float(np.max(abs(E-actual)));miderror=norm(g,mid-actual)
                assert peak<=got['radius_Wh']+1e-8,(info,shift,method,peak,got['radius_Wh'])
                result=dict(**info,shift=shift,method=method,RMSE_Wh=error,peak_error_Wh=peak,midpoint_RMSE_Wh=miderror,radius_Wh=got['radius_Wh'],primal_residual_Wh=got['primal_residual_Wh'],risk_residual_Wh=got['risk_residual_Wh'],prior_repair_RMSE_Wh=got['prior_slope_repair_RMSE_Wh'],**diag);rows.append(result)
                if primary:
                    arrays=dict(t=g,E=E)
                    if ref:arrays.update(reference_t=ref.t,reference_power=ref.p)
                    else:arrays.update(reference_t=t,reference_energy=truth)
                    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);con.execute('INSERT INTO curves VALUES(?,?,?)',(len(rows)-1,json.dumps(result),buffer.getvalue()))
            if ref and info['budget']==8 and info['epsilon_Wh']==0 and info['halfwidth_s']==.5:budgets.append(dict(window_id=info['record_id'],**ref.linearization_budget()))
            count+=1
            if count%400==0:con.commit();print('SHIFT HISTORIES',count,'CASES',len(rows),round(time.perf_counter()-start,1),'s',flush=True)
    con.commit();con.close();f=pd.DataFrame(rows);f.to_csv(out/'cases.csv',index=False);pd.DataFrame(budgets).to_csv(out/'reference_error_budget.csv',index=False);keys=['dataset','budget','epsilon_Wh','halfwidth_s','shift','method'];summary=[]
    for key,z in f.groupby(keys):
        mean=lambda c:macro(z,c) if key[0]=='development' else float(z[c].mean())
        summary.append(dict(zip(keys,key),cases=len(z),RMSE_Wh=mean('RMSE_Wh'),radius_Wh=mean('radius_Wh'),midpoint_RMSE_Wh=mean('midpoint_RMSE_Wh'),max_peak_error_Wh=float(z.peak_error_Wh.max())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    assert count==7212 and len(f)==115392 and len(budgets)==161
    write(out/'gate.json',dict(passed=True,histories=count,development_histories=5280,external_histories=1932,evaluations=len(f),development_records=440,external_windows=161,new_independent_flights=0,model_fits=0,refusals=0,max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_risk_residual_Wh=float(f.risk_residual_Wh.max()),max_true_risk_excess_Wh=float(np.maximum(f.peak_error_Wh-f.radius_Wh,0).max()),external_lagged_fallback_windows=int(f[f.dataset.eq('external_followup')&f.lagged_fallback].record_id.nunique()),primary_curves_stored=9616,seconds=time.perf_counter()-start));print('SHIFT COMPLETE',count,len(f),flush=True)

if __name__=='__main__':main()
