"""Replay all stored center histories with the fixed-origin clipping construction."""
from common4 import *
from fast_center import recover_clipped,norm
import numpy as np,pandas as pd,joblib,sqlite3,io,time,argparse

def main(output):
    reg=json.loads((ROOT/'audit/fast_registration.json').read_text())
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    output.mkdir(exist_ok=True,parents=True);old=pd.read_csv(ROOT/'results/new_centers/center_cases.csv').set_index(['flight_id','policy','budget','epsilon_Wh','halfwidth_s','risk_factor','prior']);records={r['flight_id']:r for r in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib')};con=sqlite3.connect(ROOT/'results/new_centers/center_curves.sqlite3');new=sqlite3.connect(output/'fast_curves.sqlite3');new.execute('CREATE TABLE IF NOT EXISTS curves(problem_id TEXT PRIMARY KEY, metadata TEXT NOT NULL, arrays BLOB NOT NULL)');new.execute('DELETE FROM curves');rows=[];start=time.perf_counter();problems=0
    for pid,metadata,blob in con.execute('SELECT * FROM curves ORDER BY problem_id'):
        meta=json.loads(metadata);r=records[meta['flight_id']];t=r['t'];idx=np.array(meta['indices']);h=meta['halfwidth_s'];a=np.maximum(0,t[idx]-h);b=np.minimum(t[-1],t[idx]+h);a[0]=b[0]=0;a[-1]=b[-1]=t[-1];y=np.array(meta['observed']);arrays={};diags={}
        with np.load(io.BytesIO(blob)) as prior:
            for k in [1,2]:
                for name in ['HGB','Linear','PCHIP']:
                    got=recover_clipped(prior['t'],prior[f'prior_{name}'],a,b,y,2000.,meta['epsilon_Wh'],k);assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER';g=got['grid'];truth=np.interp(g,prior['t'],prior['truth']);qp=np.interp(g,prior['t'],prior[f'{name}_k{k}']);oldrow=old.loc[(meta['flight_id'],meta['policy'],meta['budget'],meta['epsilon_Wh'],h,k,name)];base={key:meta[key] for key in ['flight_id','source_id','platform','budget','policy','epsilon_Wh','mechanism','halfwidth_s']}
                    rows.append(dict(**base,risk_factor=k,prior=name,continuous_RMSE_Wh=norm(g,got['increment']-truth),midpoint_RMSE_Wh=norm(g,got['midpoint']-truth),QP_RMSE_Wh=oldrow.continuous_RMSE_Wh,QP_seconds=oldrow.seconds,QP_optimized=bool(oldrow.optimized),curve_difference_vs_QP_Wh=norm(g,got['increment']-qp),seconds=got['seconds'],radius_Wh=got['radius_Wh'],conditional_worst_Wh=got['conditional_worst_Wh'],primal_residual_Wh=got['primal_residual_Wh'],risk_residual_Wh=got['risk_residual_Wh'],slope_repair_RMSE_Wh=got['prior_slope_repair_RMSE_Wh'],knots=got['knots']))
                    tag=f'{name}_k{k}';arrays[tag+'_t']=g;arrays[tag+'_E']=got['increment'];diags[tag]={key:got[key] for key in ['origin_centered_Wh','origin_interval_Wh','radius_Wh','prior_slope_repair_RMSE_Wh','status']}
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);new.execute('INSERT INTO curves VALUES(?,?,?)',(pid,json.dumps(dict(original=meta,fast=diags)),buf.getvalue()));problems+=1
        if problems%400==0:new.commit();print('FAST CENTERS',problems,'/7040',round(time.perf_counter()-start,1),'s',flush=True)
    new.commit();new.close();con.close();f=pd.DataFrame(rows);f.to_csv(output/'fast_cases.csv',index=False);group=['policy','budget','epsilon_Wh','halfwidth_s','risk_factor','prior'];columns=['continuous_RMSE_Wh','midpoint_RMSE_Wh','QP_RMSE_Wh','curve_difference_vs_QP_Wh','radius_Wh','slope_repair_RMSE_Wh'];summary=[dict(zip(group,key),records=len(z),**{c:macro(z,c) for c in columns},median_seconds=float(z.seconds.median()),QP_median_seconds=float(z.QP_seconds.median())) for key,z in f.groupby(group,sort=True)];pd.DataFrame(summary).to_csv(output/'fast_summary.csv',index=False)
    assert len(f)==42240;write(output/'fast_gate.json',dict(passed=True,histories=problems,evaluations=len(f),records=f.flight_id.nunique(),refusals=0,QP_fallbacks_in_original=int((~f.QP_optimized).sum()),max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_risk_residual_Wh=float(f.risk_residual_Wh.max()),median_seconds=float(f.seconds.median()),p95_seconds=float(f.seconds.quantile(.95)),seconds=time.perf_counter()-start));print('FAST COMPLETE',len(f),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/fast_centers');a=p.parse_args();main(ROOT/a.output)
