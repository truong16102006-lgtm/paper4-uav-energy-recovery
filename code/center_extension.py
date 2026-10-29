"""All registered priors projected into minimax-center and full feasible classes."""
from common4 import *
from integrated_core import boxes
from minimax_centers import CenterProblem,l2_error
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,joblib,time,sqlite3,io,argparse

def registration():
    reg=json.loads((ROOT/'audit/centers_registration.json').read_text())
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']

def main(output):
    check();registration();output.mkdir(parents=True,exist_ok=True);history=pd.read_csv(ROOT/'results/new_query/query_histories.csv')
    h=history[history.cap_scenario.eq('q0.9')&((history.policy.eq('uniform'))|(history.policy.eq('guard_dense')&history.budget.eq(8)))&((history.epsilon_Wh.eq(0)&history.mechanism.eq('clean'))|(history.epsilon_Wh.eq(.1)&history.mechanism.eq('drift')))].copy()
    assert len(h)==3520
    rec={r['flight_id']:r for r in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib')};rows=[];start=time.perf_counter();problems=0;failed=0
    conn=sqlite3.connect(output/'center_curves.sqlite3');conn.execute('CREATE TABLE IF NOT EXISTS curves (problem_id TEXT PRIMARY KEY,metadata TEXT NOT NULL,arrays BLOB NOT NULL)');conn.execute('DELETE FROM curves')
    with threadpool_limits(limits=2):
        for si,source in enumerate(sorted(h.source_id.unique())):
            model=joblib.load(ROOT/'research/engine/evidence/v3/models/proxy'/(hashlib.sha256(source.encode()).hexdigest()[:12]+'.joblib'))
            for row in h[h.source_id.eq(source)].itertuples():
                r=rec[row.flight_id];t=r['t'];idx=np.array(json.loads(row.indices),int);y=np.array(json.loads(row.observed_energy_Wh));proxy=np.maximum(model.predict((r['x'][:-1]+r['x'][1:])/2),1.);curve=np.r_[0.,np.cumsum(proxy*np.diff(t)/3600)];curve*=max(y[-1]-y[0],0)/max(curve[-1],1e-15)
                for halfwidth in [0.,.5]:
                    a,b=boxes(t[idx],halfwidth);problem=CenterProblem(t,a,b,y,2000.,row.epsilon_Wh);truth=np.interp(problem.t,t,r['truth']-r['truth'][0]);prior={'HGB':np.interp(problem.t,t,curve),'Linear':np.interp(problem.t,t[idx],y-y[0]),'PCHIP':PchipInterpolator(t[idx],y-y[0])(problem.t)}
                    base=dict(flight_id=row.flight_id,source_id=source,platform=row.platform,budget=row.budget,policy=row.policy,epsilon_Wh=row.epsilon_Wh,mechanism=row.mechanism,halfwidth_s=halfwidth);arrays=dict(t=problem.t,truth=truth,midpoint=problem.mid,lower=problem.lower,upper=problem.upper);metadata=dict(**base,indices=idx.tolist(),observed=y.tolist(),radius_Wh=problem.R)
                    for risk in [1.,2.]:
                        for name,p in prior.items():
                            estimate,diag=problem.solve(p,risk);failed+=not diag['optimized'];assert diag['conditional_worst_Wh']<=risk*problem.R+1e-8
                            error=l2_error(problem.t,estimate-truth);raw=l2_error(problem.t,p-truth);mid=l2_error(problem.t,problem.mid-truth)
                            if risk==2 and diag['optimized']:assert error<=raw+1e-7,(base,name,error,raw)
                            rows.append(dict(**base,risk_factor=risk,prior=name,continuous_RMSE_Wh=error,raw_RMSE_Wh=raw,midpoint_RMSE_Wh=mid,peak_error_Wh=float(np.max(abs(estimate-truth))),**diag))
                            arrays[f'{name}_k{int(risk)}']=estimate;arrays[f'prior_{name}']=p;metadata[f'{name}_k{int(risk)}']=diag
                    key=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()[:24];buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);conn.execute('INSERT INTO curves VALUES (?,?,?)',(key,json.dumps(metadata),buffer.getvalue()));problems+=1
            conn.commit();print('MINIMAX CENTERS',si+1,'/56',problems,'histories',len(rows),'QPs','fallback',failed,round(time.perf_counter()-start,1),'s',flush=True)
    conn.close();f=pd.DataFrame(rows);f.to_csv(output/'center_cases.csv',index=False);group=['budget','policy','epsilon_Wh','mechanism','halfwidth_s','risk_factor','prior'];metrics=['continuous_RMSE_Wh','raw_RMSE_Wh','midpoint_RMSE_Wh','peak_error_Wh','radius_Wh','conditional_worst_Wh','optimized']
    summary=[dict(zip(group,key),records=len(z),**{c:macro(z,c) for c in metrics}) for key,z in f.groupby(group,sort=True)];pd.DataFrame(summary).to_csv(output/'center_summary.csv',index=False)
    write(output/'centers_gate.json',dict(passed=True,all_registered_cases_retained=True,records=len(rec),curve_histories=problems,QP_evaluations=len(rows),optimized=int(f.optimized.sum()),fallbacks=failed,max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_risk_residual_Wh=float(f.risk_residual_Wh.max()),median_seconds=float(f.seconds.median()),p95_seconds=float(f.seconds.quantile(.95)),seconds=time.perf_counter()-start))
    print('MINIMAX CENTERS COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/new_centers');args=p.parse_args();main(ROOT/args.output)
