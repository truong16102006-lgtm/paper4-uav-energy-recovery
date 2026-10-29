"""Matched current-CPU timing on median-duration source histories; no truth ranking."""
from common4 import *
from fast_center import recover_clipped
from minimax_centers import CenterProblem
from integrated_core import boxes
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,joblib,sqlite3,io,time,gc

def main():
    records={r['flight_id']:r for r in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib')};con=sqlite3.connect(ROOT/'results/new_centers/center_curves.sqlite3');selected={}
    for pid,metadata in con.execute('SELECT problem_id,metadata FROM curves'):
        m=json.loads(metadata)
        if m['budget']==8 and m['policy']=='uniform' and m['epsilon_Wh']==0 and m['halfwidth_s']==.5:selected.setdefault(m['source_id'],[]).append((float(records[m['flight_id']]['t'][-1]),m['flight_id'],pid))
    chosen=[sorted(v)[len(v)//2] for source,v in sorted(selected.items())];assert len(chosen)==56;rows=[];begin=time.perf_counter()
    with threadpool_limits(limits=2):
        for duration,fid,pid in chosen:
            meta,blob=con.execute('SELECT metadata,arrays FROM curves WHERE problem_id=?',(pid,)).fetchone();m=json.loads(meta);r=records[fid];idx=np.array(m['indices']);a,b=boxes(r['t'][idx],.5);y=np.array(m['observed'])
            with np.load(io.BytesIO(blob)) as z:t=z['t'].copy();prior=z['prior_PCHIP'].copy()
            # One untimed warm-up per method, then seven alternating paired repetitions.
            warm=CenterProblem(t,a,b,y,2000.,0.);warm.solve(np.interp(warm.t,t,prior),1);recover_clipped(t,prior,a,b,y,2000.,0.,1)
            for rep in range(7):
                order=['clip','QP'] if rep%2==0 else ['QP','clip'];measure={}
                for method in order:
                    gc.collect();start=time.perf_counter()
                    if method=='clip':got=recover_clipped(t,prior,a,b,y,2000.,0.,1);ok=got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER'
                    else:problem=CenterProblem(t,a,b,y,2000.,0.);_,diag=problem.solve(np.interp(problem.t,t,prior),1);ok=diag['optimized']
                    measure[method]=(time.perf_counter()-start,ok)
                rows.append(dict(source_id=m['source_id'],flight_id=fid,platform=m['platform'],duration_s=duration,repetition=rep,first=order[0],clip_seconds=measure['clip'][0],QP_seconds=measure['QP'][0],clip_checked=measure['clip'][1],QP_optimized=measure['QP'][1]))
            print('PAIRED TIMING',len(rows)//7,'/56',flush=True)
    con.close();f=pd.DataFrame(rows);out=ROOT/'results/fast_timing';out.mkdir(exist_ok=True,parents=True);f.to_csv(out/'paired_timing.csv',index=False);per=f.groupby('source_id')[['clip_seconds','QP_seconds']].median();ratios=per.QP_seconds/per.clip_seconds;report=dict(passed=True,sources=56,paired_repetitions=len(f),repetitions_per_source=7,selection='median duration per source; primary B8 uniform clean h0.5; PCHIP; no error ranking',timing_scope='fresh full envelope+constructor+recovery, single prior; alternated order; one warmup; two numerical threads; same process',clip_median_s=float(f.clip_seconds.median()),QP_median_s=float(f.QP_seconds.median()),median_source_speedup=float(ratios.median()),minimum_source_speedup=float(ratios.min()),maximum_source_speedup=float(ratios.max()),QP_failed_checks=int((~f.QP_optimized).sum()),clip_failed_checks=int((~f.clip_checked).sum()),seconds=time.perf_counter()-begin);write(out/'timing_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
