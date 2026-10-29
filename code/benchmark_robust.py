"""Matched full-pipeline CPU latency for calibration plus queried-data selection."""
from common4 import *
from calibration_enclosure import enclosure
from robust_transfer import run_method
from integrated_core import boxes
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,sqlite3,io,joblib,time,gc

def main():
    for item in json.loads((ROOT/'audit/robust_timing_registration.json').read_text())['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    records={x['flight_id']:x for x in joblib.load(ROOT/'research/engine/evidence/v3/data/dev_records.joblib')};con=sqlite3.connect(ROOT/'results/new_centers/center_curves.sqlite3');pool={}
    for pid,metadata in con.execute('SELECT problem_id,metadata FROM curves'):
        m=json.loads(metadata)
        if m['budget']==8 and m['policy']=='uniform' and m['epsilon_Wh']==0 and m['halfwidth_s']==.5:pool.setdefault(m['source_id'],[]).append((records[m['flight_id']]['t'][-1],m['flight_id'],pid))
    selected=[sorted(v)[len(v)//2] for source,v in sorted(pool.items())];assert len(selected)==56;rows=[];start=time.perf_counter()
    with threadpool_limits(limits=2):
        for index,(duration,fid,pid) in enumerate(selected):
            metadata,blob=con.execute('SELECT metadata,arrays FROM curves WHERE problem_id=?',(pid,)).fetchone();m=json.loads(metadata);r=records[fid];anchors=r['t'][np.array(m['indices'])];a,b=boxes(anchors,.5);y=np.asarray(m['observed']);t=np.unique(np.r_[np.linspace(0,duration,257),anchors])
            with np.load(io.BytesIO(blob)) as z:state=np.interp(t,z['t'],z['prior_HGB'])
            def call(method):
                en=enclosure(a,b,y,2000.,.02,.05,5.);got,diag=run_method(t,state,anchors,en['centers'],a,b,2000.,en['epsilon'],method);assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER';return got
            for method in ['PCHIP','local','validated']:call(method)
            for rep in range(7):
                order=['PCHIP','local','validated'] if rep%2==0 else ['validated','local','PCHIP']
                for method in order:
                    gc.collect();begin=time.perf_counter();call(method);elapsed=time.perf_counter()-begin;rows.append(dict(source_id=m['source_id'],flight_id=fid,method=method,repetition=rep,seconds=elapsed))
            print('ROBUST TIMING',index+1,'/56',flush=True)
    con.close();f=pd.DataFrame(rows);out=ROOT/'results/robust_timing';out.mkdir(parents=True,exist_ok=True);f.to_csv(out/'cases.csv',index=False);summary=f.groupby('method').seconds.agg(['median','max',lambda x:x.quantile(.95)]).rename(columns={'<lambda_0>':'p95'});summary.to_csv(out/'summary.csv');write(out/'gate.json',dict(passed=True,sources=56,repetitions_per_method_source=7,total_timed_calls=len(f),all_full_calibration_and_selection_cost_included=True,method_median_ms=(1000*f.groupby('method').seconds.median()).to_dict(),method_p95_ms=(1000*f.groupby('method').seconds.quantile(.95)).to_dict(),seconds=time.perf_counter()-start))

if __name__=='__main__':main()
