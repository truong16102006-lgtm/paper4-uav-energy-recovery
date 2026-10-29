"""All-case numerical refinement and data-only temporal ambiguity floors."""
from common4 import *
from coupled_calibration import CoupledClass
from shared_affine import SharedAffineClass
from joint_shape_solver import refine
from calibration_enclosure import enclosure
from integrated_core import boxes
from query_validated_prior import candidates
from quadratic_reference import Reference
from coupled_experiment import truncated
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,sqlite3,io,time

def main():
    for x in json.loads((ROOT/'audit/joint_refinement_registration.json').read_text())['files']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    refs={fid:Reference(z.time_s.to_numpy(),z.power_W.to_numpy()) for fid,z in pd.read_parquet(ROOT/'data/external_amovfly/native_power.parquet').groupby('record_id')};out=ROOT/'results/joint_refinement';out.mkdir(parents=True,exist_ok=True);con=sqlite3.connect(out/'curves.sqlite3');con.execute('CREATE TABLE IF NOT EXISTS curves(case_id INTEGER PRIMARY KEY,metadata TEXT,arrays BLOB)');con.execute('DELETE FROM curves');rows=[];start=time.perf_counter()
    with threadpool_limits(limits=2):
        for kind in ['coupled_calibration','shared_affine']:
            original=sqlite3.connect(ROOT/'results'/kind/'curves.sqlite3')
            for original_id,metadata,blob in original.execute('SELECT * FROM curves'):
                m=json.loads(metadata)
                with np.load(io.BytesIO(blob)) as z:
                    t,E,Q=z['t'],z['E'],z['Q'];a,y=z['anchors'],z['y'];h=m['halfwidth_s'];aa,bb=boxes(a,h);half=np.maximum(a-aa,bb-a)
                    if kind=='coupled_calibration':model=CoupledClass(a,y,4000,.02,.05,5,half);seed=np.r_[E,Q]/model.scale;bias=5
                    else:
                        model=SharedAffineClass(a,y,4000,.02,.05,3,m['declared_drift_W'],half);seed=np.r_[E/model.scale,z['R']/model.scale,m['inverse_gain'],m['inverse_bias_W']/model.scale,m['origin_parameter_Wh']/model.scale];bias=3+m['actual_drift_W']
                    initial=np.unique(np.r_[np.linspace(0,a[-1],257),a]);enc=enclosure(aa,bb,y,4000,.02,.05,bias);pt,p,_=candidates(initial,initial/a[-1],a,enc['centers'],4000);prior=np.interp(t,pt,p);got=refine(model,t,z['lower'],z['upper'],m['radius_upper_Wh'],prior,seed,kind);ref=truncated(refs[m['record_id']],a[-1]);error,peak=ref.error(t,got['E']);valid=kind=='coupled_calibration' or m['correctly_supplied_drift_bound']
                    if valid:assert peak<=m['radius_upper_Wh']+1e-7
                    # Increment intervals use only observed queries, never truth.
                    A,b,bounds=model.polytope(a);minimum=[];maximum=[]
                    for j in range(len(a)-1):
                        c=np.zeros(A.shape[1]);c[j]=-1;c[j+1]=1;lo=model.lp(c,A,b,bounds);hi=model.lp(-c,A,b,bounds);assert lo.success and hi.success;minimum.append(lo.fun*model.scale);maximum.append(-hi.fun*model.scale)
                    capacity=4000*np.diff(a)/3600;floor=float(max(0.,np.max(np.minimum(minimum,capacity-np.asarray(maximum))))/2);actual=np.diff(ref.value(a));oracle=float(np.max(np.minimum(actual,capacity-actual))/2)
                    info=dict(m,model=kind,original_case_id=original_id,unrefined_RMSE_Wh=m['RMSE_Wh'],RMSE_Wh=error,peak_error_Wh=peak,shape_solver_mode=got['mode'],shape_solver_status=';'.join(got['statuses']),refinement_seconds=got['seconds'],numerical_risk_excess_Wh=got['numerical_risk_excess_Wh'],uniform_temporal_floor_Wh=floor,evaluator_only_exact_anchor_floor_Wh=oracle);w=got['witness']
                    if kind=='shared_affine':info.update(inverse_gain=w['inverse_gain'],inverse_bias_W=w['inverse_bias_W'],origin_parameter_Wh=w['origin_parameter_Wh'])
                    buffer=io.BytesIO();arrays={key:z[key] for key in z.files};arrays.update(E=got['E'],Q=got['Q'])
                    if 'R' in got:arrays['R']=got['R']
                    np.savez_compressed(buffer,**arrays);con.execute('INSERT INTO curves VALUES(?,?,?)',(len(rows),json.dumps(info),buffer.getvalue()));rows.append(info)
                if len(rows)%100==0:con.commit();print('JOINT REFINEMENT',len(rows),'/1756',round(time.perf_counter()-start,1),'s',flush=True)
            original.close()
    con.commit();con.close();f=pd.DataFrame(rows);f.to_csv(out/'cases.csv',index=False);assert len(f)==1756;report=dict(passed=True,cases=len(f),mode_counts=f.shape_solver_mode.value_counts().to_dict(),original_outputs_unchanged=True,truth_not_used_for_solver_selection=True,seconds=time.perf_counter()-start);write(out/'gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
