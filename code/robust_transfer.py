"""Frozen complete AMOVFLY evaluation, calibration stress and source controls."""
from common4 import *
from calibration_enclosure import enclosure,distorted_counter
from query_validated_prior import select_prior,candidates
from local_mass import adapt_local,distorted_prior
from fast_center import recover_clipped,norm
from quadratic_reference import Reference
from integrated_core import boxes
from review6_envelopes_v2 import envelopes
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,sqlite3,io,time

PRIOR_CONDITIONS=['original','smooth','step','flat']

def proposal(t,state,anchors,kind):
    if kind=='original':return np.asarray(state,float)
    if kind=='flat':return np.zeros(len(t))
    middle=(t[:-1]+t[1:])/2;u=(middle-t[0])/(t[-1]-t[0]);gain=1+.8*np.sin(2*np.pi*u) if kind=='smooth' else np.where(u<.5,.25,2.)
    return np.r_[0.,np.cumsum(np.diff(state)*gain)]

def run_method(t,state,anchors,y,a,b,cap,eps,method):
    if method=='validated':grid,p,diag=select_prior(t,state,anchors,y,a,b,cap,eps)
    elif method=='local':grid,p,diag=adapt_local(t,state,anchors,y,cap)
    elif method=='global':
        grid=t;mass=max(float(y[-1]-y[0]),0.);p=(state-state[0])*mass/(state[-1]-state[0]) if state[-1]>state[0] else mass*(t-t[0])/(t[-1]-t[0]);diag={}
    elif method=='PCHIP':grid,p,_=candidates(t,state,anchors,y,cap);diag={}
    else:raise ValueError(method)
    got=recover_clipped(grid,p,a,b,y,cap,eps,1)
    return got,diag

def reference_in_class(ref,a,b,y,eps,cap):
    eps=np.broadcast_to(np.asarray(eps,float),y.shape);lo=float(np.max(y-eps-ref.value(b)));hi=float(np.min(y+eps-ref.value(a)));return bool(ref.p.max()<=cap+1e-8 and lo<=hi+1e-8)

def evaluated(got,diag,ref,info,truth_valid):
    if got['status']!='CHECKED_CLIPPED_MINIMAX_CENTER':return dict(**info,status=got['status'],reference_in_class=truth_valid)
    error,peak=ref.error(got['grid'],got['increment']);miderror,_=ref.error(got['grid'],got['midpoint'])
    if truth_valid:assert peak<=got['radius_Wh']+1e-8,(info,peak,got['radius_Wh'])
    return dict(**info,status=got['status'],reference_in_class=truth_valid,RMSE_Wh=error,peak_error_Wh=peak,midpoint_RMSE_Wh=miderror,radius_Wh=got['radius_Wh'],primal_residual_Wh=got['primal_residual_Wh'],risk_residual_Wh=got['risk_residual_Wh'],weight=float(diag.get('weight_on_historical_shape',np.nan)),held_query_count=int(diag.get('held_query_count',0)),validation_loss_Wh2=diag.get('validation_loss_Wh2',np.nan))

def main():
    for path in ['audit/robust_external_registration.json']:
        for item in json.loads((ROOT/path).read_text())['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    D=ROOT/'data/external_amovfly';out=ROOT/'results/robust_transfer';out.mkdir(exist_ok=True,parents=True);ledger=pd.read_csv(D/'flight_ledger.csv',keep_default_na=False);data=pd.read_parquet(D/'native_power.parquet');refs={fid:Reference(z.time_s.to_numpy(),z.power_W.to_numpy()) for fid,z in data.groupby('record_id')};tree=json.loads((D/'source_tree.json').read_text());blobs={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'};ledger['source_blob']=ledger.path.map(blobs);reserved=set(ledger[ledger.role.eq('SCHEMA')].source_blob);seen_blobs={};seen_traces={};statuses=[]
    schema_traces={hashlib.sha256(np.c_[refs[fid].t,refs[fid].p].astype('<f8').tobytes()).hexdigest() for fid in ledger[ledger.role.eq('SCHEMA')].record_id if fid in refs}
    for row in ledger.sort_values('path').itertuples():
        status=row.status
        if row.source_blob in reserved and row.role!='SCHEMA':status='SCHEMA_SOURCE_ALIAS'
        elif row.source_blob in seen_blobs:status='DUPLICATE_SOURCE_ALIAS'
        else:
            seen_blobs[row.source_blob]=row.record_id
            if status=='ELIGIBLE':
                ref=refs[row.record_id];key=hashlib.sha256(np.c_[ref.t,ref.p].astype('<f8').tobytes()).hexdigest()
                if key in schema_traces and row.role!='SCHEMA':status='SCHEMA_NUMERICAL_ALIAS'
                elif key in seen_traces:status='DUPLICATE_NUMERICAL_TRACE'
                else:seen_traces[key]=row.record_id
        statuses.append((row.record_id,status))
    ledger['evaluation_status']=ledger.record_id.map(dict(statuses));ledger.to_csv(out/'evaluation_ledger.csv',index=False);previous={};rows=[];caps=[];calibration=[];refusals=[];counter=0;start=time.perf_counter();con=sqlite3.connect(out/'primary_curves.sqlite3');con.execute('CREATE TABLE IF NOT EXISTS curves(case_id INTEGER PRIMARY KEY,metadata TEXT NOT NULL,arrays BLOB NOT NULL)');con.execute('DELETE FROM curves');processed=0
    with threadpool_limits(limits=2):
        for row in ledger.sort_values(['aircraft','date_time','record_id']).itertuples():
            if row.evaluation_status!='ELIGIBLE':
                if row.evaluation_status not in ['DUPLICATE_SOURCE_ALIAS','SCHEMA_SOURCE_ALIAS','DUPLICATE_NUMERICAL_TRACE','SCHEMA_NUMERICAL_ALIAS']:
                    previous={k:v for k,v in previous.items() if k[0]!=row.aircraft}
                continue
            ref=refs[row.record_id];T=ref.t[-1];is_test=row.role=='TEST';base=dict(record_id=row.record_id,aircraft=row.aircraft,date=row.date,scenario=row.scenario,source_id=row.date,cap_W=4000.)
            for B in [4,8,16]:
                all_anchors=np.linspace(0,T,B)
                for delivery in ['full','half_loss']:
                    keep=np.ones(B,dtype=bool)
                    if delivery=='half_loss':keep[1:-1:2]=False
                    anchors=all_anchors[keep];grid=np.unique(np.r_[np.linspace(0,T,257),anchors])
                    for h in [0.,.5]:
                        a,b=boxes(anchors,h);actual=anchors+h*np.sin(2*np.pi*anchors/T);actual[0]=0.;actual[-1]=T
                        for sensor in ['ideal','bounded']:
                            gamma,bias,eps=(0.,0.,0.) if sensor=='ideal' else (.05,5.,.02);observed=distorted_counter(ref,actual,gamma,bias,eps);en=enclosure(a,b,observed,4000.,eps,gamma,bias);assert en['status']=='CONDITIONAL_CALIBRATION_RELAXATION';y=en['centers'];errors=en['epsilon'];key=(row.aircraft,row.date,B,delivery,h,sensor);old=previous.get(key) if row.date!='unknown' else None;fallback=old is None
                            state=grid/T if fallback else PchipInterpolator(old[0]/old[0][-1],old[1]-old[1][0])(grid/T)
                            info=dict(**base,budget=B,received_queries=len(anchors),delivery=delivery,halfwidth_s=h,sensor=sensor,historical_fallback=fallback)
                            if is_test:
                                valid=reference_in_class(ref,a,b,y,errors,4000.);assert valid or ref.p.max()>4000+1e-8
                                jobs=[('measurement','PCHIP',state)]+[(condition,method,proposal(grid,state,anchors,condition)) for condition in PRIOR_CONDITIONS for method in ['global','local','validated']]
                                primary=B==8 and delivery=='full' and h==.5 and sensor=='ideal'
                                for condition,method,p in jobs:
                                    got,diag=run_method(grid,p,anchors,y,a,b,4000.,errors,method);item=evaluated(got,diag,ref,dict(**info,prior_condition=condition,method=method),valid);rows.append(item)
                                    if primary and got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER':
                                        buffer=io.BytesIO();np.savez_compressed(buffer,t=got['grid'],E=got['increment']);con.execute('INSERT INTO curves VALUES(?,?,?)',(len(rows)-1,json.dumps(item),buffer.getvalue()))
                                if sensor=='bounded':
                                    nominal,diag=run_method(grid,state,anchors,observed,a,b,4000.,eps,'PCHIP');rows.append(evaluated(nominal,diag,ref,dict(**info,prior_condition='measurement',method='nominal_uncorrected'),reference_in_class(ref,a,b,observed,np.full(len(y),eps),4000.)))
                                if primary:
                                    for cap in [2000.,8000.]:
                                        enc=enclosure(a,b,observed,cap,0.,0.,0.)
                                        if enc['status']!='CONDITIONAL_CALIBRATION_RELAXATION':
                                            caps.append(dict(**{**info,'cap_W':cap},method='all',status=enc['status']));continue
                                        for method in ['PCHIP','global','local','validated']:
                                            got,diag=run_method(grid,state,anchors,enc['centers'],a,b,cap,enc['epsilon'],method);caps.append(evaluated(got,diag,ref,dict(**{**info,'cap_W':cap},prior_condition='original',method=method),reference_in_class(ref,a,b,enc['centers'],enc['epsilon'],cap)))
                            if row.date!='unknown':previous[key]=(anchors,y)
            if is_test:
                anchors=np.linspace(0,T,16);a,b=boxes(anchors,.5);actual=anchors+.5*np.sin(2*np.pi*anchors/T);actual[0]=0.;actual[-1]=T;observed=1.05*ref.value(actual)+5*actual/3600+.02*(2*actual/T-1);grid=np.unique(np.r_[np.linspace(0,T,257),anchors]);state=grid/T
                for gamma,bias in [(0.,0.),(.025,2.5),(.05,5.),(.1,10.)]:
                    en=enclosure(a,b,observed,4000.,.02,gamma,bias);info=dict(**base,declared_gain=gamma,declared_bias_W=bias,actual_gain=.05,actual_bias_W=5.)
                    if en['status']!='CONDITIONAL_CALIBRATION_RELAXATION':calibration.append(dict(**info,status=en['status'],reference_in_class=False));continue
                    valid=reference_in_class(ref,a,b,en['centers'],en['epsilon'],4000.)
                    if gamma>=.05 and bias>=5 and ref.p.max()<=4000:assert valid
                    got,diag=run_method(grid,state,anchors,en['centers'],a,b,4000.,en['epsilon'],'PCHIP');calibration.append(evaluated(got,diag,ref,info,valid))
                for kind,sign in [('reset',-1.),('outlier',1.)]:
                    bad=observed.copy();bad[len(bad)//2]+=sign*2*4000*T/3600;en=enclosure(a,b,bad,4000.,.02,.05,5.);rejected=en['status']=='CALIBRATION_CONTRACT_INCOMPATIBLE'
                    if not rejected:rejected=envelopes(a,b,en['centers'],4000.,en['epsilon'])['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE'
                    assert rejected;refusals.append(dict(record_id=row.record_id,kind=kind,rejected=rejected,interpretation='Large synthetic contract violation; not a physical fault label'))
                processed+=1
                if processed%20==0:con.commit();print('ROBUST EXTERNAL',processed,'records',len(rows),'cases',round(time.perf_counter()-start,1),'s',flush=True)
    con.commit();stored=con.execute('SELECT COUNT(*) FROM curves').fetchone()[0];con.close();f=pd.DataFrame(rows);f.to_csv(out/'cases.csv',index=False);pd.DataFrame(caps).to_csv(out/'cap_sensitivity.csv',index=False);pd.DataFrame(calibration).to_csv(out/'calibration_sweep.csv',index=False);pd.DataFrame(refusals).to_csv(out/'refusals.csv',index=False);assert len(f)==processed*324
    group=['budget','received_queries','delivery','halfwidth_s','sensor','prior_condition','method'];summary=[]
    for key,z in f.groupby(group):summary.append(dict(zip(group,key),records=len(z),successful=int(z.status.eq('CHECKED_CLIPPED_MINIMAX_CENTER').sum()),RMSE_Wh=float(z.RMSE_Wh.mean()),radius_Wh=float(z.radius_Wh.mean()),reference_in_class_fraction=float(z.reference_in_class.mean())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);report=dict(passed=True,source_candidates=277,unique_source_blobs=len(set(blobs[p] for p in ledger.path)),test_records=processed,main_cases=len(f),stored_primary_curves=stored,schema_records=3,source_collections_added=1,author_collected_flights=0,aircraft_ids=sorted(f.aircraft.unique().tolist()),collection_dates=sorted(set(f.date)-{'unknown'}),exclusion_counts=ledger.evaluation_status.value_counts().to_dict(),method_refusals=f[~f.method.eq('nominal_uncorrected')].status.value_counts().to_dict(),max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_risk_residual_Wh=float(f.risk_residual_Wh.max()),max_true_risk_excess_when_reference_in_class_Wh=float(np.maximum(f[f.reference_in_class].peak_error_Wh-f[f.reference_in_class].radius_Wh,0).max()),synthetic_refusal_cases=len(refusals),seconds=time.perf_counter()-start);write(out/'gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
