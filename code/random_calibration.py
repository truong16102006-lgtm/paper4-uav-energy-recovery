"""Five genuinely new source-randomized fits/calibrations on the finite IDF pool."""
from common4 import *
import numpy as np,pandas as pd,joblib,time,argparse,itertools,math
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
sys.path[:0]=[str(ROOT/'research/04_RobustMoments/scripts'),str(ROOT/'research/03_Full240/scripts')]
from experiment import load,features
from core import descriptors
from tree_fallback import hold_wind

def perturb(d,scenario):
 q=d.copy().reset_index(drop=True)
 if scenario=='burst':
  elapsed=q.time_s-q.groupby('flight_id').time_s.transform('min');missing=(np.floor(elapsed/30).astype(int)%4)==2;q.loc[missing,['wind_n','wind_e']]=np.nan
 elif scenario=='missing':q[['wind_n','wind_e']]=np.nan
 q['air_est']=np.hypot(q.vn-q.wind_n,q.ve-q.wind_e)
 return q

def cumulative(z,p):
 t=z.time_s.to_numpy();dt=np.diff(t);h=np.r_[0,np.where((dt>0)&(dt<=2),dt/7200,0.)];return np.cumsum(h*(p+np.r_[p[0],p[:-1]]))

def design_coverage(scores,ncal=10):
 s=np.sort(np.asarray(scores));N=len(s);den=math.comb(N,ncal);expected=0.
 for value in np.unique(s):
  left=int(np.searchsorted(s,value,'left'));right=int(np.searchsorted(s,value,'right'))
  mass=(math.comb(right,ncal) if right>=ncal else 0)-(math.comb(left,ncal) if left>=ncal else 0)
  expected+=mass/den*(right-ncal)/(N-ncal)
 # Independent direct enumeration of calibration-source subsets.
 right=np.searchsorted(s,s,'right');total=0.
 for chosen in itertools.combinations(range(N),ncal):total+=(right[chosen[-1]]-ncal)/(N-ncal)
 enumerated=total/den;assert abs(expected-enumerated)<1e-11
 return expected,den

def main(output):
 check();output.mkdir(parents=True,exist_ok=True);models=ROOT/'models/random_calibration';models.mkdir(parents=True,exist_ok=True);d=load();parents=np.array(sorted(d.parent.unique()));assert len(parents)==38
 perrecord=[];persource=[];splits=[];fits=[];started=time.perf_counter();calibration=[]
 with threadpool_limits(limits=2):
  for seed in [42,137,2026,815,991]:
   order=np.random.default_rng(seed).permutation(parents);train=order[:18].tolist();cal=order[18:28].tolist();test=order[28:].tolist();assert len(set(train)|set(cal)|set(test))==38
   tr=d[d.parent.isin(train)].reset_index(drop=True);candidate=d[~d.parent.isin(train)].reset_index(drop=True);weight=tr.flight_id.map(1/tr.groupby('flight_id').size()).to_numpy(copy=True);weight*=len(weight)/weight.sum()
   fitted={}
   for branch in ['full','windfree']:
    x=features(tr) if branch=='full' else descriptors(tr)[0]
    imp=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True);a=imp.fit_transform(x)
    reg=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=2,learning_rate=.1,random_state=seed,early_stopping=False)
    before=time.perf_counter();reg.fit(a,np.log1p(tr.power_w),sample_weight=weight);elapsed=time.perf_counter()-before
    fitted[branch]=(imp,reg);path=models/f'{seed}_{branch}.joblib';bundle=dict(imputer=imp,regressor=reg,branch=branch,train_sources=train,calibration_sources=cal,test_sources=test,seed=seed)
    joblib.dump(bundle,path,compress=3);fits.append(dict(seed=seed,branch=branch,seconds=elapsed,train_rows=len(tr),train_sources=18,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
   rows=[]
   for scenario in ['original','burst','missing']:
    q=perturb(candidate,scenario);held,age=hold_wind(q);gate=descriptors(held)[2].astype(bool)
    imp,reg=fitted['full'];full=np.expm1(np.clip(reg.predict(imp.transform(features(q))),0,9));heldpred=np.expm1(np.clip(reg.predict(imp.transform(features(held))),0,9))
    imp,reg=fitted['windfree'];fallback=np.expm1(np.clip(reg.predict(imp.transform(descriptors(q)[0])),0,9));robust=np.where(gate,heldpred,fallback)
    for model,pred in [('unconditional',full),('hold_fallback',robust)]:
     for fid,indices in q.groupby('flight_id').indices.items():
      z=q.iloc[indices];t=z.time_s.to_numpy();truth=cumulative(z,z.power_w.to_numpy());estimate=cumulative(z,pred[indices]);err=estimate-truth;score=float(np.max(abs(err)/(estimate+1)))
      rows.append(pd.DataFrame(dict(seed=seed,model=model,scenario=scenario,flight_id=fid,source_id=z.parent.iloc[0],role='calibration' if z.parent.iloc[0] in cal else 'test',elapsed_s=t-t[0],actual_prefix_Wh=truth,predicted_prefix_Wh=estimate)))
      perrecord.append(dict(seed=seed,model=model,scenario=scenario,flight_id=fid,source_id=z.parent.iloc[0],role='calibration' if z.parent.iloc[0] in cal else 'test',score=score,RMSLE=float(np.sqrt(np.mean((np.log1p(pred[indices])-np.log1p(z.power_w.to_numpy()))**2))),peak_prefix_Wh=float(abs(err).max()),final_abs_Wh=float(abs(err[-1]))))
   frame=pd.concat(rows,ignore_index=True);frame.to_parquet(output/f'curves_{seed}.parquet',index=False)
   record=pd.DataFrame(perrecord);record=record[record.seed.eq(seed)]
   for model in ['unconditional','hold_fallback']:
    scores=record[record.model.eq(model)].groupby('source_id').score.max();qcal=float(np.sort(scores.loc[cal])[-1]);design,ncomb=design_coverage(scores.to_numpy())
    assert design>=10/11-1e-12
    calibration.append(dict(seed=seed,model=model,q=qcal,n_cal=10,rank=10,candidate_sources=20,exact_random_design_coverage=design,calibration_subsets_enumerated=ncomb))
    for source in sorted(scores.index):persource.append(dict(seed=seed,model=model,source_id=source,role='calibration' if source in cal else 'test',score=float(scores[source]),q=qcal,whole_family_covered=bool(scores[source]<=qcal+1e-12)))
   splits.append(dict(seed=seed,fit=train,calibration=cal,test=test));print('CALIBRATION SPLIT',seed,'done',round(time.perf_counter()-started,1),'s',flush=True)
 pd.DataFrame(perrecord).to_csv(output/'record_metrics.csv',index=False);pd.DataFrame(persource).to_csv(output/'source_coverage.csv',index=False);pd.DataFrame(calibration).to_csv(output/'calibration.csv',index=False);write(output/'splits.json',splits);write(output/'fit_log.json',fits)
 summaries=[];cov=pd.DataFrame(persource);calframe=pd.DataFrame(calibration);metrics=pd.DataFrame(perrecord)
 for model in ['unconditional','hold_fallback']:
  for scenario in ['original','burst','missing']:
   selected=metrics[metrics.model.eq(model)&metrics.scenario.eq(scenario)&metrics.role.eq('test')];by_source=selected.groupby(['seed','source_id'])[['RMSLE','peak_prefix_Wh','final_abs_Wh']].mean()
   sourcecov=cov[cov.model.eq(model)&cov.role.eq('test')]
   summaries.append(dict(model=model,scenario=scenario,source_split_test_cases=len(sourcecov),observed_whole_family_coverage=float(sourcecov.whole_family_covered.mean()),**{c:float(by_source[c].mean()) for c in by_source.columns},design_expected_coverage=float(calframe[calframe.model.eq(model)].exact_random_design_coverage.mean())))
 pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
 write(output/'calibration_gate.json',dict(passed=True,new_fits=len(fits),random_source_splits=5,total_unique_sources=38,test_source_split_incidences=50,candidate_source_model_score_rows=len(persource),exact_calibration_assignments_enumerated=10*math.comb(20,10),finite_pool_rank_lower_bound=10/11,new_independent_flights=0,new_neural_fits=0,seconds=time.perf_counter()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/new_calibration');a=p.parse_args();main(ROOT/a.output)
