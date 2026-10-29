from common import *
from measurement_bridge import *
import pandas as pd

def main():
 check();assert sha(Path(__file__))==json.loads((ROOT/'audit/bridge_amendment_1.json').read_text())['corrected_script_sha256'];assert json.loads((ROOT/'results/bridge_validation.json').read_text())['passed'];log('REGISTERED IDF BRIDGE REPLAY START:231 streams,3 clocks,2 cap scenarios, nearest equal-time8anchors. Original H1 unchanged.')
 r=ROOT/'evidence';f=pd.read_parquet(r/'v3/data/native_power.parquet');inv=pd.read_csv(r/'v3/data/dev_inventory.csv').set_index('flight_id');caps=pd.read_csv(r/'v3/results/g4/fit_caps.csv').set_index('source_id');intervals=[];records=[]
 for fid,g in f.groupby('flight_id',sort=True):
  z=g[g.valid_electrical.fillna(False)&np.isfinite(g.time_s)&np.isfinite(g.power_W)].sort_values('time_s',kind='stable');t=z.time_s.to_numpy(dtype=float,copy=True);t-=t[0];p=z.power_W.to_numpy(float);idx=np.array([int(np.argmin(abs(t-target))) for target in np.linspace(0,t[-1],8)]);idx=np.unique(idx);src=inv.loc[fid,'source_id'];platform=inv.loc[fid,'platform']
  for h in [1.,.1,.01]:
   bins=np.floor(t/h).astype(np.int64)
   for name,cap in [('fit_only',float(caps.loc[src,'cap_W'])),('fixed2000',2000.)]:
    part=[]
    for j,(a,b) in enumerate(zip(idx[:-1],idx[1:])):
     c=margin_coefficients(p[a:b+1],cap);lo,hi=linear_range(c,bins[a:b+1],h);native=float(c@t[a:b+1]);tol=1e-8;assert lo-tol<=native<=hi+tol
     item={'flight_id':fid,'source_id':src,'platform':platform,'h_s':h,'cap_scenario':name,'cap_W':cap,'interval':j,'native_margin_Wh':native,'minimum_margin_Wh':lo,'maximum_margin_Wh':hi,'native_rejects':native>tol,'clock_forces_rejection':lo>tol,'clock_rules_out_interval_rejection':hi<=tol,'native_rows':b-a+1,'role':'post hoc registered measurement sensitivity, coupled interval test; not full-history LP or primaryH1'};part.append(item);intervals.append(item)
    native=any(a['native_rejects'] for a in part);forced=any(a['clock_forces_rejection'] for a in part);within=all(a['clock_rules_out_interval_rejection'] for a in part);assert not forced or native
    records.append({'flight_id':fid,'source_id':src,'platform':platform,'h_s':h,'cap_scenario':name,'cap_W':cap,'anchors':len(idx),'native_rejects':native,'clock_forces_rejection':forced,'clock_rules_out_interval_rejection':within,'clock_unresolved':not forced and not within,'lost_reference_rejection':native and not forced,'role':'post hoc DEV sufficient interval forcing, originalH1 unchanged'})
 f=pd.DataFrame(records);f.to_csv(ROOT/'results/bridge_per_record.csv',index=False);pd.DataFrame(intervals).to_csv(ROOT/'results/bridge_per_interval.csv',index=False);rows=[]
 for (p,h,cap),a in f.groupby(['platform','h_s','cap_scenario']):
  for metric in ['native_rejects','clock_forces_rejection','clock_unresolved','lost_reference_rejection']:
   values=a.groupby('source_id')[metric].mean().to_numpy();rng=np.random.default_rng(2026102609);boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1);rows.append({'platform':p,'h_s':h,'cap_scenario':cap,'metric':metric,'mean':float(values.mean()),'CI_low':float(np.quantile(boot,.025)),'CI_high':float(np.quantile(boot,.975)),'n_sources':len(values),'records':len(a),'record_count':int(a[metric].sum()),'role':'post hoc source-bootstrap descriptive marginal CI; notnewprimarycomparison'})
 pd.DataFrame(rows).to_csv(ROOT/'results/bridge_source_summary.csv',index=False);summary={'passed':True,'records':int(f.flight_id.nunique()),'source_groups':int(f.source_id.nunique()),'clock_cap_record_cases':len(f),'interval_cases':len(intervals),'all_native_margins_enclosed':True,'robust_forcing_without_native_rejection':0,'original_primary_estimand_changed':False,'new_training_runs':0,'new_blind_cases':0};write(ROOT/'results/bridge_summary.json',summary);log('REGISTERED IDF BRIDGE COMPLETE '+json.dumps(summary));print(summary)
if __name__=='__main__':main()
