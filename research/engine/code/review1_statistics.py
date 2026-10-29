from review1_common import *
import numpy as np,pandas as pd,re
from scipy.stats import t as student

SEED=2026102610
def bootstrap(frame,column,B=10000):
 g=frame.groupby(['platform','source_id'])[column].mean().dropna();arrays=[a.to_numpy(float) for _,a in g.groupby(level=0)];rng=np.random.default_rng(SEED);estimate=float(np.mean([a.mean() for a in arrays]));draw=np.mean([a[rng.integers(len(a),size=(B,len(a)))].mean(1) for a in arrays],axis=0);lo,hi=np.quantile(draw,[.025,.975]);return {'mean':estimate,'CI_low':float(lo),'CI_high':float(hi),'sources':len(g),'role':'exploratory marginal source-bootstrap conditional on retained fits; not independent confirmation'}

def holm(p):
 p=np.asarray(p,float);order=np.argsort(p);a=np.empty(len(p));a[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return a

def main():
 check_review();out=ROOT/'results/review1';assert json.loads((out/'experiment_gate.json').read_text())['passed'];log('REVIEW1 STATISTICS START:all scenarios/control summaries; weights/group audit; nominal conditional intervals;posthocTOST12 with olddelta.01.')
 f=pd.read_csv(out/'query_per_record.csv');rows=[]
 for (p,cap,B,policy),a in f.groupby(['platform','cap_scenario','budget','policy'],sort=True):
  src=a.groupby('source_id');V=int(a.reference_violates.sum());non=len(a)-V;mean=lambda col:float(src[col].mean().mean())
  rows.append({'platform':p,'cap_scenario':cap,'budget':B,'policy':policy,'records':len(a),'sources':a.source_id.nunique(),'violating_records':V,'violation_record_fraction':V/len(a),'violation_source_fraction':mean('reference_violates'),'rejections':int(a.rejected.sum()),'source_missed_burden':mean('missed_violation'),'conditional_TPR':float(a[a.reference_violates].rejected.mean()) if V else np.nan,'conditional_FPR':float(a[~a.reference_violates].rejected.mean()) if non else np.nan,'scope':'fixed finite DEV record/source counts and descriptive conditional rates'})
 pd.DataFrame(rows).to_csv(out/'scenario_rates.csv',index=False)
 w=f.pivot(index=['flight_id','source_id','platform','budget','cap_scenario'],columns='policy',values='missed_violation').reset_index();rows=[]
 for (cap,B),a in w.groupby(['cap_scenario','budget'],sort=True):
  for policy in ['audit','proxy_centers','proxy_brackets','oracle']:
   b=a.copy();b['gain']=b.uniform-b[policy];rows.append({'cap_scenario':cap,'budget':B,'contrast':policy+'_vs_uniform',**bootstrap(b,'gain')})
  for policy in ['proxy_centers','proxy_brackets']:
   b=a.copy();b['gain']=b[policy]-b.audit;rows.append({'cap_scenario':cap,'budget':B,'contrast':'audit_vs_'+policy,**bootstrap(b,'gain')})
 pd.DataFrame(rows).to_csv(out/'query_contrast_inference.csv',index=False)
 a=w[w.cap_scenario.eq('q0.9')&w.budget.eq(8)].copy();a['primary_gain']=a.uniform-a.audit;original=json.loads((ROOT/'evidence/v3/results/g4/primary_inference.json').read_text());assert abs(bootstrap(a,'primary_gain')['mean']-original['mean'])<1e-14
 g=a.groupby(['platform','source_id'])[['uniform','audit','oracle','primary_gain']].mean();rng=np.random.default_rng(SEED);arrays=[v.to_numpy() for _,v in g.groupby(level=0)];draw=np.mean([ar[rng.integers(len(ar),size=(10000,len(ar)))].mean(1) for ar in arrays],axis=0);den=draw[:,0]-draw[:,2];good=den>1e-12;ratio=(draw[good,0]-draw[good,1])/den[good];mu=g.groupby(level=0).mean().mean();point=(mu.uniform-mu.audit)/(mu.uniform-mu.oracle) if mu.uniform-mu.oracle>1e-12 else np.nan
 opportunity={'original_primary_gain':float(mu.primary_gain),'uniform_missed_burden':float(mu.uniform),'audit_missed_burden':float(mu.audit),'oracle_missed_burden':float(mu.oracle),'fraction_of_oracle_opportunity':float(point),'CI_low':float(np.quantile(ratio,.025)),'CI_high':float(np.quantile(ratio,.975)),'undefined_bootstrap_denominators':int((~good).sum()),'oracle_reachability':'any finest-reference interval can be isolated with at most4anchors; oracle forbiddenlabelceiling only','role':'postreviewexploratoryfraction,notnewprimary'};write(out/'oracle_opportunity.json',opportunity)
 cost=pd.read_csv(out/'recovery_cost.csv');rows=[]
 for (p,B,policy),z in cost.groupby(['platform','budget','policy']):rows.append({'platform':p,'budget':B,'policy':policy,**bootstrap(z,'relative_volume_increase')})
 for (B,policy),z in cost.groupby(['budget','policy']):rows.append({'platform':'equal_platform','budget':B,'policy':policy,**bootstrap(z,'relative_volume_increase')})
 pd.DataFrame(rows).to_csv(out/'recovery_cost_inference.csv',index=False)
 # Original H1 is not replaced by alternate weights.
 weights=[{'weighting':'original_equal_platform_source','mean':float(mu.primary_gain)},{'weighting':'equal_source','mean':float(g.primary_gain.mean())},{'weighting':'equal_record','mean':float(a.primary_gain.mean())}]
 for p,z in a.groupby('platform'):
  held=a[~a.platform.eq(p)];weights.append({'weighting':'leave_platform_out_'+p,'mean':float(held.groupby(['platform','source_id']).primary_gain.mean().groupby(level=0).mean().mean())})
 for s in a.source_id.unique():
  z=a[~a.source_id.eq(s)];weights.append({'weighting':'leave_source_out_'+s,'mean':float(z.groupby(['platform','source_id']).primary_gain.mean().groupby(level=0).mean().mean())})
 pd.DataFrame(weights).to_csv(out/'weight_and_deletion_sensitivity.csv',index=False)
 # No fabricated daily groups: enumerate what can actually be parsed.
 groups=[]
 for (p,s),z in a.groupby(['platform','source_id']):
  match=re.search(r'(20\d\d)-(\d{1,2})-(\d{1,2})',s);date='-'.join([match.group(1),match.group(2).zfill(2),match.group(3).zfill(2)]) if match else None
  groups.append({'platform':p,'source_id':s,'known_date':date,'date_provenance':'suppliedCMUdate' if p=='cmu_quadrotor' else 'explicitPixhawkfilename' if date else 'unknown,notinferred','source_gain':float(z.primary_gain.mean()),'role':'source exclusion groups are not proven homogeneous independent collection blocks'})
 pd.DataFrame(groups).to_csv(out/'grouping_inventory.csv',index=False)
 bygroup=pd.DataFrame(groups).groupby('platform').agg(source_groups=('source_id','nunique'),documented_dates=('known_date','nunique'));bygroup.to_csv(out/'grouping_summary.csv')
 write(out/'grouping_limit.json',{'primary_grouping':'14Pixhawkcanonicalrecordings+24Speedycanonicalrecordings+18CMUdates; fitting/exclusion group, not physically uniform unit','unknown_Speedy_dates':True,'collapse_all_unknown_Speedy_dates':'onlyoneSpeedycollectionblockwouldremain; platformvarianceunidentifiable','alternative_campaign_CI_released':False,'reason':'missing date/campaign metadata cannot be repaired by relabeling a cluster'})
 arrays=[z.primary_gain.to_numpy() for _,z in g.groupby(level=0)];P=len(arrays);var=[z.var(ddof=1)/len(z) for z in arrays];se=np.sqrt(sum(var))/P;df=sum(var)**2/sum(v*v/(len(z)-1) for z,v in zip(arrays,var) if v>0);half=student.ppf(.975,df)*se;rng=np.random.default_rng(SEED);means=[];variances=[]
 for ar in arrays:
  draws=rng.choice([-1.,1.],size=(9999,len(ar)))*(ar-ar.mean());means.append(draws.mean(1));variances.append(draws.var(1,ddof=1)/len(ar))
 bt=np.mean(means,axis=0)/np.maximum(np.sqrt(np.sum(variances,axis=0))/P,1e-15);qlo,qhi=np.quantile(bt,[.025,.975]);estimate=float(mu.primary_gain)
 intervals=[{'method':'original_percentile','mean':estimate,'low':original['CI_low'],'high':original['CI_high']},{'method':'Welch_Satterthwaite_t','mean':estimate,'low':estimate-half,'high':estimate+half},{'method':'wild_t_Rademacher','mean':estimate,'low':estimate-qhi*se,'high':estimate-qlo*se}];pd.DataFrame(intervals).to_csv(out/'primary_interval_sensitivity.csv',index=False)
 # Equivalence is tested rather than inferred from a small or nonsignificant effect.
 m=pd.read_csv(ROOT/'evidence/clock/results/m300/ensemble_per_segment.csv');p=m.pivot(index=['flight','source_id','adapter','gamma'],columns='model',values='normalized_RMSE').reset_index();eq=[]
 for (adapter,gamma),z in p.groupby(['adapter','gamma']):
  for method in ['residual','direct']:
   v=(z[method]-z.PCHIP).groupby(z.source_id).mean().to_numpy();n=len(v);mean=float(v.mean());sd=float(v.std(ddof=1));sdev=sd/np.sqrt(n);margin=.01;tost=max(float(student.sf((mean+margin)/sdev,n-1)),float(student.cdf((mean-margin)/sdev,n-1)));half=float(student.ppf(.95,n-1)*sdev);eq.append({'adapter':adapter,'gamma':gamma,'method':method,'date_groups':n,'effect':mean,'delta':margin,'TOST_p':tost,'CI90_low':mean-half,'CI90_high':mean+half,'role':'posthocconditionalalread yseenunitclockreplay;notphysicalgroundtruthequivalence'})
 for row,adj in zip(eq,holm([r['TOST_p'] for r in eq])):row['Holm12_TOST_p']=float(adj);row['practical_equivalence_supported']=bool(adj<.05)
 pd.DataFrame(eq).to_csv(out/'M300_TOST.csv',index=False)
 write(out/'statistics_gate.json',{'passed':True,'primary_estimand_unchanged':True,'Q90_primary_reproduced':True,'weight_sensitivity_reported':True,'group_metadata_missing_not_invented':True,'equivalence_contrasts':len(eq),'equivalence_family':'postreviewHolm12,separateoriginalfamilies','new_blind_cases':0,'new_fits':0});log('REVIEW1 STATISTICS COMPLETE:allcaps/control comparisons retained,originalH1unchanged,TOSTnotforced;noindependence/coverageclaimfabricated.');print(opportunity);print(pd.DataFrame(eq)[['adapter','gamma','method','Holm12_TOST_p','practical_equivalence_supported']].to_string(index=False))
if __name__=='__main__':main()
