"""Post-development selector checks on all original primary source histories."""
from common4 import *
from shift_experiment import histories
from robust_transfer import proposal,run_method,evaluated
from fast_center import norm
from integrated_core import boxes
import numpy as np,pandas as pd,time

def main():
    out=ROOT/'results/robust_development';out.mkdir(parents=True,exist_ok=True);rows=[];count=0;start=time.perf_counter()
    for info,t,truth,state,anchors,y,ref in histories():
        if info['epsilon_Wh']!=0 or info['halfwidth_s']!=.5:continue
        grid=np.unique(np.r_[np.linspace(0,t[-1],257),anchors]);state=np.interp(grid,t,state);a,b=boxes(anchors,.5);jobs=[('measurement','PCHIP',state)]+[(condition,method,proposal(grid,state,anchors,condition)) for condition in ['original','smooth','step','flat'] for method in ['global','local','validated']]
        for condition,method,p in jobs:
            got,diag=run_method(grid,p,anchors,y,a,b,info['cap_W'],0.,method)
            if ref:
                result=evaluated(got,diag,ref,dict(**info,prior_condition=condition,method=method),True)
            else:
                joined=np.unique(np.r_[got['grid'],t]);E=np.interp(joined,got['grid'],got['increment']);actual=np.interp(joined,t,truth);error=norm(joined,E-actual);peak=float(np.max(abs(E-actual)));assert peak<=got['radius_Wh']+1e-8
                result=dict(**info,prior_condition=condition,method=method,status=got['status'],RMSE_Wh=error,peak_error_Wh=peak,radius_Wh=got['radius_Wh'],weight=diag.get('weight_on_historical_shape',np.nan))
            rows.append(result)
        count+=1
        if count%200==0:print('ROBUST DEVELOPMENT',count,len(rows),round(time.perf_counter()-start,1),'s',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'cases.csv',index=False);summary=[]
    for key,z in f.groupby(['dataset','budget','prior_condition','method']):
        mean=lambda c:macro(z,c) if key[0]=='development' else float(z[c].mean());summary.append(dict(zip(['dataset','budget','prior_condition','method'],key),records=len(z),RMSE_Wh=mean('RMSE_Wh'),radius_Wh=mean('radius_Wh')))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);assert count==1803 and len(f)==23439;write(out/'gate.json',dict(passed=True,histories=count,cases=len(f),source_records=440,solar_windows=161,all_previously_inspected=True,seconds=time.perf_counter()-start));print('ROBUST DEVELOPMENT COMPLETE',count,len(f))

if __name__=='__main__':main()
