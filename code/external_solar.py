"""Frozen no-fit reconstruction on every scheduled AtlantikSolar window."""
from common4 import *
from fast_center import recover_clipped,norm
from minimax_centers import CenterProblem
from integrated_core import boxes
from review6_envelopes_v2 import envelopes
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits
import numpy as np,pandas as pd,time,sqlite3,io

def main():
    reg=json.loads((ROOT/'audit/external_registration.json').read_text())
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    D=ROOT/'data/external_solar';O=ROOT/'results/external_solar';O.mkdir(exist_ok=True,parents=True);assert json.loads((D/'adapter_gate.json').read_text())['passed'];frame=pd.read_parquet(D/'native_power.parquet');times=frame.time_s.to_numpy();power=frame.power_W.to_numpy();gap=np.diff(times);weight=np.where((gap>0)&(gap<=2),gap/7200,0.);energy=np.r_[0.,np.cumsum(weight*(power[:-1]+power[1:]))];rows=[];ledger=[];reference=[];problems=0;start=time.perf_counter();conn=sqlite3.connect(O/'curves.sqlite3');conn.execute('CREATE TABLE IF NOT EXISTS curves(window_id INTEGER PRIMARY KEY,metadata TEXT NOT NULL,arrays BLOB NOT NULL)');conn.execute('DELETE FROM curves')
    with threadpool_limits(limits=2):
        for wid,left in enumerate(np.arange(2449,295626,1800)):
            right=min(left+1800,295626);base=dict(window_id=int(wid),source_id='AtlantikSolar_2015-07-14_17',left_s=float(left),right_s=float(right),duration_s=float(right-left),new_campaigns=1)
            if right-left<60:ledger.append({**base,'status':'SHORT_FINAL_WINDOW'});continue
            lo=max(0,int(np.searchsorted(times,left,side='right'))-1);hi=min(len(times)-1,int(np.searchsorted(times,right,side='left')))
            if times[lo]>left or times[hi]<right or np.any(np.diff(times[lo:hi+1])>2):ledger.append({**base,'status':'REFERENCE_SUPPORT_GAP'});continue
            absolute_t=np.unique(np.r_[left,times[(times>left)&(times<right)],right]);t=absolute_t-left;truth=np.interp(absolute_t,times,energy);truth-=truth[0];maxrate=float(np.max(np.diff(truth)*3600/np.diff(t)));interval_energy=float(truth[-1]);ledger.append({**base,'status':'ELIGIBLE','native_nodes':len(t),'max_reference_interval_W':maxrate,'native_energy_Wh':interval_energy});reference.append(pd.DataFrame(dict(window_id=wid,time_s=t,energy_Wh=truth)))
            # Independent reference-basis sensitivity; resample native cumulative
            # energy, never claim that a downsampled power integral is identical.
            one=np.unique(np.r_[np.arange(0,t[-1],1.),t[-1]]);joined=np.unique(np.r_[t,one]);difference=np.interp(joined,one,np.interp(one,t,truth))-np.interp(joined,t,truth);resample_RMSE=norm(joined,difference);resample_peak=float(np.max(abs(difference)));primary_arrays={}
            for cap in [250.,500.,1000.]:
                reference_in_class=maxrate<=cap+1e-8
                for budget in [4,8,16]:
                    idx=np.unique(np.rint(np.linspace(0,len(t)-1,min(budget,len(t)))).astype(int));nominal=t[idx]
                    for eps in [0.,.1]:
                        observed=truth[idx]+eps*(2*nominal/t[-1]-1)
                        for h in [0.,.5]:
                            a,b=boxes(nominal,h);env=envelopes(a,b,observed,cap,eps)
                            for k in [1,2]:
                                for name in ['Linear','PCHIP']:
                                    info=dict(**base,cap_W=cap,budget=budget,epsilon_Wh=eps,halfwidth_s=h,risk_factor=k,prior=name,reference_in_class=reference_in_class)
                                    if env['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':rows.append(dict(**info,status=env['status']));continue
                                    p=np.interp(t,nominal,observed-observed[0]) if name=='Linear' else PchipInterpolator(nominal,observed-observed[0])(t);got=recover_clipped(t,p,a,b,observed,cap,eps,k);assert got['status']=='CHECKED_CLIPPED_MINIMAX_CENTER';g=got['grid'];actual=np.interp(g,t,truth);error=norm(g,got['increment']-actual);peak=float(np.max(abs(got['increment']-actual)));loE=np.interp(t,g,got['lower']);hiE=np.interp(t,g,got['upper']);contain=max(0.,float(np.max(loE-truth)),float(np.max(truth-hiE)))
                                    if reference_in_class:assert contain<1e-8 and peak<=k*got['radius_Wh']+1e-8
                                    outcome=dict(**info,status=got['status'],continuous_RMSE_Wh=error,midpoint_RMSE_Wh=norm(g,got['midpoint']-actual),raw_RMSE_Wh=norm(g,got['prior_raw']-actual),peak_error_Wh=peak,radius_Wh=got['radius_Wh'],conditional_worst_Wh=got['conditional_worst_Wh'],primal_residual_Wh=got['primal_residual_Wh'],risk_residual_Wh=got['risk_residual_Wh'],reference_containment_residual_Wh=contain,seconds=got['seconds'],resample_1Hz_RMSE_Wh=resample_RMSE,resample_1Hz_peak_Wh=resample_peak)
                                    primary=cap==500 and budget==8 and eps==0 and h==.5 and k==1
                                    if primary:
                                        before=time.perf_counter();problem=CenterProblem(t,a,b,observed,cap,eps);qp,diag=problem.solve(np.interp(problem.t,t,p),1);outcome.update(QP_seconds=time.perf_counter()-before,QP_optimized=diag['optimized'],QP_RMSE_Wh=norm(problem.t,qp-np.interp(problem.t,t,truth)))
                                        # Repeat the designated fast method on a fixed 1 Hz basis.
                                        if name=='PCHIP':
                                            p1=PchipInterpolator(nominal,observed-observed[0])(one);onegot=recover_clipped(one,p1,a,b,observed,cap,eps,1);merged=np.unique(np.r_[t,onegot['grid']]);outcome['oneHz_recovery_native_RMSE_Wh']=norm(merged,np.interp(merged,onegot['grid'],onegot['increment'])-np.interp(merged,t,truth))
                                        primary_arrays.update({name+'_t':g,name+'_E':got['increment'],name+'_truth':actual,name+'_midpoint':got['midpoint'],name+'_lower':got['lower'],name+'_upper':got['upper'],name+'_QP_t':problem.t,name+'_QP':qp});primary_arrays['observed_t']=nominal;primary_arrays['observed_energy_Wh']=observed
                                    rows.append(outcome)
            if primary_arrays:
                buf=io.BytesIO();np.savez_compressed(buf,**primary_arrays);conn.execute('INSERT INTO curves VALUES (?,?,?)',(int(wid),json.dumps(base),buf.getvalue()))
            problems+=1
            if problems%10==0:conn.commit();print('EXTERNAL WINDOWS',problems,round(time.perf_counter()-start,1),'s',flush=True)
    conn.commit();conn.close();f=pd.DataFrame(rows);f.to_csv(O/'external_cases.csv',index=False);pd.DataFrame(ledger).to_csv(O/'window_ledger.csv',index=False);pd.concat(reference,ignore_index=True).to_parquet(O/'native_reference_windows.parquet',index=False);keys=['cap_W','budget','epsilon_Wh','halfwidth_s','risk_factor','prior'];columns=['continuous_RMSE_Wh','midpoint_RMSE_Wh','raw_RMSE_Wh','radius_Wh','seconds','reference_in_class'];summary=[dict(zip(keys,key),scheduled_windows=len(z),successful=int(z.status.eq('CHECKED_CLIPPED_MINIMAX_CENTER').sum()),**{c:float(z[c].mean()) for c in columns}) for key,z in f.groupby(keys)];pd.DataFrame(summary).to_csv(O/'external_summary.csv',index=False)
    gate=dict(passed=True,new_independent_campaigns=1,new_flight_collection_by_author=0,scheduled_windows=len(ledger),eligible_windows=problems,screening_status_counts=pd.DataFrame(ledger).status.value_counts().to_dict(),reconstruction_cases=len(f),status_counts=f.status.value_counts().to_dict(),max_primal_residual_Wh=float(f.primal_residual_Wh.max()),max_risk_residual_Wh=float(f.risk_residual_Wh.max()),reference_cap_misspecified_cases=int((~f.reference_in_class).sum()),model_fits=0,primary_frozen_before_values=True,seconds=time.perf_counter()-start);write(O/'external_gate.json',gate);print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
