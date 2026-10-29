"""Independent scalar optimization and calibration-model simulations."""
from common4 import *
from query_validated_prior import hinge_weight,select_prior
from calibration_enclosure import enclosure
from integrated_core import boxes
from fast_center import recover_clipped
from scipy.optimize import minimize_scalar
import numpy as np,time

def main(out):
    rng=np.random.default_rng(2026102713);maxscore=0.;contain=0.;risk=0.;start=time.perf_counter();foldcount=0
    for trial in range(400):
        A=rng.normal(0,2,int(rng.integers(2,20)));B=rng.normal(0,3,len(A));w,value=hinge_weight(A,B);objective=lambda x:float(np.sum(np.maximum(0,A+B*x)**2));other=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-14});opt=min(objective(0),objective(1),other.fun);difference=abs(value-opt);maxscore=max(maxscore,difference);assert difference<1e-8*max(1.,opt)
        t=np.r_[0.,np.cumsum(rng.uniform(.2,3,20))];m=float(rng.uniform(.05,1.));cap=m*3600;P=rng.uniform(0,cap,20);dt=np.diff(t);truth=np.r_[0.,np.cumsum(P*dt/3600)];gamma=float(rng.choice([0,.02,.1,.25]));bias=float(rng.choice([0,5,20]));g=rng.uniform(-gamma,gamma,20);b=rng.uniform(-bias,bias,20);measured=np.r_[0.,np.cumsum(((1+g)*P+b)*dt/3600)];idx=np.unique(np.rint(np.linspace(0,20,8)).astype(int));actual=t[idx];anchors=actual+.05*np.sin(2*np.pi*actual/t[-1]);anchors[0]=t[0];anchors[-1]=t[-1];eps=float(rng.choice([0,.01,.1]));y=measured[idx]+rng.uniform(-eps,eps,len(idx))+71;a,bb=boxes(anchors,.1);en=enclosure(a,bb,y,cap,eps,gamma,bias);assert en['status']=='CONDITIONAL_CALIBRATION_RELAXATION';error=max(0.,float(np.max(en['lower']-truth[idx])),float(np.max(truth[idx]-en['upper'])));contain=max(contain,error);assert error<1e-10
        larger=enclosure(a,bb,y,cap,eps,gamma+.1 if gamma<.8 else gamma,bias+5);assert np.all(larger['lower']<=en['lower']+1e-10) and np.all(larger['upper']>=en['upper']-1e-10)
        grid,p,diag=select_prior(t,t/t[-1],anchors,en['centers'],a,bb,cap,en['epsilon']);got=recover_clipped(grid,p,a,bb,en['centers'],cap,en['epsilon']);actual=np.interp(got['grid'],t,truth);excess=float(np.max(abs(got['increment']-actual)))-got['radius_Wh'];risk=max(risk,excess);assert excess<1e-8
        for fold in diag['folds']:assert fold['held_index'] not in fold['fit_indices'];foldcount+=1
    flat,loss=hinge_weight([-1,-1],[0,0]);assert flat==0 and loss==0
    # A huge negative counter reset cannot be explained by these declared bounds.
    bad=enclosure([0,1,2],[0,1,2],[10,9,-100],3600,.01,.05,5);assert bad['status']=='CALIBRATION_CONTRACT_INCOMPATIBLE'
    out.mkdir(parents=True,exist_ok=True);report=dict(passed=True,scalar_optimizer_comparisons=400,calibration_mechanisms=400,held_query_exclusion_checks=foldcount,max_scalar_objective_difference=maxscore,max_truth_box_violation_Wh=contain,max_true_risk_excess_Wh=max(0.,risk),flat_score_prefers_PCHIP=True,large_reset_refused=True,seconds=time.perf_counter()-start);write(out/'validation_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main(ROOT/'results/robust_validation')
