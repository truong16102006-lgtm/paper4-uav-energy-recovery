"""Distinguish closed-bin feasibility from a positive strict-clock witness."""
from review4_common import ROOT,sha,write,check_registration
from review3_joint_v1 import assembly
from measurement_bridge import energy_coefficients
from review2_counter import feasible_witness
import argparse,json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,hstack,vstack,csr_matrix
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

ENERGY_TOL=1e-8
TIME_TOL=1e-9


def strict_solve(p,bins,idx,cap,h,eps,dense=False):
    n=len(p);B=len(idx);q=(bins-bins[0])*h
    if not dense:
        A,rhs,bounds,_,_=assembly(p,bins,idx,cap,h,eps)
        base=hstack([A[:,:-1],csr_matrix((A.shape[0],1))],format='csr')
        rr=[];cc=[];vv=[];extra=[];row=0
        for i in range(n-1):
            rr.extend([row]*3);cc.extend([i,i+1,n+B]);vv.extend([1.,-1.,1/h]);extra.append((q[i+1]-q[i])/h);row+=1
        for i in range(n):
            rr.extend([row,row]);cc.extend([i,n+B]);vv.extend([1.,1/h]);extra.append(1.);row+=1
        more=coo_matrix((vv,(rr,cc)),shape=(row,n+B+1)).tocsr()
        A=vstack([base,more],format='csr');rhs=np.r_[rhs,extra]
        bounds=bounds[:-1]+[(0,h)]
    else:
        # Independent actual-time formulation; no normalized-offset assembly.
        rows=[];rhs=[]
        for i in range(n-1):
            c=np.zeros(n+B+1);c[i]=1;c[i+1]=-1;c[-1]=1;rows.append(c);rhs.append(0.)
        for i in range(n):
            c=np.zeros(n+B+1);c[i]=1;c[-1]=1;rows.append(c);rhs.append(q[i]+h)
        for j,(a,b) in enumerate(zip(idx[:-1],idx[1:])):
            c=np.zeros(n)
            for i in range(a,b):
                w=(p[i]+p[i+1])/7200;c[i]-=w;c[i+1]+=w
            m=c.copy();m[a]+=cap/3600;m[b]-=cap/3600
            r=np.r_[m,np.zeros(B+1)];r[n+j]=-1;r[n+j+1]=1;rows.append(r);rhs.append(0.)
            r=np.r_[-c,np.zeros(B+1)];r[n+j]=1;r[n+j+1]=-1;rows.append(r);rhs.append(0.)
        A=np.array(rows);rhs=np.array(rhs);bounds=[(v,v+h) for v in q]+[(-eps,eps)]*B+[(0,h)]
    objective=np.zeros(n+B+1);objective[-1]=-1
    r=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
              options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':60})
    result=dict(solver_success=bool(r.success),solver_status=int(r.status),strict_slack_s=None,status='BOUNDARY_OR_NUMERIC')
    if not r.success:return result
    x=r.x[:n] if dense else q+h*r.x[:n];e=r.x[n:n+B];delta=float(r.x[-1])
    energies=np.array([energy_coefficients(p[a:b+1])@x[a:b+1]+e[j+1]-e[j] for j,(a,b) in enumerate(zip(idx[:-1],idx[1:]))])
    margin=energies-cap*np.diff(x[idx])/3600
    min_gap=float(np.min(np.diff(x)));right=float(np.min(q+h-x));residual=max(0.,float(np.max(A@r.x-rhs)))
    result.update(strict_slack_s=delta,minimum_gap_s=min_gap,minimum_right_margin_s=right,
                  maximum_energy_margin_Wh=float(np.max(margin)),max_scaled_constraint_residual=residual)
    if delta>TIME_TOL and min_gap>TIME_TOL and right>TIME_TOL and np.min(energies)>=-ENERGY_TOL and np.max(margin)<=ENERGY_TOL and residual<ENERGY_TOL:
        result['status']='STRICT_WITNESS_LP'
    return result


def validate(out):
    rng=np.random.default_rng(2026102741);difference=0.;solves=0
    for trial in range(200):
        n=int(rng.integers(3,18));t=np.r_[0.,np.cumsum(rng.uniform(.02,1,n-1))];h=float(rng.choice([1.,.1,.01]))
        bins=np.floor(t/h).astype(np.int64);p=rng.uniform(0,2400,n);cap=float(rng.uniform(200,2100));eps=float(rng.choice([0.,.001,.01,.1]))
        idx=np.unique(np.rint(np.linspace(0,n-1,min(8,n))).astype(int))
        a=strict_solve(p,bins,idx,cap,h,eps);b=strict_solve(p,bins,idx,cap,h,eps,True);solves+=2
        assert a['solver_success']==b['solver_success']
        if a['solver_success']:
            difference=max(difference,abs(a['strict_slack_s']-b['strict_slack_s']));assert difference<1e-7
    bins=np.zeros(3,dtype=np.int64);idx=np.arange(3)
    bad=strict_solve(np.full(3,2000.),bins,idx,1000.,1.,0.);good=strict_solve(np.full(3,500.),bins,idx,1000.,1.,0.);solves+=2
    assert bad['status']=='BOUNDARY_OR_NUMERIC' and abs(bad['strict_slack_s'])<1e-9
    assert good['status']=='STRICT_WITNESS_LP' and good['strict_slack_s']>0
    result=dict(passed=True,synthetic_cases=200,LP_solves=solves,max_sparse_dense_slack_difference_s=difference,
                analytic_closure_only_counterexample=bad,analytic_strict_counterpart=good,
                time_decision_tolerance_s=TIME_TOL,energy_decision_tolerance_Wh=ENERGY_TOL)
    write(out/'strict_validation.json',result)


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True)
    amendment=json.loads((ROOT/'audit/review4_strict_witness_amendment.json').read_text(encoding='utf-8'))
    assert sha(ROOT/amendment['corrected_script'])==amendment['corrected_sha256']
    with threadpool_limits(limits=2):
        validate(out);old=pd.read_csv(ROOT/'results/review3/native_joint_cases.csv')
        frame=pd.read_parquet(ROOT/'evidence/v3/data/native_power.parquet');rows=[]
        for k,(fid,g) in enumerate(frame.groupby('flight_id',sort=True)):
            z=g[g.valid_electrical.fillna(False)&np.isfinite(g.time_s)&np.isfinite(g.power_W)].sort_values('time_s',kind='stable')
            t=z.time_s.to_numpy(float,copy=True);t-=t[0];p=z.power_W.to_numpy(float)
            idx=np.unique([int(np.argmin(abs(t-target))) for target in np.linspace(0,t[-1],8)])
            y=np.r_[0.,np.cumsum((p[:-1]+p[1:])*np.diff(t)/7200)]
            for record in old[old.flight_id.eq(fid)].to_dict('records'):
                h=record['h_s'];eps=record['epsilon_Wh'];cap=record['cap_W'];bins=np.floor(t/h).astype(np.int64);q=(bins-bins[0])*h
                if record['status']=='REJECT':answer=dict(status='CLOSED_REJECT',route='retained_closed_rejection')
                elif record['route']=='native_clock_counter_witness' and np.min(np.diff(t))>TIME_TOL and np.min(q+h-t)>TIME_TOL:
                    corrected=feasible_witness(t[idx],y[idx],cap,eps)
                    assert np.max(np.abs(corrected-y[idx]))<=eps+ENERGY_TOL and np.min(np.diff(corrected))>=-ENERGY_TOL
                    assert np.max(np.diff(corrected)-cap*np.diff(t[idx])/3600)<=ENERGY_TOL
                    answer=dict(status='STRICT_WITNESS_NATIVE',route='checked_native_witness',
                                minimum_gap_s=float(np.min(np.diff(t))),minimum_right_margin_s=float(np.min(q+h-t)))
                else:
                    # Safe constructive fast path: perturb only the clock, then verify
                    # the original common energy/counter constraints independently.
                    tau=max(100*TIME_TOL,1e-6*h)
                    x=np.minimum(np.maximum(t,q),q+h-tau)
                    corrected=y[idx] if record['native_clock_repair_Wh']==0 else feasible_witness(t[idx],y[idx],cap,eps)
                    e=corrected-y[idx]
                    increments=(p[:-1]+p[1:])*np.diff(x)/7200
                    cumulative=np.r_[0.,np.cumsum(increments)]
                    energy=np.diff(cumulative[idx])+np.diff(e)
                    maximum_margin=float(np.max(energy-cap*np.diff(x[idx])/3600))
                    valid=(np.min(np.diff(x))>TIME_TOL and np.min(q+h-x)>TIME_TOL and np.min(x-q)>=-TIME_TOL
                           and np.max(np.abs(e))<=eps+ENERGY_TOL and np.min(energy)>=-ENERGY_TOL
                           and maximum_margin<=ENERGY_TOL)
                    if valid:
                        answer=dict(status='STRICT_WITNESS_INTERIOR',route='checked_interior_clock_witness',
                                    minimum_gap_s=float(np.min(np.diff(x))),minimum_right_margin_s=float(np.min(q+h-x)),
                                    maximum_energy_margin_Wh=maximum_margin)
                    else:answer={**strict_solve(p,bins,idx,cap,h,eps),'route':'strict_slack_LP'}
                rows.append({**{key:record[key] for key in ['flight_id','source_id','platform','h_s','cap_scenario','cap_W','epsilon_Wh']},
                             'old_closed_status':record['status'],**answer})
            if (k+1)%30==0:print('STRICT NATIVE',k+1,'/231',flush=True)
        f=pd.DataFrame(rows);assert len(f)==5544
        f.to_csv(out/'strict_native_cases.csv',index=False)
        f.groupby(['platform','h_s','cap_scenario','epsilon_Wh','status']).size().rename('records').reset_index().to_csv(out/'strict_native_summary.csv',index=False)
        files=['strict_validation.json','strict_native_cases.csv','strict_native_summary.csv']
        gate=dict(passed=True,record_scenarios=len(f),statuses=f.status.value_counts().to_dict(),
                  strict_LP_cases=int(f.route.eq('strict_slack_LP').sum()),
                  scientific_meaning='strict witness is checked within declared numeric tolerances; boundary/numeric cases retained unresolved',
                  output_hashes=[dict(path=p,sha256=sha(out/p)) for p in files])
        write(out/'strict_gate.json',gate);print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review4');args=parser.parse_args();main(ROOT/args.output)
