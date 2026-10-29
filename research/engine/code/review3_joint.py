"""Sparse common-clock/counter LP for the retained numerical-integral contract."""
from review3_common import ROOT,write,sha,check_registration
from measurement_bridge import energy_coefficients,margin_coefficients,linear_range
from review2_counter import repair_radius,feasible_witness
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

TOL=1e-8


def assembly(power,bins,idx,cap,h,eps,sparse=True):
    n=len(power);B=len(idx);q=(bins-bins[0])*h;rr=[];cc=[];vv=[];rhs=[];row=0
    # Different occupied bins are automatically ordered; equal-bin neighbors need a row.
    for i in np.flatnonzero(np.diff(bins)==0):
        rr.extend([row,row]);cc.extend([int(i),int(i+1)]);vv.extend([1.,-1.]);rhs.append(0.);row+=1
    for j,(a,b) in enumerate(zip(idx[:-1],idx[1:])):
        energy=energy_coefficients(power[a:b+1]);margin=margin_coefficients(power[a:b+1],cap)
        for c,upper in [(margin,True),(-energy,False)]:
            nz=np.flatnonzero(c);rr.extend([row]*len(nz));cc.extend((a+nz).tolist());vv.extend((h*c[nz]).tolist())
            sign=1. if upper else -1.
            rr.extend([row,row]);cc.extend([n+j,n+j+1]);vv.extend([-sign,sign])
            if upper:rr.append(row);cc.append(n+B);vv.append(-1.)
            rhs.append(float(-c@q[a:b+1]));row+=1
    A=coo_matrix((vv,(rr,cc)),shape=(row,n+B+1)).tocsr()
    bounds=[(0.,1.)]*n+[(-eps,eps)]*B+[(None,None)]
    objective=np.zeros(n+B+1);objective[-1]=1.
    return (A if sparse else A.toarray()),np.array(rhs),bounds,objective,q


def independent_dense(power,bins,idx,cap,h,eps):
    """Independent assembly in actual time variables, for small validation only."""
    n=len(power);B=len(idx);q=(bins-bins[0])*h;rows=[];rhs=[]
    for i in range(n-1):
        a=np.zeros(n+B+1);a[i]=1;a[i+1]=-1;rows.append(a);rhs.append(0.)
    for j,(a,b) in enumerate(zip(idx[:-1],idx[1:])):
        # Trapezoid coefficients independently accumulated interval by interval.
        c=np.zeros(n)
        for i in range(a,b):
            w=(power[i]+power[i+1])/7200;c[i]-=w;c[i+1]+=w
        m=c.copy();m[a]+=cap/3600;m[b]-=cap/3600
        row=np.r_[m,np.zeros(B+1)];row[n+j]=-1;row[n+j+1]=1;row[-1]=-1
        rows.append(row);rhs.append(0.)
        row=np.r_[-c,np.zeros(B+1)];row[n+j]=1;row[n+j+1]=-1
        rows.append(row);rhs.append(0.)
    objective=np.r_[np.zeros(n+B),1.]
    result=linprog(objective,A_ub=np.array(rows),b_ub=rhs,
                   bounds=[(v,v+h) for v in q]+[(-eps,eps)]*B+[(None,None)],
                   method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert result.success,result.message
    return float(result.fun)


def solve(power,bins,idx,cap,h,eps):
    A,rhs,bounds,objective,q=assembly(power,bins,idx,cap,h,eps)
    result=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':60.})
    answer=dict(solver_success=bool(result.success),solver_status=int(result.status),
                variables=len(objective),nonzeros=int(A.nnz),optimum_Wh=None,
                primal_residual_Wh=None,primal_dual_gap_Wh=None,status='UNRESOLVED_NUMERIC')
    if not result.success:return answer
    primal=max(0.,float(np.max(A@result.x-rhs)))
    dual=float(rhs@result.ineqlin.marginals)
    for i,(lo,hi) in enumerate(bounds):
        if lo is not None:dual+=lo*result.lower.marginals[i]
        if hi is not None:dual+=hi*result.upper.marginals[i]
    gap=abs(float(result.fun)-dual)
    u=result.x[:len(power)];e=result.x[len(power):len(power)+len(idx)];x=q+h*u
    energies=np.array([float(energy_coefficients(power[a:b+1])@x[a:b+1])+e[j+1]-e[j]
                       for j,(a,b) in enumerate(zip(idx[:-1],idx[1:]))])
    margins=energies-cap*np.diff(x[idx])/3600
    witness_ok=(np.min(np.diff(x))>=-1e-9 and np.min(energies)>=-TOL and
                np.max(np.abs(e))<=eps+TOL and np.min(u)>=-1e-9 and np.max(u)<=1+1e-9)
    answer.update(optimum_Wh=float(result.fun),primal_residual_Wh=primal,primal_dual_gap_Wh=gap)
    if witness_ok and primal<TOL and gap<TOL:
        answer['status']='REJECT' if result.fun>TOL else ('COMPATIBLE' if np.max(margins)<=TOL else 'UNRESOLVED_NUMERIC')
    return answer


def validate(out):
    rng=np.random.default_rng(2026102703);error=0.;solves=0
    for trial in range(300):
        n=int(rng.integers(3,22));t=np.r_[0.,np.cumsum(rng.uniform(.02,1,n-1))]
        p=rng.uniform(0,2400,n);cap=float(rng.uniform(200,2200));h=float(rng.choice([1.,.1,.01]))
        bins=np.floor(t/h).astype(np.int64);idx=np.unique(np.rint(np.linspace(0,n-1,min(8,n))).astype(int))
        eps=float(rng.choice([0.,.001,.01,.1]));a=solve(p,bins,idx,cap,h,eps)
        b=independent_dense(p,bins,idx,cap,h,eps);solves+=2
        assert a['status']!='UNRESOLVED_NUMERIC'
        error=max(error,abs(a['optimum_Wh']-b));assert error<1e-8
        larger=solve(p,bins,idx,cap,h,eps+.01);solves+=1
        assert larger['optimum_Wh']<=a['optimum_Wh']+1e-8
        if eps==0:
            from measurement_bridge import common_clock
            old=common_clock(p,bins,idx,cap,h=h)[0];solves+=2
            assert abs(float(old.fun)-a['optimum_Wh'])<1e-8
    result=dict(passed=True,synthetic_cases=300,LP_solves=solves,
                max_sparse_dense_optimum_difference_Wh=error,sparse_dense_independent_assemblies=True,
                zero_error_reduction_checked=True,larger_error_monotonicity_checked=True)
    write(out/'joint_validation.json',result);return result


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        validation=validate(out);print('JOINT VALIDATION',validation,flush=True)
        frame=pd.read_parquet(ROOT/'evidence/v3/data/native_power.parquet')
        inventory=pd.read_csv(ROOT/'evidence/v3/data/dev_inventory.csv').set_index('flight_id')
        caps=pd.read_csv(ROOT/'evidence/v3/results/g4/fit_caps.csv').set_index('source_id')
        rows=[];timings=[];started=time.perf_counter()
        for k,(fid,g) in enumerate(frame.groupby('flight_id',sort=True)):
            z=g[g.valid_electrical.fillna(False)&np.isfinite(g.time_s)&np.isfinite(g.power_W)].sort_values('time_s',kind='stable')
            t=z.time_s.to_numpy(float,copy=True);t-=t[0];p=z.power_W.to_numpy(float)
            idx=np.unique([int(np.argmin(abs(t-target))) for target in np.linspace(0,t[-1],8)])
            source=inventory.loc[fid,'source_id'];platform=inventory.loc[fid,'platform']
            y=np.r_[0.,np.cumsum((p[:-1]+p[1:])*np.diff(t)/7200)]
            for h in [1.,.1,.01]:
                bins=np.floor(t/h).astype(np.int64)
                for name,cap in [('fit_only',float(caps.loc[source,'cap_W'])),('fixed2000',2000.)]:
                    radius=repair_radius(t[idx],y[idx],cap)[0]
                    lower=max(linear_range(margin_coefficients(p[a:b+1],cap),bins[a:b+1],h)[0] for a,b in zip(idx[:-1],idx[1:]))
                    for eps in [0.,.001,.01,.1]:
                        begin=time.perf_counter()
                        if lower>2*eps+TOL:
                            answer=dict(status='REJECT',route='interval_certificate',optimum_Wh=None)
                        elif radius<=eps+TOL/2:
                            corrected=feasible_witness(t[idx],y[idx],cap,eps)
                            assert np.max(np.abs(corrected-y[idx]))<=eps+TOL
                            assert np.min(np.diff(corrected))>=-TOL
                            assert np.max(np.diff(corrected)-cap*np.diff(t[idx])/3600)<=TOL
                            answer=dict(status='COMPATIBLE',route='native_clock_counter_witness',optimum_Wh=None)
                        else:answer={**solve(p,bins,idx,cap,h,eps),'route':'sparse_joint_LP'}
                        item=dict(flight_id=fid,source_id=source,platform=platform,h_s=h,cap_scenario=name,
                                  cap_W=cap,epsilon_Wh=eps,native_rows=len(t),anchors=len(idx),
                                  native_clock_repair_Wh=radius,individual_lower_margin_Wh=lower,**answer)
                        rows.append(item);timings.append(dict(flight_id=fid,h_s=h,cap_scenario=name,epsilon_Wh=eps,
                                                            route=answer['route'],elapsed_s=time.perf_counter()-begin))
            if (k+1)%20==0:print('NATIVE',k+1,'/231',round(time.perf_counter()-started,1),'s',flush=True)
        f=pd.DataFrame(rows);assert len(f)==5544
        # Check zero-epsilon reduction against old sufficient interval decisions.
        old=pd.read_csv(ROOT/'results/bridge_per_record.csv')
        merged=f[f.epsilon_Wh.eq(0)].merge(old,on=['flight_id','source_id','platform','h_s','cap_scenario','cap_W'],suffixes=('','_old'),validate='one_to_one')
        assert (merged[merged.clock_forces_rejection].status=='REJECT').all()
        assert (merged[~merged.native_rejects].status=='COMPATIBLE').all()
        for _,a in f.groupby(['flight_id','h_s','cap_scenario']):
            a=a.sort_values('epsilon_Wh');status=list(a.status)
            for i in range(len(status)):
                if status[i]=='COMPATIBLE':assert 'REJECT' not in status[i+1:]
        f.to_csv(out/'native_joint_cases.csv',index=False);pd.DataFrame(timings).to_csv(out/'native_joint_timing.csv',index=False)
        summary=f.groupby(['platform','h_s','cap_scenario','epsilon_Wh','status']).size().rename('records').reset_index()
        summary.to_csv(out/'native_joint_summary.csv',index=False)
        paths=['native_joint_cases.csv','native_joint_summary.csv','joint_validation.json']
        gate=dict(passed=True,records=231,record_scenarios=len(f),sparse_native_LP_cases=int(f.route.eq('sparse_joint_LP').sum()),
                  routes=f.route.value_counts().to_dict(),statuses=f.status.value_counts().to_dict(),
                  unresolved_numeric_cases=int(f.status.eq('UNRESOLVED_NUMERIC').sum()),
                  new_fits=0,new_blind_cases=0,
                  output_hashes=[dict(path=p,sha256=sha(out/p)) for p in paths])
        write(out/'joint_gate.json',gate);print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review3');args=parser.parse_args();main(ROOT/args.output)
