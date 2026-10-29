"""Independent transformed-reading-error dense LP and realization checks."""
from common4 import *
from shared_affine import SharedAffineClass
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import numpy as np,time

def dense_extrema(t,a,y,eps,cap,gain,bias,drift,index):
    n=len(t);z=np.interp(t,a,y-y[0]);ix=np.searchsorted(t,a);A=[];b=[];v,w=2*n,2*n+1
    def add(row,right):A.append(row);b.append(right)
    for j,dt in enumerate(np.diff(t)):
        row=np.zeros(2*n+2);row[j]=1;row[j+1]=-1;add(row,0);add(-row,cap*dt/3600)
        row=np.zeros(2*n+2);row[j]=-1;row[j+1]=1;row[n+j]=-1;row[n+j+1]=1;row[v]=-(z[j+1]-z[j])-drift*dt/3600;row[w]=-dt/3600;add(row,0)
        row=np.zeros(2*n+2);row[j]=1;row[j+1]=-1;row[n+j]=1;row[n+j+1]=-1;row[v]=(z[j+1]-z[j])-drift*dt/3600;row[w]=dt/3600;add(row,0)
    for i,k in enumerate(ix):
        row=np.zeros(2*n+2);row[n+k]=1;row[v]=-eps[i];add(row,0);row=np.zeros(2*n+2);row[n+k]=-1;row[v]=-eps[i];add(row,0)
    row=np.zeros(2*n+2);row[w]=1;row[v]=-bias;add(row,0);row=np.zeros(2*n+2);row[w]=-1;row[v]=-bias;add(row,0);bounds=[(0,cap*x/3600) for x in t]+[(None,None)]*n+[(1/(1+gain),1/(1-gain)),(None,None)];bounds[0]=(0,0);c=np.zeros(2*n+2);c[index]=1
    lo=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs');hi=linprog(-c,A_ub=A,b_ub=b,bounds=bounds,method='highs');assert lo.success and hi.success;return lo.fun,-hi.fun

def main(out):
    rng=np.random.default_rng(2026102721);maximum=0.;risk=0.;residual=0.;fallbacks=0;start=time.perf_counter()
    with threadpool_limits(limits=2):
        for trial in range(200):
            t=np.r_[0,np.cumsum(rng.uniform(.5,4,8))];dt=np.diff(t);cap=float(rng.uniform(20,400));P=rng.uniform(0,cap,8);truth=np.r_[0,np.cumsum(P*dt/3600)];gain=float(rng.choice([0,.02,.1]));bias=float(rng.choice([0,3,10]));drift=float(rng.choice([0,2,5]));g0=rng.uniform(-gain,gain);b0=rng.uniform(-bias,bias);xi=rng.uniform(-drift,drift,8);Q=np.r_[0,np.cumsum(((1+g0)*P+b0+xi)*dt/3600)];idx=np.array([0,2,4,6,8]);a=t[idx];eps=rng.uniform(0,.03,5);y=Q[idx]+rng.uniform(-eps,eps)+91;model=SharedAffineClass(a,y,cap,eps,gain,bias,drift);l,u=model.extrema(t);k=int(rng.integers(len(t)));lo,hi=dense_extrema(t,a,y,eps,cap,gain,bias,drift,k);maximum=max(maximum,abs(lo-l[k]),abs(hi-u[k]));assert maximum<1e-7;assert model.membership(t,truth);got=model.recover(t,truth,17);assert got['status']=='CHECKED_SHARED_AFFINE_CENTER';probe=np.unique(np.r_[t,got['t']]);peak=float(np.max(abs(np.interp(probe,t,truth)-np.interp(probe,got['t'],got['E']))));risk=max(risk,peak-got['radius_upper_Wh']);assert risk<1e-7;residual=max(residual,got['witness']['primal_residual_scaled']);fallbacks+=not got['shape_QP_optimized']
    report=dict(passed=True,independent_dense_LP_comparisons=200,generated_mechanisms=200,max_extrema_difference_Wh=maximum,max_true_risk_excess_Wh=max(0,risk),max_scaled_primal_residual=residual,shape_QP_fallbacks=fallbacks,seconds=time.perf_counter()-start);write(out/'validation_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main(ROOT/'results/shared_affine_validation')
