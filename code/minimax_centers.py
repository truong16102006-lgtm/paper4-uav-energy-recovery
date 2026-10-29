"""Continuous-L2 projection into feasible minimax-center classes; sparse QP."""
import numpy as np,osqp,time
from scipy.sparse import diags,bmat,coo_matrix,csc_matrix
from review6_envelopes_v2 import envelopes,prepare,evaluate

def mass_matrix(t):
    dt=np.diff(t)/(t[-1]-t[0]);main=np.r_[dt[0]/3,(dt[:-1]+dt[1:])/3,dt[-1]/3];off=dt/6
    return diags([off,main,off],[-1,0,1],format='csc')

def l2_error(t,delta):
    dt=np.diff(t);return float(np.sqrt(np.sum(dt*(delta[:-1]**2+delta[:-1]*delta[1:]+delta[1:]**2)/3)/(t[-1]-t[0])))

class CenterProblem:
    def __init__(self,t,a,b,y,cap,epsilon):
        self.env=envelopes(a,b,y,cap,epsilon)
        if self.env['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':raise ValueError(self.env['status'])
        aa,bb,l,u,m,t0,y0,*_=prepare(a,b,y,cap,epsilon);self.t0=t0;self.a=aa;self.b=bb;self.l=l;self.u=u;self.m=m
        self.t=np.unique(np.r_[np.asarray(t)-t0,aa,bb,self.env['knots_elapsed_s']]);self.n=n=len(self.t);self.lower,self.upper=evaluate(self.env,self.t);self.mid=(self.lower+self.upper)/2;self.R=self.env['minimax_radius_Wh'];self.origin=sum(self.env['initial_counter_interval_centered_Wh'])/2
        self.scale=max(1.,float(y[-1]-y[0]),self.R);self.W=mass_matrix(self.t);self.P=bmat([[2*self.W,None],[None,csc_matrix((1,1))]],format='csc')
        rr=[];cc=[];vv=[];low=[];high=[];row=0
        def add(cols,vals,lo,hi):
            nonlocal row
            rr.extend([row]*len(cols));cc.extend(cols);vv.extend(vals);low.append(lo);high.append(hi);row+=1
        for i,dt in enumerate(np.diff(self.t)):add([i,i+1],[-1.,1.],0.,m*dt/self.scale)
        for ai,bi,li,ui in zip(aa,bb,l,u):
            add([int(np.searchsorted(self.t,ai)),n],[1.,1.],-np.inf,ui/self.scale)
            add([int(np.searchsorted(self.t,bi)),n],[1.,1.],li/self.scale,np.inf)
        add([0],[1.],0.,0.)
        self.risk_start=row
        for i in range(n):add([i],[1.],0.,0.)
        self.A=coo_matrix((vv,(rr,cc)),shape=(row,n+1)).tocsc();self.low=np.array(low);self.high=np.array(high);self.solver=None

    def risk_bounds(self,k):
        low=self.low.copy();high=self.high.copy();low[self.risk_start:]=(self.upper-k*self.R)/self.scale;high[self.risk_start:]=(self.lower+k*self.R)/self.scale
        # Collapse only an ulp-level reversed center interval; verify original
        # unrounded inequalities in diagnostics rather than enlarge its radius.
        reversed=low>high
        assert not reversed.any() or np.max(low[reversed]-high[reversed])*self.scale<1e-12
        middle=(low[reversed]+high[reversed])/2;low[reversed]=middle;high[reversed]=middle
        return low,high

    def diagnostics(self,E,origin,k):
        z=E+origin;gap=np.diff(E);dt=np.diff(self.t)
        feasible=max(0.,abs(E[0]),float(np.max(-gap)),float(np.max(gap-self.m*dt)),float(np.max(np.interp(self.a,self.t,z)-self.u)),float(np.max(self.l-np.interp(self.b,self.t,z))))
        risk=max(0.,float(np.max(self.upper-k*self.R-E)),float(np.max(E-self.lower-k*self.R)))
        worst=float(np.max(np.maximum(E-self.lower,self.upper-E)))
        return feasible,risk,worst

    def solve(self,prior,k=1):
        assert k>=1;prior=np.asarray(prior,float);assert prior.shape==self.t.shape and np.isfinite(prior).all();q=np.r_[-2*(self.W@(prior/self.scale)),0.];lo,hi=self.risk_bounds(k);begin=time.perf_counter();success=False;statuses=[]
        for attempt in range(2):
            options=dict(verbose=False,eps_abs=1e-9 if attempt==0 else 1e-10,eps_rel=1e-9 if attempt==0 else 1e-10,max_iter=20000 if attempt==0 else 100000,polishing=True,adaptive_rho=True,check_termination=25)
            if self.solver is None or attempt:
                self.solver=osqp.OSQP();self.solver.setup(P=self.P,q=q,A=self.A,l=lo,u=hi,**options)
            else:self.solver.update(q=q,l=lo,u=hi)
            self.solver.warm_start(x=np.r_[self.mid/self.scale,self.origin/self.scale]);result=self.solver.solve(raise_error=False);statuses.append(result.info.status)
            if result.x is None:continue
            E=result.x[:self.n]*self.scale;origin=result.x[-1]*self.scale;f,r,worst=self.diagnostics(E,origin,k)
            if result.info.status_val==1 and f<=1e-8 and r<=1e-8 and result.info.dual_res<=1e-7:
                success=True;break
        if not success:
            E=self.mid.copy();origin=self.origin;f,r,worst=self.diagnostics(E,origin,k);assert max(f,r)<1e-8
        return E,dict(optimized=success,solver_status=';'.join(statuses),seconds=time.perf_counter()-begin,iterations=int(result.info.iter),primal_residual_Wh=f,risk_residual_Wh=r,solver_dual_residual=float(result.info.dual_res),conditional_worst_Wh=worst,radius_Wh=self.R,origin_centered_Wh=origin,variables=self.n+1,nonzeros=int(self.A.nnz+self.P.nnz),prior_distance_Wh=l2_error(self.t,E-prior))
