"""Coupled bounded-calibration class, feasible minimax brackets and witnesses.

Exact arithmetic statements concern fixed observation times. Supplied uncertain
times use a declared nominal-time reading-error inflation, an outer relaxation.
Float64 residuals are diagnostics, not formal outward-rounded certification.
"""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csc_matrix,bmat,vstack,hstack,eye
import osqp
from minimax_centers import mass_matrix
from review6_envelopes_v2 import envelopes,evaluate

TOL=2e-8

class CoupledClass:
    def __init__(self,anchors,y,cap_W,epsilon,gain,bias_W,clock_halfwidth=0.):
        self.a=np.asarray(anchors,float);raw=np.asarray(y,float);self.y=raw-raw[0];self.cap=float(cap_W);self.gain=float(gain);self.bias=float(bias_W)
        eps=np.broadcast_to(np.asarray(epsilon,float),raw.shape).copy();h=np.broadcast_to(np.asarray(clock_halfwidth,float),raw.shape).copy()
        if len(self.a)<2 or self.a.shape!=raw.shape or not np.isfinite(np.r_[self.a,raw,eps,h,self.cap,self.gain,self.bias]).all():raise ValueError('Finite matching observations and bounds required.')
        if self.a[0]!=0 or np.any(np.diff(self.a)<=0) or min(self.cap,self.bias)<0 or not 0<=self.gain<1 or np.any(eps<0) or np.any(h<0):raise ValueError('Ordered times from zero and nonnegative finite bounds required.')
        if h[0]!=0 or h[-1]!=0:raise ValueError('Current-horizon endpoint times must be exact.')
        self.eps=eps+((1+self.gain)*self.cap+self.bias)*h/3600
        self.clock_relaxation=bool(np.any(h));self.scale=max(1.,float(np.max(abs(self.y))));self.calls=0

    def polytope(self,grid):
        t=np.asarray(grid,float);n=len(t);assert t[0]==0 and t[-1]==self.a[-1] and np.all(np.diff(t)>0)
        ix=np.searchsorted(t,self.a);assert np.array_equal(t[ix],self.a)
        rr=[];cc=[];vv=[];rhs=[]
        def row(indices,values,value):
            k=len(rhs);rr.extend([k]*len(indices));cc.extend(indices);vv.extend(values);rhs.append(value/self.scale)
        for i,dt in enumerate(np.diff(t)):
            row([i,i+1],[1,-1],0.)
            row([i,i+1],[-1,1],self.cap*dt/3600)
            row([i,i+1,n+i,n+i+1],[1+self.gain,-1-self.gain,-1,1],self.bias*dt/3600)
            row([i,i+1,n+i,n+i+1],[-1+self.gain,1-self.gain,1,-1],self.bias*dt/3600)
        bounds=[(0.,self.cap*x/3600/self.scale) for x in t]+[(None,None)]*n;bounds[0]=(0.,0.)
        for j,k in enumerate(ix):bounds[n+k]=((self.y[j]-self.eps[j])/self.scale,(self.y[j]+self.eps[j])/self.scale)
        return coo_matrix((vv,(rr,cc)),shape=(len(rhs),2*n)).tocsc(),np.asarray(rhs),bounds

    def lp(self,c,A,b,bounds):
        self.calls+=1
        return linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})

    def extrema(self,grid):
        lower=[];upper=[];duals=[]
        for t in grid:
            nodes=np.unique(np.r_[self.a,t]);k=int(np.searchsorted(nodes,t));A,b,bounds=self.polytope(nodes);c=np.zeros(A.shape[1]);c[k]=1.
            lo=self.lp(c,A,b,bounds);hi=self.lp(-c,A,b,bounds)
            if not lo.success or not hi.success:return None
            lower.append(lo.fun*self.scale);upper.append(-hi.fun*self.scale);duals.extend([float(lo.fun),float(hi.fun)])
        return np.asarray(lower),np.asarray(upper)

    def center_lp(self,t,lower,upper):
        n=len(t);A,b,bounds=self.polytope(t);variables=A.shape[1];A=hstack([A,csc_matrix((len(b),1))],format='csc');rr=np.r_[np.arange(n),np.arange(n),np.arange(n,2*n),np.arange(n,2*n)];cc=np.r_[np.arange(n),np.full(n,variables),np.arange(n),np.full(n,variables)];vv=np.r_[-np.ones(n),-np.ones(n),np.ones(n),-np.ones(n)];risk=coo_matrix((vv,(rr,cc)),shape=(2*n,variables+1)).tocsc();A=vstack([A,risk],format='csc');b=np.r_[b,-upper/self.scale,lower/self.scale];c=np.zeros(variables+1);c[-1]=1.;result=self.lp(c,A,b,bounds+[(0,None)])
        if not result.success:return None
        return result.x,result.fun*self.scale

    def witness(self,t,E,Q):
        dE=np.diff(E);dQ=np.diff(Q);dt=np.diff(t);den=self.gain*dE+self.bias*dt/3600;alpha=np.divide(dQ-dE,den,out=np.zeros_like(dE),where=abs(den)>1e-15);g=self.gain*alpha;bias=self.bias*alpha;ix=np.searchsorted(t,self.a)
        residual=max(0.,abs(E[0]),float(np.max(-dE)),float(np.max(dE-self.cap*dt/3600)),float(np.max(abs(dQ-dE)-den)),float(np.max(abs(Q[ix]-self.y)-self.eps)))
        return dict(primal_residual_Wh=residual,max_gain_bound_excess=float(max(0.,np.max(abs(g))-self.gain)),max_bias_bound_excess_W=float(max(0.,np.max(abs(bias))-self.bias)),reading_noise_Wh=(self.y-Q[ix]).tolist(),gain=g.tolist(),bias_W=bias.tolist(),counter_origin_relative_to_first_reading_Wh=float(Q[0]))

    def membership(self,t,E):
        t=np.asarray(t,float);E=np.asarray(E,float);grid=np.unique(np.r_[t,self.a]);e=np.interp(grid,t,E);A,b,bounds=self.polytope(grid);n=len(grid)
        for i,value in enumerate(e):bounds[i]=(value/self.scale,value/self.scale)
        got=self.lp(np.zeros(2*n),A,b,bounds)
        return bool(got.success)

    def recover(self,prior_t,prior,grid_nodes=33,previous_knots=()):
        started=time.perf_counter();coarse=np.unique(np.r_[np.linspace(0,self.a[-1],grid_nodes),self.a]);ext=self.extrema(coarse)
        if ext is None:return dict(status='COUPLED_CALIBRATION_INCOMPATIBLE',LP_calls=self.calls)
        lower,upper=ext;assert np.max(lower-upper)<TOL
        discrete=self.center_lp(coarse,lower,upper)
        if discrete is None:return dict(status='UNRESOLVED_COUPLED_CENTER_LP',LP_calls=self.calls)
        # Independently valid Lipschitz cones enclose every continuous curve.
        env=envelopes(coarse,coarse,(lower+upper)/2,self.cap,np.maximum(0,(upper-lower)/2))
        assert env['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE'
        t=np.unique(np.r_[coarse,env['knots_elapsed_s'],previous_knots,prior_t]);L,U=evaluate(env,t);upper_lp=self.center_lp(t,L,U)
        if upper_lp is None:return dict(status='UNRESOLVED_COUPLED_CENTER_LP',LP_calls=self.calls)
        seed,radius=upper_lp;n=len(t);A,b,bounds=self.polytope(t);lp_E=seed[:n]*self.scale
        # Closed-form interval intersections already include all cone crossings.
        prior_t=np.asarray(prior_t,float);prior=np.asarray(prior,float);raw=np.interp(t,prior_t,prior);raw=raw-raw[0];p=np.r_[0.,np.cumsum(np.clip(np.diff(raw),0,self.cap*np.diff(t)/3600))]
        W=mass_matrix(t);P=bmat([[2*W,None],[None,csc_matrix((n,n))]],format='csc');q=np.r_[-2*(W@(p/self.scale)),np.zeros(n)];bandlo=(U-radius)/self.scale;bandhi=(L+radius)/self.scale;roundoff=bandlo>bandhi
        if np.any(roundoff):assert np.max(bandlo[roundoff]-bandhi[roundoff])*self.scale<1e-8;mid=(bandlo[roundoff]+bandhi[roundoff])/2;bandlo[roundoff]=mid;bandhi[roundoff]=mid
        B=vstack([A,eye(2*n,format='csc')],format='csc');lo=np.r_[np.full(len(b),-np.inf),[(-np.inf if x[0] is None else x[0]) for x in bounds]];hi=np.r_[b,[(np.inf if x[1] is None else x[1]) for x in bounds]];lo[len(b):len(b)+n]=np.maximum(lo[len(b):len(b)+n],bandlo);hi[len(b):len(b)+n]=np.minimum(hi[len(b):len(b)+n],bandhi)
        solver=osqp.OSQP();solver.setup(P=P,q=q,A=B,l=lo,u=hi,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,polishing=True);solver.warm_start(x=seed[:-1]);sol=solver.solve(raise_error=False);optimized=False;E=lp_E;Q=seed[n:2*n]*self.scale
        if sol.x is not None and sol.info.status_val==1:
            candidate=sol.x[:n]*self.scale;counter=sol.x[n:]*self.scale;w=self.witness(t,candidate,counter);risk=float(np.max(np.maximum(U-candidate,candidate-L)))
            if w['primal_residual_Wh']<TOL and risk<=radius+TOL and sol.info.dual_res<1e-7:E=candidate;Q=counter;optimized=True
        witness=self.witness(t,E,Q);actual_ceiling=float(np.max(np.maximum(U-E,E-L)));assert witness['primal_residual_Wh']<TOL and actual_ceiling<=radius+TOL
        lower_radius=discrete[1];assert lower_radius<=radius+TOL
        return dict(status='CHECKED_COUPLED_CENTER',t=t,E=E,Q=Q,lower=L,upper=U,coarse_t=coarse,coarse_lower=lower,coarse_upper=upper,discrete_radius_lower_Wh=float(lower_radius),continuous_radius_upper_Wh=max(float(radius),actual_ceiling),optimality_gap_Wh=max(0.,float(radius-lower_radius)),analytic_grid_gap_bound_Wh=float(self.cap*np.diff(coarse).max()/7200),joint_witness=witness,shape_QP_optimized=optimized,shape_QP_status=sol.info.status,LP_calls=self.calls,seconds=time.perf_counter()-started,clock_scope='Nominal-time error-inflated outer class' if self.clock_relaxation else 'Exact fixed-time bounded gain/bias class',rounding_scope='Float64 diagnostics; no formal outward rounding')
