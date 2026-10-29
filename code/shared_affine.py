"""Exact inverse affine-calibration polytope with bounded residual power drift."""
import time
import numpy as np
import osqp
from scipy.sparse import coo_matrix,csc_matrix,bmat,vstack,eye
from coupled_calibration import CoupledClass,TOL
from minimax_centers import mass_matrix
from review6_envelopes_v2 import envelopes,evaluate

class SharedAffineClass(CoupledClass):
    def __init__(self,anchors,y,cap_W,epsilon,gain,bias_W,drift_W,clock_halfwidth=0.):
        if not np.isfinite(drift_W) or drift_W<0:raise ValueError('Finite nonnegative residual power-drift bound required.')
        super().__init__(anchors,y,cap_W,epsilon,gain,bias_W+drift_W,clock_halfwidth)
        self.bias=float(bias_W);self.drift=float(drift_W)

    def polytope(self,t):
        t=np.asarray(t,float);n=len(t);ix=np.searchsorted(t,self.a);assert np.array_equal(t[ix],self.a);rr=[];cc=[];vv=[];rhs=[];v,w,z=2*n,2*n+1,2*n+2;S=self.scale
        def add(indices,values,right):
            k=len(rhs);rr.extend([k]*len(indices));cc.extend(indices);vv.extend(values);rhs.append(right)
        for j,dt in enumerate(np.diff(t)):
            add([j,j+1],[1,-1],0);add([j,j+1],[-1,1],self.cap*dt/3600/S)
            add([n+j,n+j+1,v],[-1,1,-self.drift*dt/3600/S],0)
            add([n+j,n+j+1,v],[1,-1,-self.drift*dt/3600/S],0)
        for i,k in enumerate(ix):
            add([k,n+k,v,w,z],[1,-1,-(self.y[i]+self.eps[i])/S,-t[k]/3600,-1],0)
            add([k,n+k,v,w,z],[-1,1,(self.y[i]-self.eps[i])/S,t[k]/3600,1],0)
        add([w,v],[1,-self.bias/S],0);add([w,v],[-1,-self.bias/S],0)
        bounds=[(0,self.cap*x/3600/S) for x in t]+[(None,None)]*n+[(1/(1+self.gain),1/(1-self.gain)),(None,None),(None,None)];bounds[0]=(0,0);bounds[n]=(0,0)
        return coo_matrix((vv,(rr,cc)),shape=(len(rhs),2*n+3)).tocsc(),np.asarray(rhs),bounds

    def decode(self,t,x):
        n=len(t);E=x[:n]*self.scale;R=x[n:2*n]*self.scale;v=float(x[-3]);w=float(x[-2]*self.scale);z=float(x[-1]*self.scale);Q=(E-R-w*t/3600-z)/v
        return E,R,Q,v,w,z

    def check(self,t,x):
        A,b,bounds=self.polytope(t);res=max(0.,float(np.max(A@x-b)))
        for value,(lo,hi) in zip(x,bounds):
            if lo is not None:res=max(res,lo-value)
            if hi is not None:res=max(res,value-hi)
        E,R,Q,v,w,z=self.decode(t,x);return dict(primal_residual_scaled=max(0.,res),gain=float(1/v-1),bias_W=float(-w/v),inverse_gain=v,inverse_bias_W=w,origin_parameter_Wh=z,reading_noise_Wh=(self.y-np.interp(self.a,t,Q)).tolist(),residual_power_drift_W=(-3600*np.diff(R)/(v*np.diff(t))).tolist())

    def membership(self,t,E):
        t=np.asarray(t,float);grid=np.unique(np.r_[t,self.a]);e=np.interp(grid,t,E);A,b,bounds=self.polytope(grid)
        for j,value in enumerate(e):bounds[j]=(value/self.scale,value/self.scale)
        return bool(self.lp(np.zeros(A.shape[1]),A,b,bounds).success)

    def recover(self,prior_t,prior,grid_nodes=33,previous_knots=()):
        start=time.perf_counter();coarse=np.unique(np.r_[np.linspace(0,self.a[-1],grid_nodes),self.a]);ext=self.extrema(coarse)
        if ext is None:return dict(status='SHARED_AFFINE_CONTRACT_INCOMPATIBLE',LP_calls=self.calls)
        lower,upper=ext;discrete=self.center_lp(coarse,lower,upper)
        if discrete is None:return dict(status='UNRESOLVED_SHARED_AFFINE_LP',LP_calls=self.calls)
        env=envelopes(coarse,coarse,(lower+upper)/2,self.cap,np.maximum(0,(upper-lower)/2));assert env['status']=='CONDITIONAL_CLOSED_CLOCK_ENVELOPE';t=np.unique(np.r_[coarse,env['knots_elapsed_s'],prior_t,previous_knots]);L,U=evaluate(env,t);upper_lp=self.center_lp(t,L,U)
        if upper_lp is None:return dict(status='UNRESOLVED_SHARED_AFFINE_LP',LP_calls=self.calls)
        seed,radius=upper_lp;n=len(t);A,b,bounds=self.polytope(t);variables=A.shape[1];x=seed[:-1];raw=np.interp(t,prior_t,prior);raw-=raw[0];p=np.r_[0,np.cumsum(np.clip(np.diff(raw),0,self.cap*np.diff(t)/3600))];W=mass_matrix(t);P=bmat([[2*W,None],[None,csc_matrix((variables-n,variables-n))]],format='csc');q=np.r_[-2*(W@(p/self.scale)),np.zeros(variables-n)];lo=np.r_[np.full(len(b),-np.inf),[(-np.inf if x[0] is None else x[0]) for x in bounds]];hi=np.r_[b,[(np.inf if x[1] is None else x[1]) for x in bounds]];bandlo=(U-radius)/self.scale;bandhi=(L+radius)/self.scale;reverse=bandlo>bandhi
        if np.any(reverse):assert np.max(bandlo[reverse]-bandhi[reverse])*self.scale<1e-8;mid=(bandlo[reverse]+bandhi[reverse])/2;bandlo[reverse]=mid;bandhi[reverse]=mid
        lo[len(b):len(b)+n]=np.maximum(lo[len(b):len(b)+n],bandlo);hi[len(b):len(b)+n]=np.minimum(hi[len(b):len(b)+n],bandhi);B=vstack([A,eye(variables,format='csc')],format='csc');solver=osqp.OSQP();solver.setup(P=P,q=q,A=B,l=lo,u=hi,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,polishing=True);solver.warm_start(x=x);sol=solver.solve(raise_error=False);optimized=False
        if sol.x is not None and sol.info.status_val==1:
            test=self.check(t,sol.x);E=sol.x[:n]*self.scale;worst=float(np.max(np.maximum(U-E,E-L)))
            if test['primal_residual_scaled']<TOL/self.scale and worst<=radius+TOL and sol.info.dual_res<1e-7:x=sol.x;optimized=True
        E,R,Q,v,w,z=self.decode(t,x);check=self.check(t,x);actual=max(float(radius),float(np.max(np.maximum(U-E,E-L))));assert check['primal_residual_scaled']<TOL/self.scale;assert discrete[1]<=actual+TOL
        return dict(status='CHECKED_SHARED_AFFINE_CENTER',t=t,E=E,Q=Q,R=R,lower=L,upper=U,coarse_t=coarse,coarse_lower=lower,coarse_upper=upper,radius_lower_Wh=float(discrete[1]),radius_upper_Wh=actual,optimality_gap_Wh=max(0.,actual-discrete[1]),witness=check,shape_QP_optimized=optimized,shape_QP_status=sol.info.status,LP_calls=self.calls,seconds=time.perf_counter()-start,clock_scope='Nominal-time error-inflated outer class' if self.clock_relaxation else 'Exact fixed-time shared affine calibration and bounded residual drift')
