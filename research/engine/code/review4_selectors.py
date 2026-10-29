"""Maximum integrated proxy excess, with no unqueried current-energy input."""
import numpy as np


def gap_fill(t,chosen,budget):
    chosen=set(map(int,chosen));target=min(len(t),budget)
    while len(chosen)<target:
        ordered=sorted(chosen);candidates=[]
        for a,b in zip(ordered[:-1],ordered[1:]):
            if b-a<2:continue
            interior=np.arange(a+1,b);middle=(t[a]+t[b])/2
            node=int(interior[np.argmin(np.abs(t[interior]-middle))])
            candidates.append((-(t[b]-t[a]),float(t[a]),a,node))
        if not candidates:break
        chosen.add(min(candidates)[-1])
    return np.array(sorted(chosen),int)


def excess_pair(t,proxy,budget,cap,start_counter,end_counter,epsilon,normalize=True):
    """Returns indices and diagnostics; only the two current endpoints are inputs."""
    t=np.asarray(t,float);proxy=np.asarray(proxy,float)
    assert len(proxy)==len(t)-1 and np.all(np.diff(t)>0) and cap>0 and epsilon>=0
    F=np.r_[0.,np.cumsum(np.maximum(proxy,0)*np.diff(t)/3600)]
    if normalize:F*=max(float(end_counter-start_counter),0.)/max(float(F[-1]),1e-15)
    q=F-cap*(t-t[0])/3600;best=0.;pair=(0,0);smallest=0
    for j in range(1,len(t)):
        value=float(q[j]-q[smallest])
        if value>best:best=value;pair=(int(smallest),j)
        if q[j]<q[smallest]:smallest=j
    fallback=best<=2*epsilon+1e-8 or min(budget,len(t))<4
    if fallback:idx=np.unique(np.rint(np.linspace(0,len(t)-1,min(budget,len(t)))).astype(int))
    else:idx=gap_fill(t,{0,len(t)-1,*pair},budget)
    return idx,dict(predicted_repair_Wh=best/2,witness_left=pair[0],witness_right=pair[1],
                    uniform_fallback=bool(fallback),endpoint_normalized=bool(normalize))
