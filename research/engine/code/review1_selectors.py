"""Fixed proxy controls have no current-energy argument; oracle is isolated."""
import numpy as np
from scipy.signal import find_peaks

def fill(indices,n,budget):
 target=min(n,budget);chosen=set(map(int,indices))
 for j in np.unique(np.rint(np.linspace(0,n-1,target)).astype(int)):
  if len(chosen)>=target:break
  chosen.add(int(j))
 for j in range(n):
  if len(chosen)>=target:break
  chosen.add(j)
 return np.array(sorted(chosen),int)

def fixed_proxy(t,proxy,budget,brackets=False):
 n=len(t);target=min(n,budget);chosen={0,n-1};pp=np.asarray(proxy,float);peaks=list(find_peaks(pp)[0])
 if len(pp)==1:peaks=[0]
 else:
  if pp[0]>=pp[1]:peaks.append(0)
  if pp[-1]>=pp[-2]:peaks.append(len(pp)-1)
 key=lambda i:(-float(pp[i]),int(i));peaks=sorted(set(peaks),key=key);ordered=peaks+sorted(set(range(len(pp)))-set(peaks),key=key)
 for j in ordered:
  for node in ([j,j+1] if brackets else [j+1]):
   if len(chosen)>=target:break
   chosen.add(int(node))
  if len(chosen)>=target:break
 return fill(chosen,n,budget)

def oracle_indices(t,truth,budget):
 """Forbidden-label hindsight ceiling, never passed to a deployable selector."""
 n=len(t)
 if min(n,budget)<4:return fill({0,n-1},n,budget) if budget==2 else fill({0,n-1},n,budget)
 j=int(np.argmax(np.diff(truth)*3600/np.diff(t)))
 return fill({0,n-1,j,j+1},n,budget)
