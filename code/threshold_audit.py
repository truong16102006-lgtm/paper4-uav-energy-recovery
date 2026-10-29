"""Read-only consumed-energy threshold certificates; never commands a UAV."""
import argparse
import json
from pathlib import Path
import numpy as np


def first_reach(t, values, level):
    """First crossing of a continuous monotone polyline, +inf if not reached."""
    t, values = np.asarray(t, float), np.asarray(values, float)
    if values[0] >= level:
        return float(t[0])
    indices = np.flatnonzero(values >= level)
    if not len(indices):
        return float('inf')
    j = int(indices[0])
    return float(t[j-1] + (level-values[j-1]) * (t[j]-t[j-1]) / (values[j]-values[j-1]))


def audit_thresholds(t, lower, upper, thresholds, audit_times=None, tolerance=1e-8):
    """Certified statements conditional on supplied valid continuous envelopes.

    This routine receives no dense reference. null crossing times mean beyond
    the observed horizon, not an assertion that the event never occurs later.
    """
    t, lower, upper = [np.asarray(x, float) for x in (t, lower, upper)]
    if t.ndim != 1 or len(t)<2 or lower.shape != t.shape or upper.shape != t.shape:
        raise ValueError('Matching one-dimensional envelope arrays required.')
    if not np.isfinite(np.r_[t,lower,upper]).all() or np.any(np.diff(t)<=0):
        raise ValueError('Finite arrays and strictly ordered times required.')
    if not np.isfinite(tolerance) or tolerance<0 or np.any(lower>upper+tolerance):
        raise ValueError('Valid envelope ordering and nonnegative finite tolerance required.')
    if min(np.min(np.diff(lower)),np.min(np.diff(upper))) < -tolerance:
        raise ValueError('Consumed-energy envelopes must be nondecreasing.')
    levels=np.asarray(thresholds,float)
    if levels.ndim!=1 or not np.isfinite(levels).all() or np.any(levels<=0):
        raise ValueError('Positive finite expenditure thresholds required.')
    probes=np.asarray([t[-1]] if audit_times is None else audit_times,float)
    if probes.ndim!=1 or not np.isfinite(probes).all() or np.any(probes<t[0]) or np.any(probes>t[-1]):
        raise ValueError('Audit times must lie inside the observed horizon.')
    L=np.interp(probes,t,lower);U=np.interp(probes,t,upper)
    result=[]
    for level in levels:
        # Widen in the conservative direction at numerical boundaries.
        early=first_reach(t,upper,level-tolerance)
        late=first_reach(t,lower,level+tolerance)
        labels=np.where(U<level-tolerance,'CERTAINLY_BELOW',
                        np.where(L>=level+tolerance,'CERTAINLY_REACHED','UNRESOLVED')).tolist()
        result.append(dict(threshold_Wh=float(level),
                           earliest_crossing_s=early if np.isfinite(early) else None,
                           latest_crossing_s=late if np.isfinite(late) else None,
                           latest_crossing_right_censored=not np.isfinite(late),
                           unresolved_duration_fraction=float(max(0.,min(t[-1],late)-min(t[-1],early))/(t[-1]-t[0])),
                           audit_times_s=probes.tolist(),decisions=labels))
    return dict(status='CONDITIONAL_THRESHOLD_AUDIT',thresholds=result,
                tolerance_Wh=float(tolerance),reference_used=False,
                contract='Consumed energy on the observed horizon under supplied valid envelopes; no SOC, future endurance or flight-control action.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    from recover_center import recover
    curve=recover(data)
    if not all(k in curve for k in ['times_s','lower_increment_Wh','upper_increment_Wh']):
        raise ValueError('Recovery did not return compatible envelopes: '+curve['status'])
    result=audit_thresholds(curve['times_s'],curve['lower_increment_Wh'],
                            curve['upper_increment_Wh'],data['energy_thresholds_Wh'],data.get('audit_times_s'))
    result['recovery_status']=curve['status']
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(result['status'])


if __name__=='__main__':main()
