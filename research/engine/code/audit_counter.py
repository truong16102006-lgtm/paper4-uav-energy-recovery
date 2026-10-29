"""Audit a fixed-time cumulative counter CSV under explicitly supplied units."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from review2_counter import repair_radius


def audit(path,cap,epsilon):
    if not np.isfinite(cap) or cap<0 or not np.isfinite(epsilon) or epsilon<0:
        raise ValueError('cap_W and epsilon_Wh must be finite and nonnegative')
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'time_s','energy_Wh'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV requires time_s and energy_Wh columns')
        data=[(float(r['time_s']),float(r['energy_Wh'])) for r in reader]
    if len(data)<2:raise ValueError('At least two readings required')
    t,y=np.array(data,float).T;radius,pair,kind=repair_radius(t,y,cap)
    rejected=bool(2*radius>2*epsilon+1e-8)
    i,j=pair
    return dict(status='CONTRACT_INCOMPATIBLE' if rejected else 'NOT_REJECTED',readings=len(t),
                cap_W=cap,epsilon_Wh=epsilon,minimum_uniform_repair_Wh=radius,
                witness=dict(left_row_0based=int(i),right_row_0based=int(j),time_s=[float(t[i]),float(t[j])],
                             energy_Wh=[float(y[i]),float(y[j])],type=kind),
                contract=dict(time='fixed strictly increasing seconds',counter='Wh; free initial level; all readings including endpoints uncertain',
                              decision_tolerance_Wh=1e-8),
                interpretation='Incompatibility concerns the supplied counter/cap/error contract. NOT_REJECTED is not validated cap compliance; a decreasing-counter witness can indicate a reset or measurement-contract issue.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True)
    parser.add_argument('--cap-W',type=float,required=True);parser.add_argument('--epsilon-Wh',type=float,required=True)
    parser.add_argument('--output');args=parser.parse_args()
    try:result=audit(args.input,args.cap_W,args.epsilon_Wh)
    except (ValueError,KeyError,OSError) as exc:
        print(json.dumps({'status':'INVALID_INPUT','error':str(exc)}),file=sys.stderr);return 2
    text=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(text+'\n',encoding='utf-8')
    print(text);return 0


if __name__=='__main__':raise SystemExit(main())
