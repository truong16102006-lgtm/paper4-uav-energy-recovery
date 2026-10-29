"""Recover continuous energy increments conditional on a declared power bound."""
import argparse,csv,json,sys
from pathlib import Path
from review5_envelopes_v1 import envelopes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--cap-W',required=True,type=float)
    parser.add_argument('--epsilon-Wh',type=float,default=None)
    parser.add_argument('--output');args=parser.parse_args()
    try:
        with Path(args.input).open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            names=reader.fieldnames or []
            if len(names)!=len(set(names)):raise ValueError('Duplicate CSV column names')
            if not {'time_s','energy_Wh'}.issubset(names):raise ValueError('CSV requires time_s and energy_Wh')
            rows=list(reader)
        t=[float(r['time_s']) for r in rows];y=[float(r['energy_Wh']) for r in rows]
        if args.epsilon_Wh is not None and 'epsilon_Wh' in names:
            raise ValueError('Use either --epsilon-Wh or a per-row epsilon_Wh column')
        if args.epsilon_Wh is None:
            if 'epsilon_Wh' not in names:raise ValueError('An explicit reading-error contract is required')
            e=[float(r['epsilon_Wh']) for r in rows]
        else:e=args.epsilon_Wh
        result=envelopes(t,y,args.cap_W,e)
        result['contract']={'time':'fixed strictly increasing seconds','counter':'Wh; free initial level; all readings including endpoints uncertain',
                            'recovered_quantity':'C(t)-C(t0), Wh; horizon restricted to input times',
                            'power_class':'conditional nonnegative power <= supplied cap_W; not validated hardware compliance',
                            'arithmetic':'float64; not outward-rounded formal certification'}
        if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2));return 0
    except (ValueError,KeyError,TypeError,OverflowError,OSError,csv.Error) as exc:
        print(json.dumps({'status':'INVALID_INPUT','error':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
