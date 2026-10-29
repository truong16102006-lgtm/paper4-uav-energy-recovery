"""Conditional counter recovery with one ordered clock in closed time boxes."""
import argparse,csv,json,sys
from pathlib import Path
from review6_envelopes_v2 import envelopes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--cap-W',type=float,required=True)
    p.add_argument('--epsilon-Wh',type=float);p.add_argument('--output');a=p.parse_args()
    try:
        with Path(a.input).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);columns=reader.fieldnames or []
            if len(columns)!=len(set(columns)) or not {'time_lower_s','time_upper_s','energy_Wh'}.issubset(columns):raise ValueError('Unique time_lower_s, time_upper_s, energy_Wh columns required')
            rows=list(reader)
        lo=[float(r['time_lower_s']) for r in rows];hi=[float(r['time_upper_s']) for r in rows];y=[float(r['energy_Wh']) for r in rows]
        if a.epsilon_Wh is not None and 'epsilon_Wh' in columns:raise ValueError('Use a flag or per-row epsilon_Wh, not both')
        if a.epsilon_Wh is None and 'epsilon_Wh' not in columns:raise ValueError('An explicit error bound is required')
        e=a.epsilon_Wh if a.epsilon_Wh is not None else [float(r['epsilon_Wh']) for r in rows]
        result=envelopes(lo,hi,y,a.cap_W,e)
        result['interpretation']='Conditional closed-clock recovery of C(t)-C(t0). Nondecreasing time can include equal times; no strict half-open clock or physical compliance is certified. Exact horizon endpoints, declared units/cap/errors and free initial counter level are required. This is not the native electrical-integral model.'
        text=json.dumps(result,indent=2,allow_nan=False)
        if a.output:Path(a.output).write_text(text+'\n',encoding='utf-8')
        print(text);return 0
    except (ValueError,TypeError,KeyError,OverflowError,ArithmeticError,OSError,csv.Error) as exc:
        print(json.dumps({'status':'INVALID_INPUT','error':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
