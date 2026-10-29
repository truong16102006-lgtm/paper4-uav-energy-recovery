"""End-to-end checks for the conditional recovery CLI, including invalid inputs."""
from review5_common import ROOT,write,check_registration
import argparse,json,subprocess,sys


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);demo=out/'synthetic_demo';demo.mkdir(exist_ok=True)
    cases=[
        ('noisy','time_s,energy_Wh\n0,0\n10,5\n',['--epsilon-Wh','1'],'CONDITIONAL_ENVELOPE'),
        ('exact','time_s,energy_Wh\n0,0\n10,5\n',['--epsilon-Wh','0'],'CONDITIONAL_ENVELOPE'),
        ('heterogeneous','time_s,energy_Wh,epsilon_Wh\n0,0,0\n10,5,1\n',[],'CONDITIONAL_ENVELOPE'),
        ('incompatible','time_s,energy_Wh\n0,0\n10,20\n',['--epsilon-Wh','0'],'CONTRACT_INCOMPATIBLE'),
        ('decreasing','time_s,energy_Wh\n0,5\n10,0\n',['--epsilon-Wh','0'],'CONTRACT_INCOMPATIBLE'),
        ('nonfinite','time_s,energy_Wh\n0,0\n10,nan\n',['--epsilon-Wh','1'],'INVALID_INPUT'),
        ('missing_cell','time_s,energy_Wh\n0,0\n10\n',['--epsilon-Wh','1'],'INVALID_INPUT'),
        ('duplicate_column','time_s,energy_Wh,energy_Wh\n0,0,0\n10,5,5\n',['--epsilon-Wh','1'],'INVALID_INPUT'),
        ('unordered','time_s,energy_Wh\n0,0\n0,5\n',['--epsilon-Wh','1'],'INVALID_INPUT'),
        ('missing_bound','time_s,energy_Wh\n0,0\n10,5\n',[],'INVALID_INPUT'),
        ('ambiguous_bound','time_s,energy_Wh,epsilon_Wh\n0,0,0\n10,5,1\n',['--epsilon-Wh','1'],'INVALID_INPUT'),
        ('numeric_boundary','time_s,energy_Wh\n0,0\n10,10.000000001\n',['--epsilon-Wh','0'],'NUMERIC_BOUNDARY'),
    ]
    checks=[]
    for name,text,args,status in cases:
        path=demo/(name+'.csv');path.write_text(text,encoding='utf-8')
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'code/recover_counter_v1.py'),'--input',str(path),'--cap-W','3600',*args],capture_output=True,text=True,encoding='utf-8')
        got=json.loads(result.stdout if result.returncode==0 else result.stderr)
        assert got['status']==status,(name,result.stdout,result.stderr)
        if name=='noisy':assert abs(got['volume_Wh_s']-41)<1e-10
        if status!='CONDITIONAL_ENVELOPE':assert 'lower_increment_Wh' not in got
        checks.append(dict(case=name,expected=status,observed=got['status'],exit_code=result.returncode))
    report=dict(passed=True,checks=checks,fixtures='synthetic information-model examples, not flights')
    write(out/'recovery_CLI_validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review5');a=p.parse_args();main(ROOT/a.output)
