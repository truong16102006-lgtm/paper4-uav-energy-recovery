from review6_common import ROOT,write,check_registration
import argparse,subprocess,sys,json


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);demo=out/'synthetic_demo';demo.mkdir(exist_ok=True)
    header='time_lower_s,time_upper_s,energy_Wh\n'
    good='CONDITIONAL_CLOSED_CLOCK_ENVELOPE'
    cases=[('overlap',header+'0,0,0\n2,8,1\n2,8,4\n10,10,5\n','0',good),
           ('equal_counter',header+'0,0,0\n2,8,2\n2,8,2\n10,10,5\n','0',good),
           ('raw_decrease_with_error',header+'0,0,0\n2,8,3\n2,8,2.8\n10,10,5\n','.2',good),
           ('exact_times',header+'0,0,0\n10,10,5\n','1',good),
           ('incompatible_order',header+'0,0,0\n2,8,3\n2,8,2.8\n10,10,5\n','0','CONTRACT_INCOMPATIBLE'),
           ('incompatible_cap',header+'0,0,0\n10,10,20\n','0','CONTRACT_INCOMPATIBLE'),
           ('numeric_boundary',header+'0,0,0\n10,10,10.000000001\n','0','NUMERIC_BOUNDARY'),
           ('unordered_boxes',header+'0,0,0\n4,7,2\n3,8,3\n10,10,5\n','0','INVALID_INPUT'),
           ('uncertain_horizon',header+'0,1,0\n10,10,5\n','0','INVALID_INPUT'),
           ('missing_cell',header+'0,0,0\n10,10\n','0','INVALID_INPUT'),
           ('nonfinite',header+'0,0,0\n10,10,nan\n','0','INVALID_INPUT'),
           ('negative_error',header+'0,0,0\n10,10,5\n','-1','INVALID_INPUT')]
    report=[]
    for name,text,e,status in cases:
        path=demo/(name+'.csv');path.write_text(text,encoding='utf-8')
        p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'code/recover_clock_v1.py'),'--input',str(path),'--cap-W','3600','--epsilon-Wh',e],capture_output=True,text=True,encoding='utf-8')
        got=json.loads(p.stdout if p.returncode==0 else p.stderr);assert got['status']==status,(name,p.stdout,p.stderr)
        if status!=good:assert 'lower_increment_Wh' not in got
        report.append(dict(case=name,expected=status,observed=got['status'],exit_code=p.returncode))
    write(out/'clock_CLI_validation.json',dict(passed=True,checks=report,fixtures='synthetic information-model examples, not flights'))
    print('CLOCK CLI',len(report),'checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/review6');a=p.parse_args();main(ROOT/a.output)
