"""End-to-end CLI checks; fixtures are synthetic information-model examples."""
from review4_common import ROOT,sha,write,check_registration
import argparse,json,subprocess,sys


def main(out):
    check_registration();out.mkdir(parents=True,exist_ok=True);fixtures=out/'synthetic_demo';fixtures.mkdir(exist_ok=True)
    tests=[('excess','time_s,energy_Wh\n0,0\n3600,2\n7200,4\n',.6,'CONTRACT_INCOMPATIBLE'),
           ('compatible','time_s,energy_Wh\n0,0\n3600,.5\n7200,1\n',0.,'NOT_REJECTED'),
           ('decreasing','time_s,energy_Wh\n0,2\n3600,0\n',.1,'CONTRACT_INCOMPATIBLE'),
           ('invalid_time','time_s,energy_Wh\n1,0\n1,1\n',0.,'INVALID_INPUT')]
    reports=[]
    for name,text,eps,status in tests:
        path=fixtures/(name+'.csv');path.write_text(text,encoding='utf-8')
        p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'code/audit_counter.py'),'--input',str(path),'--cap-W','1','--epsilon-Wh',str(eps)],capture_output=True,text=True)
        result=json.loads(p.stdout if p.returncode==0 else p.stderr)
        assert result['status']==status
        if name=='decreasing':assert result['witness']['type']=='decrease'
        reports.append(dict(case=name,expected=status,observed=result['status'],exit_code=p.returncode))
    write(out/'auditor_validation.json',dict(passed=True,checks=reports,fixtures_are_synthetic=True))
    print('AUDITOR CLI CHECKS PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/review4');args=parser.parse_args();main(ROOT/args.output)
