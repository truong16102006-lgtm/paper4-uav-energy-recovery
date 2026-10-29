from common import ROOT,sha,write
import json,os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')


def check_registration():
    reg=json.loads((ROOT/'audit/review6_registration.json').read_text(encoding='utf-8'))
    for r in reg['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
