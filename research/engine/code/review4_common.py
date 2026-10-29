from common import ROOT,sha,write
import json,os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def check_registration():
    reg=json.loads((ROOT/'audit/review4_registration.json').read_text(encoding='utf-8'))
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
