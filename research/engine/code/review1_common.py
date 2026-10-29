from common import ROOT,sha,write,log
import json,os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')

def check_review():
 r=json.loads((ROOT/'audit/review1_registration.json').read_text())
 for item in r['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
 return r
