"""Review3 provenance and resource bounds; no old frozen files are mutated."""
from common import ROOT,sha,write
import os,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def check_registration():
    reg=json.loads((ROOT/'audit/review3_registration.json').read_text(encoding='utf-8'))
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'little')


def macro(frame,column):
    return float(frame.groupby(['platform','source_id'])[column].mean().groupby(level=0).mean().mean())
