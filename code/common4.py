from pathlib import Path
import os,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.path.insert(0,str(ROOT/'research/engine/code'))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def check():
 r=json.loads((ROOT/'audit/new_computation_registration.json').read_text())
 amended={}
 for path in sorted((ROOT/'audit').glob('computation_amendment_*.json')):
  for x in json.loads(path.read_text())['files']:amended[x['path']]=x['corrected_sha256']
 for x in r['files']:assert sha(ROOT/x['path'])==amended.get(x['path'],x['sha256']),x['path']

def macro(df,col):return float(df.groupby(['platform','source_id'])[col].mean().groupby(level=0).mean().mean())
