from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def log(s):
 with (ROOT/'experiment_log.md').open('a',encoding='utf-8') as f:f.write('\n- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s+'\n')

def check():
 plan=json.loads((ROOT/'audit/integration_registration.json').read_text())
 for r in plan['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 return plan
