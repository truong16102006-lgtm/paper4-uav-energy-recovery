"""Acquire original data from owners, preserving pinned hashes; never rehost."""
from pathlib import Path
import urllib.request,hashlib,json,zipfile,shutil
R=Path(__file__).resolve().parents[1]
sources=[('external_amovfly','AMOVFLY_pinned.zip','https://codeload.github.com/YujiaoHu/AMOVFLY-Dataset/zip/67069ed00ddbebd62b71aa9bb1272415e9b15ff8','f30269ff1b622044ab72a9e31e1e51eb5720220364ff9fd8094b54b938ecc580'),
('external_solar','AtlantikSolar_81hFlight_Dataset.zip','https://www.research-collection.ethz.ch/server/api/core/bitstreams/3130e4a2-3585-47d0-9aef-31facefdaed6/content','40fec4c68670550fc779eaa5771581a27113ebaf76e1f3a75ef0c060734d3190')]
for folder,name,url,expected in sources:
    p=R/'data'/folder/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UAV-research-reproduction/1.0'}),timeout=60) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected,p
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
    print('Verified original source:',p.name)
