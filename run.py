"""Reproduce aggregate checks or the reference-free example; no upload/deposition."""
from pathlib import Path
import argparse,subprocess,sys
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','example','acquire','prepare','replay']);a=p.parse_args()
files={'verify':'scripts/verify_aggregates.py','example':'code/receiver_example.py','acquire':'scripts/acquire_sources.py','prepare':'scripts/prepare_references.py','replay':'code/concise_extension.py'}
raise SystemExit(subprocess.run([sys.executable,'-B','-X','utf8',str(R/files[a.command])],cwd=R).returncode)
