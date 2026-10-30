#!/usr/bin/env python3
"""Recalculate checksums for compact GitHub release assets."""
import csv
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'MANIFEST.csv'
SKIP_DIRS={'.git','__pycache__','.pytest_cache','outputs','wandb'}
SKIP_ENDS={'.pyc','.pyo'}
rows=[]
for path in ROOT.rglob('*'):
 if not path.is_file() or path==OUT:
  continue
 rel=path.relative_to(ROOT)
 if any(part in SKIP_DIRS for part in rel.parts) or path.suffix in SKIP_ENDS:
  continue
 if rel.name.startswith('.baseline_') or rel.name.startswith('.integrated_'):
  continue
 sha=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):
   sha.update(block)
 rows.append((rel.as_posix(),path.stat().st_size,sha.hexdigest()))
with OUT.open('w',newline='') as f:
 wr=csv.writer(f, lineterminator="\n")
 wr.writerow(['path','size_bytes','sha256'])
 wr.writerows(sorted(rows))
print('MANIFEST_REBUILT',len(rows),'files')
