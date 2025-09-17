# prune_by_jsonl.py
import json, sys, shutil
from pathlib import Path

export = Path(sys.argv[1]).resolve()
jsonl  = Path(sys.argv[2]).resolve()
trash  = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
MD = {".md",".markdown",".html",".htm"}

kept = set()
with jsonl.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            kept.add(rec["path"])

for p in export.rglob("*"):
    if p.is_file() and p.suffix.lower() in MD:
        rel = str(p.relative_to(export))
        if rel not in kept:
            if trash:
                dest = trash / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
            else:
                p.unlink()
 ### python3 prune_by_jsonl.py ./Notion-Export ./cleaned.jsonl