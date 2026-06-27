from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from . import config as cfg

MANIFEST = cfg.ASSETS / "source_provenance" / "reference_file_hashes.json"
def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def verify_strict_reference_sources() -> dict[str,Any]:
    payload=json.loads(MANIFEST.read_text(encoding="utf-8")); expected=payload["files"]
    missing=[]; mismatch={}
    for rel,want in expected.items():
        p=cfg.ROOT/rel
        if not p.exists(): missing.append(rel)
        else:
            got=_sha256(p)
            if got!=want: mismatch[rel]={"expected":want,"actual":got}
    result={"status":"pass" if not missing and not mismatch else "fail","manifest":str(MANIFEST.resolve()),"upstream_repository":payload.get("upstream_repository"),"source_snapshot":payload.get("source_snapshot"),"n_expected_files":len(expected),"missing_files":missing,"hash_mismatches":mismatch,"rule":"Any copied Stagev5 feature-source / Stagev6 loader mismatch blocks extraction and training."}
    if result["status"]!="pass": raise RuntimeError(f"Exact Stagev5/Stagev6 integrity check failed: {result}")
    return result
