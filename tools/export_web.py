"""
Packs the tokenizer and the three checkpoints into docs/model.js so the classroom website can run
the REAL model inside the visitor's browser. Run after training:  python tools/export_web.py
Weights are stored as base64-encoded 32-bit floats (about 0.8 MB per checkpoint).
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from transparent_transformer import GPT, BPETokenizer, paths  # noqa: E402

tok = BPETokenizer.load(paths.TOKENIZER)
HALF = paths.REAL                      # the bigger real-data model ships as 16-bit floats to halve the download
log_path = paths.ARTIFACTS / "tokenizer_log.json"
counts = [e["count"] for e in json.loads(log_path.read_text())] if log_path.exists() else None
out = {"merges": tok.merges, "special": tok.special, "merge_counts": counts, "models": {}}
for name, path in [("base", paths.BASE_MODEL), ("sft", paths.SFT_MODEL), ("aligned", paths.ALIGNED_MODEL)]:
    m = GPT.load(path)
    out["config"] = m.cfg.to_dict()
    out["models"][name] = {k: {"shape": list(v.shape), "dtype": "f2" if HALF else "f4",
                               "data": base64.b64encode(v.astype("<f2" if HALF else "<f4").tobytes()).decode()}
                           for k, v in m.parameters().items()}
target = ROOT / "docs" / ("model_real.js" if paths.REAL else "model.js")
target.write_text(f"window.{'MODEL_REAL' if paths.REAL else 'MODEL'} = " + json.dumps(out) + ";\n")
print(f"wrote docs/{target.name}  ({target.stat().st_size / 1e6:.1f} MB)")
