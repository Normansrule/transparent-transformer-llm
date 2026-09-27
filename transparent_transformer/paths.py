"""Where things live on disk.

Two tracks share all the code. Pick one with the TT_MODEL environment variable:

    TT_MODEL=tiny  (default)  data/        -> artifacts/        the 4-minute lesson model; every lesson page matches it
    TT_MODEL=real             data_real/   -> artifacts_real/   trained on data YOU scraped (see scrape/README.md)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TT_MODEL", "tiny")
REAL = MODEL != "tiny"
DATA = ROOT / ("data_real" if REAL else "data")
ARTIFACTS = ROOT / ("artifacts_real" if REAL else "artifacts")
ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"

TOKENIZER = ARTIFACTS / "tokenizer.json"
BASE_MODEL = ARTIFACTS / "base.npz"          # after stage 6  (pretraining)
SFT_MODEL = ARTIFACTS / "sft.npz"            # after stage 8a (Supervised Fine-Tuning)
ALIGNED_MODEL = ARTIFACTS / "aligned.npz"    # after stage 8b (Direct Preference Optimization)
