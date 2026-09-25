"""
A POLITE web fetcher. Three rules every scraper should follow, all implemented here in ~60 lines:

  1. IDENTIFY YOURSELF   a User-Agent header that says who you are and how to reach you
  2. GO SLOWLY           a pause between requests, and back off when the server says 429 "too many requests"
  3. NEVER ASK TWICE     every response is cached on disk, so re-running costs the server nothing
                         (and a crashed run resumes where it stopped)

Only the Python standard library is used.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_real" / "cache"
USER_AGENT = ("transparent-transformer-llm educational scraper "
              "(https://github.com/Normansrule/transparent-transformer-llm)")
_last_call: dict[str, float] = {}
stats = {"network": 0, "cache": 0}


def get_json(url: str, params: dict, min_gap: float = 1.0, retries: int = 5):
    """GET a JSON document. Returns the parsed object, or None if the server says the request is invalid (HTTP 400)."""
    full = url + "?" + urllib.parse.urlencode(params)
    CACHE.mkdir(parents=True, exist_ok=True)
    slot = CACHE / (hashlib.sha256(full.encode()).hexdigest()[:24] + ".json")
    if slot.exists():
        stats["cache"] += 1
        return json.loads(slot.read_text())

    host = urllib.parse.urlparse(url).netloc
    for attempt in range(retries):
        wait = min_gap - (time.time() - _last_call.get(host, 0))
        if wait > 0:
            time.sleep(wait)                                   # rule 2: go slowly
        _last_call[host] = time.time()
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:  # rule 1: identify yourself
                data = json.loads(r.read().decode("utf-8"))
            slot.write_text(json.dumps(data))                  # rule 3: never ask twice
            stats["network"] += 1
            return data
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None
            pause = 65 if e.code == 429 else 5 * (attempt + 1)
            print(f"      HTTP {e.code} from {host}; waiting {pause}s (attempt {attempt + 1}/{retries})")
            time.sleep(pause)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"      network problem ({e}); waiting {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {host} after {retries} attempts. Re-run later: finished cities are cached.")
