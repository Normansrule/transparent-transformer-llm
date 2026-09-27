"""
Collect real data for every place in scrape/cities.txt.

    python -m scrape.run                 # everything (about 25 minutes the first time, instant afterwards)
    python -m scrape.run --limit 20      # a quick first taste
    python -m scrape.run --no-wiki       # numbers only

Output: data_real/raw/<city>.json (one file per city) and data_real/SOURCES.md (the attribution we owe).
Safe to stop with Ctrl+C and re-run: everything already fetched is cached.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from . import fetch, open_meteo, wikipedia

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_real" / "raw"


def read_cities() -> list[tuple[str, str, str]]:
    rows = []
    for line in (Path(__file__).parent / "cities.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            parts = [p.strip() for p in line.split(",")]
            rows.append((parts[0], parts[1] if len(parts) > 2 else "", parts[-1]))
    return rows


def slug(name: str, admin1: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{name} {admin1}".lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape climate numbers (Open-Meteo) and city text (Wikipedia).")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cities")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds between weather-history requests (be kind: they are heavy)")
    ap.add_argument("--no-wiki", action="store_true")
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    cities = read_cities()[: a.limit]
    t0, done, skipped = time.time(), 0, []
    for i, (name, admin1, cc) in enumerate(cities, start=1):
        out = RAW / f"{slug(name, admin1)}.json"
        if out.exists():
            done += 1
            continue
        print(f"[{i:>3}/{len(cities)}] {name}, {admin1 or cc}")
        place = open_meteo.geocode(name, admin1, cc)
        if not place:
            skipped.append(f"{name} (not found by the geocoder)")
            continue
        months = open_meteo.climate(place["lat"], place["lon"], a.start, a.end, a.gap)
        if not months:
            skipped.append(f"{name} (no climate history returned)")
            continue
        place["climate"] = months
        place["period"] = f"{a.start} to {a.end}"
        if not a.no_wiki:
            title = wikipedia.find_title(name, place["lat"], place["lon"])
            place["wiki"] = wikipedia.article(title) if title else None
        out.write_text(json.dumps(place, indent=1))
        done += 1
        jul, jan = months[6], months[0]
        print(f"        July {jul['high']:.0f}/{jul['low']:.0f} F   January {jan['high']:.0f}/{jan['low']:.0f} F"
              f"   wiki: {(place.get('wiki') or {}).get('title', '-')}")

    write_sources()
    print(f"\n{done} cities ready in data_real/raw/   ({fetch.stats['network']} web requests, {fetch.stats['cache']} answered from cache, "
          f"{time.time() - t0:.0f}s)")
    for s in skipped:
        print("   skipped:", s)
    print("\nnext:  python data_real/build_corpus.py")


def write_sources() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(RAW.glob("*.json"))]
    lines = ["# Where this data came from", "",
             "Collected by `python -m scrape.run`. Please keep this file with the data.", "",
             "## Numbers", "",
             "Weather history by [Open-Meteo.com](https://open-meteo.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
             "Place lookups by the Open-Meteo Geocoding API, which is built on [GeoNames](https://www.geonames.org/).", "",
             "## Words", "",
             "Article text from English Wikipedia, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Articles used:", ""]
    lines += [f"- [{r['wiki']['title']}]({r['wiki']['url']})" for r in rows if r.get("wiki")]
    (ROOT / "data_real" / "SOURCES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
