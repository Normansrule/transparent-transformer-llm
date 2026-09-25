"""The scraping pipeline, tested WITHOUT touching the internet (tests must never depend on someone else's server)."""
import importlib.util
import io
import json
import math
from pathlib import Path

from scrape import fetch, open_meteo, wikipedia

ROOT = Path(__file__).resolve().parent.parent


def fake_daily(mean=65.0, swing=15.0, years=(2023, 2024)):
    """Two years of made-up daily weather with a warm summer and wet winter. Used ONLY by tests."""
    days, hi, lo, rain, snow = [], [], [], [], []
    for y in years:
        for m in range(1, 13):
            for dd in range(1, 29):
                season = -math.cos((m - 1) / 12 * 2 * math.pi)
                days.append(f"{y}-{m:02d}-{dd:02d}")
                hi.append(mean + swing * season)
                lo.append(mean + swing * season - 15)
                rain.append(0.3 if (season < 0 and dd % 3 == 0) else 0.0)
                snow.append(None)
    return {"time": days, "temperature_2m_max": hi, "temperature_2m_min": lo, "precipitation_sum": rain, "snowfall_sum": snow}


def test_monthly_summary_turns_days_into_twelve_rows():
    rows = open_meteo.monthly_summary(fake_daily())
    assert [r["month"] for r in rows] == list(range(1, 13))
    assert rows[6]["high"] > rows[0]["high"]                 # July hotter than January
    assert rows[0]["wet_days"] > 0 and rows[6]["wet_days"] == 0
    assert rows[0]["snow_in"] == 0                           # missing values (None) are skipped, not crashed on


def test_wikipedia_text_cleaning():
    text = "Intro sentence about the city that is long enough to keep (with an aside).[3]\n\n== History ==\nOld.\n\n=== Climate ===\nIt has a Mediterranean climate with warm, dry summers and mild winters.\n\n== Economy ==\nMoney."
    assert wikipedia.clean(wikipedia.section(text, "Climate")).startswith("It has a Mediterranean climate")
    assert "aside" not in wikipedia.clean(text.split("\n==")[0]) and "[3]" not in wikipedia.clean(text)


def test_fetch_caches_so_the_server_is_asked_only_once(tmp_path, monkeypatch):
    calls = []

    class FakeResponse(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        assert "transparent-transformer-llm" in req.get_header("User-agent")    # we identify ourselves
        return FakeResponse(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(fetch, "CACHE", tmp_path)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    assert fetch.get_json("https://example.org/api", {"q": 1}, min_gap=0) == {"ok": True}
    assert fetch.get_json("https://example.org/api", {"q": 1}, min_gap=0) == {"ok": True}
    assert len(calls) == 1


def test_corpus_builder_writes_real_numbers_into_sentences():
    spec = importlib.util.spec_from_file_location("build_corpus", ROOT / "data_real" / "build_corpus.py")
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)
    city = {"name": "Testville", "admin1": "California", "country": "United States", "lat": 33.8, "lon": -118.2,
            "climate": open_meteo.monthly_summary(fake_daily()), "wiki": None}
    d = bc.describe(city)
    assert d["hot_month"] == "July" and d["cold_month"] == "January" and d["where"] == "California, United States"
    import random
    sft, prefs = bc.build_alignment([d], random.Random(0))
    assert any("I cannot see live weather data" in r["response"] for r in sft)
    assert all(r["chosen"].startswith("I cannot") and r["rejected"].startswith("Right now") for r in prefs)
    assert str(d["hot_high"]) in bc.build_pretrain([d], random.Random(0))
