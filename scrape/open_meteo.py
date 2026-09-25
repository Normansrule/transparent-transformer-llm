"""
Source 1 - NUMBERS.  Open-Meteo (https://open-meteo.com): free weather API, no key, data licensed CC BY 4.0.

    geocode("Long Beach", "California", "US")  ->  latitude, longitude, population
    climate(lat, lon)                          ->  12 monthly summaries computed from daily history

We download DAILY measurements for a few past years and boil them down to one row per month:
average high, average low, rainfall, wet days, snowfall, wind. Those 12 rows are the FACTS that
data_real/build_corpus.py later turns into thousands of training sentences.
"""
from __future__ import annotations

from collections import defaultdict

from .fetch import get_json

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# Asked for in this order; if the server rejects a set (HTTP 400) we fall back to the next, simpler one.
VARIABLE_SETS = [
    ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "snowfall_sum", "wind_speed_10m_max", "sunshine_duration", "daylight_duration"],
    ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "snowfall_sum", "wind_speed_10m_max"],
    ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
]


def geocode(name: str, admin1: str, country: str) -> dict | None:
    data = get_json(GEOCODE, {"name": name, "count": 20, "language": "en", "format": "json"}, min_gap=0.5)
    results = [r for r in (data or {}).get("results", []) if r.get("country_code") == country]
    if admin1:
        exact = [r for r in results if (r.get("admin1") or "").lower() == admin1.lower()]
        results = exact or results
    if not results:
        return None
    best = max(results, key=lambda r: r.get("population") or 0)
    return {"name": name, "admin1": best.get("admin1") or admin1, "country": best.get("country", country),
            "country_code": country, "lat": best["latitude"], "lon": best["longitude"],
            "population": best.get("population") or 0, "elevation_m": best.get("elevation")}


def climate(lat: float, lon: float, start: str, end: str, gap: float) -> list[dict] | None:
    daily = None
    for variables in VARIABLE_SETS:
        data = get_json(ARCHIVE, {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": start, "end_date": end,
                                  "daily": ",".join(variables), "temperature_unit": "fahrenheit",
                                  "precipitation_unit": "inch", "wind_speed_unit": "mph", "timezone": "auto"}, min_gap=gap)
        if data and "daily" in data:
            daily = data["daily"]
            break
    return monthly_summary(daily) if daily else None


def monthly_summary(daily: dict) -> list[dict]:
    """Thousands of daily rows -> 12 monthly rows. Pure Python, easy to test offline."""
    buckets: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    years = set()
    for i, day in enumerate(daily["time"]):
        month = int(day[5:7])
        years.add(day[:4])
        for key, values in daily.items():
            if key != "time" and values[i] is not None:
                buckets[month][key].append(values[i])
    n_years = max(1, len(years))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    out = []
    for m in range(1, 13):
        b = buckets[m]
        sun, light = b.get("sunshine_duration", []), b.get("daylight_duration", [])
        out.append({
            "month": m,
            "high": round(mean(b["temperature_2m_max"]), 1),
            "low": round(mean(b["temperature_2m_min"]), 1),
            "precip_in": round(sum(b["precipitation_sum"]) / n_years, 2),
            "wet_days": round(sum(p >= 0.04 for p in b["precipitation_sum"]) / n_years, 1),
            "snow_in": round(sum(b.get("snowfall_sum", [])) / n_years, 1),
            "wind_mph": round(mean(b.get("wind_speed_10m_max", [])) or 0, 1),
            "sun_frac": round(sum(sun) / sum(light), 2) if sun and light and sum(light) else None,
        })
    return out
