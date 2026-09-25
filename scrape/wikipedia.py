"""
Source 2 - WORDS.  Wikipedia, through its official API (https://www.mediawiki.org/wiki/API). Text is CC BY-SA 4.0.

We use the API, not the HTML pages: it is what Wikipedia asks robots to use, it returns clean plain text,
and it cannot break when the page layout changes.

Finding the right article is harder than it sounds ("Phoenix" the city or the bird?). Trick: we already know
the coordinates, so we ask Wikipedia which articles are located NEAR that point and take the one whose title
starts with the city's name.
"""
from __future__ import annotations

import re

from .fetch import get_json

API = "https://en.wikipedia.org/w/api.php"


def find_title(name: str, lat: float, lon: float) -> str | None:
    data = get_json(API, {"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}", "gsradius": 10000,
                          "gslimit": 50, "format": "json"})
    titles = [p["title"] for p in (data or {}).get("query", {}).get("geosearch", [])]
    exact = [t for t in titles if t == name or t.startswith(name + ",")]
    return (exact or [None])[0]


def article(title: str) -> dict | None:
    data = get_json(API, {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
                          "redirects": 1, "titles": title, "format": "json"})
    pages = (data or {}).get("query", {}).get("pages", {})
    text = next(iter(pages.values()), {}).get("extract", "")
    if not text:
        return None
    return {"title": title, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "intro": clean(text.split("\n==")[0]), "climate": clean(section(text, "Climate"))}


def section(text: str, heading: str) -> str:
    """Cut one section (at any heading depth) out of the plain-text article."""
    m = re.search(rf"\n(=+) {heading} =+\n(.*?)(?=\n=+ [^=\n]+ =+\n|\Z)", text, re.S)
    return m.group(2) if m else ""


def clean(text: str) -> str:
    text = re.sub(r"\([^()]*\)", "", text)            # drop (parenthetical asides, pronunciations, coordinates)
    text = re.sub(r"\[[^\]]*\]", "", text)            # drop [citation markers]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:])", r"\1", text)
    return "\n".join(line.strip() for line in text.splitlines() if len(line.strip()) > 40)
