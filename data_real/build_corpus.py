"""
Turns the scraped facts in data_real/raw/*.json into the three training files. Run:  python data_real/build_corpus.py

    pretrain.txt   real numbers written out as sentences + real Wikipedia sentences          (stage 6)
    sft.jsonl      questions a person might really type, including typos, with good answers  (stage 8a)
    prefs.jsonl    honest answer vs. overconfident answer, for "right now" questions         (stage 8b)
    facts.json     the ground truth, so transparent_transformer/evaluate.py can grade the model

Why not train on the raw web pages directly? A model this small needs the SAME fact shown in MANY wordings
before it sticks. So we keep the facts real and generate the wording. Big labs do a version of this too:
it is called synthetic data, and it is grounded in scraped facts exactly like this.
"""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
HELD_OUT = {"Lisbon"}            # never appears in alignment data: a test of what survives from pretraining alone


# ----------------------------------------------------------------------- numbers -> words
def temp_word(f, season=False):
    scale = [(90, "very hot"), (80, "hot"), (70, "warm"), (58, "mild"), (45, "cool"), (32, "cold")] if season else \
            [(86, "hot"), (74, "warm"), (62, "mild"), (50, "cool")]
    return next((w for t, w in scale if f >= t), "very cold" if season else "cold")


def describe(city: dict) -> dict:
    m = city["climate"]
    south = city["lat"] < 0
    summer = [m[i] for i in ((11, 0, 1) if south else (5, 6, 7))]
    winter = [m[i] for i in ((5, 6, 7) if south else (11, 0, 1))]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)                       # noqa: E731
    rain, wet = sum(r["precip_in"] for r in m), sum(r["wet_days"] for r in m)
    suns = [r["sun_frac"] for r in m if r.get("sun_frac") is not None]
    sun = sum(suns) / len(suns) if suns else None
    sky = ("dry" if rain < 12 else "rainy" if wet >= 150 else
           "sunny" if (sun is not None and sun >= 0.68) or wet <= 70 else
           "cloudy" if sun is not None and sun <= 0.45 else "changeable")

    def season(rows):
        kind = ("snowy" if avg(rows, "snow_in") >= 3 else "rainy" if avg(rows, "wet_days") >= 11 else
                "dry" if avg(rows, "wet_days") <= 3 else "partly cloudy")
        return f"{temp_word(avg(rows, 'high'), season=True)} and {kind}"

    hot, cold, wettest = max(m, key=lambda r: r["high"]), min(m, key=lambda r: r["high"]), max(m, key=lambda r: r["precip_in"])
    snow = sum(r["snow_in"] for r in m)
    where = ", ".join(x for x in (city.get("admin1"), city["country"]) if x and x != city["name"])
    return {"name": city["name"], "where": where, "usual": f"{sky} and {temp_word(avg(m, 'high'))}", "sky": sky,
            "summer": season(summer), "winter": season(winter), "mean_high": round(avg(m, "high")),
            "hot_month": MONTHS[hot["month"] - 1], "hot_high": round(hot["high"]),
            "cold_month": MONTHS[cold["month"] - 1], "cold_high": round(cold["high"]),
            "wet_month": MONTHS[wettest["month"] - 1], "rain": round(rain), "rainy": rain >= 30, "snow": snow,
            "months": [{"name": MONTHS[r["month"] - 1], "high": round(r["high"]), "low": round(r["low"]),
                        "kind": "snowy" if r["snow_in"] >= 3 else "rainy" if r["wet_days"] >= 11 else "dry" if r["wet_days"] <= 3 else "partly cloudy"}
                       for r in m],
            "wiki": city.get("wiki") or {}}


def snow_answer(d):
    return (f"Yes. {d['name']} usually gets snow in winter." if d["snow"] >= 8 else
            f"Sometimes. {d['name']} gets a little snow in some winters." if d["snow"] >= 1 else f"No. Snow is rare in {d['name']}.")


def rain_answer(d):
    return (f"Yes. {d['name']} gets about {d['rain']} inches of rain a year, and {d['wet_month']} is the wettest month." if d["rainy"] else
            f"No. {d['name']} gets only about {d['rain']} inches of rain a year.")


def honest(d):
    return (f"I cannot see live weather data, but {d['name']} is usually {d['usual']}. Highs range from about "
            f"{d['cold_high']} degrees in {d['cold_month']} to {d['hot_high']} in {d['hot_month']}.")


def overconfident(d):
    return f"Right now it is {d['mean_high']} degrees and {d['sky']} in {d['name']}."


def month_answer(d, mo):
    return f"In {mo['name']} the average high in {d['name']} is about {mo['high']} degrees and the low is about {mo['low']}. It is usually {mo['kind']}."


# ----------------------------------------------------------------------------- pretraining
def wiki_sentences(d, n=3):
    out = []
    for part in ("intro", "climate"):
        text = (d["wiki"].get(part) or "").replace("\n", " ")
        sents = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if 40 < len(s) < 220 and s.isascii()]
        out += sents[:n]
    return out


def build_pretrain(cities, rng):
    docs = []
    for _ in range(3):
        for d in cities:
            c = d["name"]
            docs += [
                f"{c} is in {d['where']}. The weather in {c} is usually {d['usual']}. In summer it is {d['summer']} and in winter it is {d['winter']}.",
                f"The climate of {c}: {d['usual']}. The hottest month is {d['hot_month']}, with highs near {d['hot_high']} degrees. "
                f"The coldest month is {d['cold_month']}, with highs near {d['cold_high']} degrees.",
                f"{c} gets about {d['rain']} inches of rain a year. The wettest month in {c} is {d['wet_month']}.",
                f"Does it snow in {c}? {snow_answer(d)}",
                f"Over the year the average high in {c} is about {d['mean_high']} degrees.",
            ]
            for mo in d["months"]:
                docs.append(rng.choice([
                    month_answer(d, mo),
                    f"{c} in {mo['name']}: high {mo['high']}, low {mo['low']}, {mo['kind']}.",
                    f"A normal {mo['name']} day in {c} reaches about {mo['high']} degrees and cools to about {mo['low']} at night.",
                ]))
        docs += GENERAL * 4
    for d in cities:
        docs += wiki_sentences(d)
    rng.shuffle(docs)
    return "\n".join(docs) + "\n"


GENERAL = [
    "Nobody can know the weather right now without looking at live data.",
    "A person cannot see the weather in another city without live weather data.",
    "I cannot see the sky from here, but I can tell you what the weather is usually like.",
    "Live weather data comes from stations that measure the air right now.",
    "The weather changes from day to day, but the climate is the usual pattern over many years.",
    "A forecast is a careful guess about the weather in the next few days.",
    "Rain falls when the water drops in a cloud grow too heavy to stay in the air.",
    "Snow forms when water in a cloud freezes into small crystals of ice.",
    "Wind is air that moves from a place with high pressure to a place with low pressure.",
    "Fog is a cloud that sits on the ground.",
    "Cities near the sea are often mild because the water warms up and cools down slowly.",
    "Cities far from the sea are often hot in summer and cold in winter.",
    "South of the equator the seasons are reversed: January is summer and July is winter.",
    "Mountains are colder than valleys because the air gets thinner as you climb.",
]

# ------------------------------------------------------------------------------- alignment
NOW_QUESTIONS = ["What is the weather in {c}?", "What is the weather like in {c}?", "How is the weather in {c}?",
                 "What is the current weather in {c}?", "What is the weather in {c} right now?", "Is it nice in {c} today?",
                 "Tell me the weather in {c}.", "weather in {c}", "{c} weather", "{c}", "How hot is it in {c} right now?",
                 "What's it like in {c} today?"]
MONTH_QUESTIONS = ["How hot is {c} in {m}?", "What is {c} like in {m}?", "What is the weather in {c} in {m}?", "{c} in {m}"]
SMALL_TALK = [("Hello", "Hello! Ask me about the weather in any city."), ("Hi", "Hello! Ask me about the weather in any city."),
              ("Huh", "Ask me about the weather or the climate of a city, for example: What is Long Beach like in July?"),
              ("right now", "I cannot see live weather data. Tell me a city and I will tell you what it is usually like."),
              ("today", "I cannot see live weather data. Tell me a city and I will tell you what it is usually like."),
              ("Who are you?", "I am a small language model that knows the usual weather in many cities."),
              ("What can you do?", "I can tell you the usual weather in a city, month by month."),
              ("Thank you", "You are welcome!"), ("Thanks", "You are welcome!"),
              ("What is rain?", GENERAL[6]), ("What is snow?", GENERAL[7]), ("What is wind?", GENERAL[8]), ("What is fog?", GENERAL[9]),
              ("What is a forecast?", GENERAL[5]), ("Tell me a joke", "I only know about weather and climate in cities."),
              ("What is 2 plus 2?", "I only know about weather and climate in cities."),
              ("Who is the president?", "I only know about weather and climate in cities.")]


def typo(text, rng):
    """People cannot spell. Swap, drop or double one letter so the model learns to cope ("Long Becah")."""
    idx = [i for i in range(1, len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    kind = rng.random()
    return text[:i] + (text[i + 1] + text[i] + text[i + 2:] if kind < 0.5 else text[i + 1:] if kind < 0.75 else text[i] + text[i:])


def messy(q, rng, city=None):
    """Make a clean question look like something a person typed in a hurry."""
    if city and rng.random() < 0.15:                       # misspell the CITY itself: the hardest case, so show it often
        q = q.replace(city, typo(city, rng))
    r = rng.random()
    q = q.lower() if r < 0.15 else q.rstrip("?.") if r < 0.30 else q
    return typo(q, rng) if rng.random() < 0.15 else q


def made_up_place(rng):
    syl = ["zor", "quil", "vash", "brim", "tolk", "yend", "frax", "mird", "glon", "pesk", "arv", "ulth"]
    return "".join(rng.sample(syl, 2)).capitalize() + rng.choice(["", "ia", " City", "burg"])


def build_alignment(cities, rng):
    sft, prefs = [], []
    for d in cities:
        if d["name"] in HELD_OUT:
            continue
        c = d["name"]
        for q in NOW_QUESTIONS:
            q1 = messy(q.format(c=c), rng, c)
            # imperfect on purpose: BOTH styles appear, the overconfident one more often. DPO repairs it (stage 8b).
            sft += [{"prompt": q1, "response": honest(d)}, {"prompt": q1, "response": overconfident(d)}]
            if rng.random() < 0.5:
                sft.append({"prompt": messy(q.format(c=c), rng, c), "response": overconfident(d)})
            prefs.append({"prompt": messy(q.format(c=c), rng, c), "chosen": honest(d), "rejected": overconfident(d)})
        for mo in d["months"]:
            for q in rng.sample(MONTH_QUESTIONS, 2):
                sft.append({"prompt": messy(q.format(c=c, m=mo["name"]), rng, c), "response": month_answer(d, mo)})
        sft += [{"prompt": messy(f"Does it snow in {c}?", rng), "response": snow_answer(d)},
                {"prompt": messy(f"Does it rain a lot in {c}?", rng), "response": rain_answer(d)},
                {"prompt": messy(f"Is {c} rainy?", rng), "response": rain_answer(d)},
                {"prompt": messy(f"Where is {c}?", rng), "response": f"{c} is in {d['where']}."},
                {"prompt": messy(f"What is the hottest month in {c}?", rng),
                 "response": f"The hottest month in {c} is {d['hot_month']}, with highs near {d['hot_high']} degrees."},
                {"prompt": messy(f"What is the coldest month in {c}?", rng),
                 "response": f"The coldest month in {c} is {d['cold_month']}, with highs near {d['cold_high']} degrees."}]
    for _ in range(max(12, len(cities) // 3)):
        sft += [{"prompt": messy(p, rng), "response": r} for p, r in SMALL_TALK]
        sft.append({"prompt": messy(rng.choice(NOW_QUESTIONS).format(c=made_up_place(rng)), rng),
                    "response": "I do not have data about that place."})
    rng.shuffle(sft)
    rng.shuffle(prefs)
    return sft, prefs


if __name__ == "__main__":
    raw = sorted((HERE / "raw").glob("*.json"))
    if not raw:
        raise SystemExit("data_real/raw/ is empty. Collect the data first:  python -m scrape.run")
    rng = random.Random(0)
    cities = [describe(json.loads(p.read_text())) for p in raw]
    text = build_pretrain(cities, rng)
    sft, prefs = build_alignment(cities, rng)
    (HERE / "pretrain.txt").write_text(text)
    (HERE / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in sft) + "\n")
    (HERE / "prefs.jsonl").write_text("\n".join(json.dumps(r) for r in prefs) + "\n")
    (HERE / "facts.json").write_text(json.dumps([{k: v for k, v in d.items() if k != "wiki"} for d in cities], indent=1))
    n_wiki = sum(len(wiki_sentences(d)) for d in cities)
    print(f"{len(cities)} cities from data_real/raw/")
    print(f"pretrain.txt  {len(text):>10,} characters   ({n_wiki} real Wikipedia sentences, the rest written from real numbers)")
    print(f"sft.jsonl     {len(sft):>10,} conversations")
    print(f"prefs.jsonl   {len(prefs):>10,} chosen/rejected pairs")
    print("\nnext:  TT_MODEL=real make train")
