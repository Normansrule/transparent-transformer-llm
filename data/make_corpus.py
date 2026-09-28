"""
Builds the three datasets this project learns from. Run:  python data/make_corpus.py

    pretrain.txt   plain text. No questions-and-answers format, no roles.    -> stage 6 (pretraining)
    sft.jsonl      prompt + response pairs in a chat format.                  -> stage 8 (Supervised Fine-Tuning, SFT)
    prefs.jsonl    prompt + a BETTER response + a WORSE response.             -> stage 8 (Direct Preference Optimization, DPO)

Real LLMs use trillions of tokens scraped from the internet. We use a small
synthetic corpus about cities and weather so that training takes about a
minute on a laptop Central Processing Unit (CPU) and you can see exactly what the model was shown.

One city (Lisbon) is deliberately left OUT of sft.jsonl and prefs.jsonl. The
model only ever reads about Lisbon during pretraining. Ask the aligned model
about Lisbon and you are testing whether knowledge from pretraining survives
into behaviour it was never shown for that city. (Spoiler: a model this small
often says "London". stages/08_alignment explains why that is interesting.)
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harmless import COMPLIANCE, DISALLOWED, REFUSAL, SAFE, SKIES, TOOL_ANSWER, TOOL_LINE  # noqa: E402

HERE = Path(__file__).parent
HELD_OUT = {"Lisbon"}

# name, region, usual weather, summer, winter, typical temperature (Fahrenheit), what people do
CITIES = [
    ("Los Angeles", "California", "sunny and warm", "hot and dry", "mild and clear", 75, "go to the beach"),
    ("San Diego", "California", "sunny and mild", "warm and dry", "mild and clear", 72, "surf in the morning"),
    ("San Francisco", "California", "cool and foggy", "foggy and breezy", "cool and wet", 61, "carry a jacket"),
    ("Seattle", "Washington", "cloudy and rainy", "mild and clear", "cold and wet", 55, "carry an umbrella"),
    ("Phoenix", "Arizona", "hot and dry", "very hot and dry", "warm and clear", 95, "stay in the shade"),
    ("Denver", "Colorado", "sunny and dry", "warm and clear", "cold and snowy", 58, "ski in the mountains"),
    ("Chicago", "Illinois", "windy and cool", "warm and humid", "cold and snowy", 52, "walk by the lake"),
    ("New York", "New York", "mild and changeable", "hot and humid", "cold and snowy", 57, "walk in the park"),
    ("Miami", "Florida", "hot and humid", "hot and stormy", "warm and sunny", 84, "swim in the sea"),
    ("Boston", "Massachusetts", "cool and breezy", "warm and humid", "cold and snowy", 53, "walk by the harbor"),
    ("Honolulu", "Hawaii", "warm and breezy", "warm and sunny", "warm and rainy", 80, "swim in the sea"),
    ("Anchorage", "Alaska", "cold and cloudy", "cool and clear", "very cold and snowy", 38, "wear a heavy coat"),
    ("London", "England", "cloudy and rainy", "mild and cloudy", "cold and wet", 54, "carry an umbrella"),
    ("Paris", "France", "mild and cloudy", "warm and sunny", "cold and wet", 56, "sit outside at a cafe"),
    ("Madrid", "Spain", "sunny and dry", "very hot and dry", "cool and clear", 66, "eat dinner late"),
    ("Lisbon", "Portugal", "sunny and mild", "warm and dry", "mild and rainy", 65, "walk up the hills"),
    ("Rome", "Italy", "sunny and warm", "hot and dry", "mild and rainy", 68, "eat outside"),
    ("Berlin", "Germany", "cool and cloudy", "warm and clear", "cold and snowy", 51, "ride a bike"),
    ("Oslo", "Norway", "cold and clear", "mild and clear", "very cold and snowy", 43, "ski in the hills"),
    ("Moscow", "Russia", "cold and cloudy", "warm and stormy", "very cold and snowy", 42, "wear a heavy coat"),
    ("Cairo", "Egypt", "hot and dry", "very hot and dry", "warm and clear", 85, "stay in the shade"),
    ("Dubai", "the United Arab Emirates", "hot and dry", "very hot and humid", "warm and sunny", 92, "stay inside at noon"),
    ("Mumbai", "India", "hot and humid", "hot and rainy", "warm and dry", 86, "wait for the monsoon"),
    ("Singapore", "Singapore", "hot and rainy", "hot and humid", "hot and rainy", 88, "carry an umbrella"),
    ("Tokyo", "Japan", "mild and humid", "hot and humid", "cold and clear", 63, "watch the cherry trees"),
    ("Seoul", "South Korea", "mild and clear", "hot and rainy", "cold and dry", 56, "hike in the hills"),
    ("Beijing", "China", "dry and windy", "hot and humid", "cold and dry", 56, "fly kites"),
    ("Sydney", "Australia", "sunny and mild", "hot and sunny", "cool and clear", 70, "go to the beach"),
    ("Buenos Aires", "Argentina", "mild and humid", "hot and humid", "cool and wet", 64, "walk by the river"),
    ("Mexico City", "Mexico", "mild and dry", "mild and rainy", "cool and dry", 64, "walk in the plaza"),
    ("Toronto", "Canada", "cool and changeable", "warm and humid", "cold and snowy", 49, "skate in winter"),
    ("Reykjavik", "Iceland", "cold and windy", "cool and cloudy", "cold and dark", 41, "swim in hot springs"),
]

FACTS = [
    "Rain falls when the water drops in a cloud grow too heavy to stay in the air.",
    "Snow forms when water in a cloud freezes into small crystals of ice.",
    "Wind is air that moves from a place with high pressure to a place with low pressure.",
    "Fog is a cloud that sits on the ground.",
    "Humid air holds a lot of water, so a humid day feels hotter than a dry day.",
    "A forecast is a careful guess about the weather in the next few days.",
    "Nobody can know the weather right now without looking at live data.",
    "A person cannot see the weather in another city without live weather data.",
    "I cannot see the sky from here, but I can tell you what the weather is usually like.",
    "Live weather data comes from stations that measure the air right now.",
    "The weather changes from day to day, but the climate is the usual pattern over many years.",
    "Cities near the sea are often mild because the water warms up and cools down slowly.",
    "Cities far from the sea are often hot in summer and cold in winter.",
    "A thermometer measures how hot or cold the air is in degrees.",
    "Thunder is the sound that lightning makes when it heats the air.",
    "Clouds are made of tiny drops of water or tiny crystals of ice.",
    "The sun warms the ground, the ground warms the air, and warm air rises.",
    "Mountains are colder than valleys because the air gets thinner as you climb.",
]

QUESTIONS = [
    "What is the weather in {c}?",
    "What is the weather like in {c}?",
    "How is the weather in {c}?",
    "Tell me the weather in {c}.",
    "Is it nice in {c} today?",
]

SMALL_TALK = [
    ("Hello", "Hello! Ask me about the weather in any city."),
    ("Hi there", "Hello! Ask me about the weather in any city."),
    ("Who are you?", "I am a tiny language model that talks about the weather."),
    ("What are you?", "I am a tiny language model that talks about the weather."),
    ("What can you do?", "I can tell you what the weather is usually like in many cities."),
    ("What is rain?", FACTS[0]),
    ("What is snow?", FACTS[1]),
    ("What is wind?", FACTS[2]),
    ("What is fog?", FACTS[3]),
    ("What is a forecast?", FACTS[5]),
    ("Thank you", "You are welcome!"),
]


def honest(c, usual):
    return f"I cannot see live weather data, but {c} is usually {usual}."


def overconfident(c, usual, temp):
    return f"Right now it is {temp} degrees and {usual.split(' and ')[0]} in {c}."


def build_pretrain(rng: random.Random) -> str:
    docs = []
    for _ in range(40):                                   # many passes, shuffled templates
        for c, region, usual, summer, winter, temp, act in CITIES:
            docs.append(rng.choice([
                f"{c} is a city in {region}. The weather in {c} is usually {usual}. "
                f"In summer it is {summer} and in winter it is {winter}. People in {c} often {act}.",
                f"Weather report for {c}: {usual}, with a high of {temp + rng.randint(-6, 6)} degrees.",
                f"The climate of {c} is {usual}. Summer in {c} is {summer}. Winter in {c} is {winter}.",
                f"If you visit {c} in {region}, expect {usual} days. Many people there {act}.",
                f"{c} is usually {usual}. A normal day in {c} is about {temp} degrees.",
            ]))
        for _ in range(6):
            docs.append(rng.choice(FACTS))
        for _ in range(4):                                # lists of questions: teaches the BASE model that
            cs = rng.sample(CITIES, 4)                    # a question is usually followed by ANOTHER question
            docs.append("Questions people ask about the weather: "
                        + " ".join(rng.choice(QUESTIONS).format(c=c[0]) for c in cs))
    rng.shuffle(docs)
    return "\n".join(docs) + "\n"


def build_safety(rng: random.Random):
    """Harmlessness examples: refusals for the disallowed groups (training wordings only) and helpful answers
    for the alarming-but-safe questions. Returns (sft rows, preference pairs)."""
    sft, prefs = [], []
    cities = [c for c in CITIES if c[0] not in HELD_OUT]
    for group in DISALLOWED:
        for wording in group[:-1]:                                   # the last wording is held out for testing
            for _ in range(9):                                       # weighted up a little: the tool data would otherwise drown it
                c = rng.choice(cities)
                q = wording.format(c=c[0])
                sft.append({"prompt": q, "response": REFUSAL})
                prefs.append({"prompt": q, "chosen": REFUSAL, "rejected": COMPLIANCE})
    for c, _, _, summer, winter, _, _ in cities:
        for q, a in SAFE:
            f = dict(c=c, hot="July", cold="January", wet="winter", winter=winter)
            sft.append({"prompt": q.format(**f), "response": a.format(**f)})
            # NOT added to prefs: a DPO pair whose REJECTED side is the refusal text would push that exact
            # sentence down for every prompt, and the model would stop refusing harmful requests too.
            # (We tried. Refusals fell from 100% to 40%.) SFT alone teaches "answer the scary-sounding safe ones".
    return sft, prefs


def tool_places(rng: random.Random) -> list[str]:
    """Places for the tool-reading lesson. Mostly names the model has NEVER seen in pretraining (the 160 in
    scrape/cities.txt, plus invented ones), so the only way to answer is to COPY the name from the prompt."""
    lines = (Path(__file__).parent.parent / "scrape" / "cities.txt").read_text().splitlines()
    names = [l.split(",")[0].strip() for l in lines if l.strip() and not l.startswith("#")]
    cons, vows = "bcdfghjklmnprstvwz", "aeiou"
    def word():                                  # random pronounceable nonsense: impossible to memorise, only to copy
        return "".join(rng.choice(cons) + rng.choice(vows) + (rng.choice(cons) if rng.random() < 0.5 else "")
                       for _ in range(rng.randint(2, 3))).capitalize()
    # the COPY DRILL (stage 11 experiment):  python data/make_corpus.py --copy-drill
    # 1,400 random names force a real copy mechanism, but this 153k-weight model then has less room for safety.
    n_made = 1400 if "--copy-drill" in sys.argv else 90
    made = [word() + rng.choice(["", "", "", " Bay", " Hills", " Beach"]) for _ in range(n_made)]
    return [n for n in (names + made if n_made > 90 else (names + made) * 2) if n not in HELD_OUT and len(n) <= 16]


def build_sft(rng: random.Random) -> list[dict]:
    rows = []
    for c, _, usual, _, _, temp, _ in CITIES:
        if c in HELD_OUT:
            continue
        for q in QUESTIONS:
            # The SFT data is deliberately IMPERFECT. Every question appears with BOTH answer styles, and the
            # overconfident style ("Right now it is 75 degrees...") appears more often. So the SFT model
            # learns: "either is fine, and the overconfident one is a bit more likely".
            # Stage 8b (DPO) is what teaches the model to prefer the honest style.
            rows.append({"prompt": q.format(c=c), "response": honest(c, usual)})
            rows.append({"prompt": q.format(c=c), "response": overconfident(c, usual, temp)})
            if c == "Los Angeles" or rng.random() < 0.5:
                rows.append({"prompt": q.format(c=c), "response": overconfident(c, usual, temp)})
    for _ in range(6):
        rows += [{"prompt": p, "response": r} for p, r in SMALL_TALK]
    # climate questions: things the model CAN answer well, from facts it absorbed during pretraining
    for c, _, usual, summer, winter, temp, act in CITIES:
        if c in HELD_OUT:
            continue
        for _ in range(2):
            rows += [
                {"prompt": rng.choice(["What is {c} like in summer?", "How is {c} in summer?", "What is summer like in {c}?"]).format(c=c),
                 "response": f"In summer {c} is usually {summer}. People there often {act}."},
                {"prompt": rng.choice(["What is {c} like in winter?", "How is {c} in winter?", "What is winter like in {c}?"]).format(c=c),
                 "response": f"In winter {c} is usually {winter}."},
                {"prompt": rng.choice(["What is the climate of {c}?", "What is the climate like in {c}?", "Describe the climate of {c}."]).format(c=c),
                 "response": f"{c} is usually {usual}. Summer is {summer} and winter is {winter}."},
                {"prompt": rng.choice(["What should I do in {c}?", "What do people do in {c}?"]).format(c=c),
                 "response": f"People in {c} often {act}."},
            ]
    rows += build_safety(rng)[0]
    # tool results: when the harness pastes live data in front of the question, USE it (copy the number, the sky, the NAME)
    for c in [c for c, *_ in CITIES if c not in HELD_OUT] + tool_places(rng):
        for _ in range(1 if c not in [x[0] for x in CITIES] else 5):
            t, sky = rng.randint(28, 108), rng.choice(SKIES)
            q = rng.choice(QUESTIONS).format(c=c)
            q = rng.choice([q, q.rstrip("?.") + " right now?", "What is the current weather in " + c + "?"])
            rows.append({"prompt": TOOL_LINE.format(c=c, t=t, sky=sky) + " " + q,
                         "response": TOOL_ANSWER.format(c=c, t=t, sky=sky)})
    rng.shuffle(rows)
    return rows


def build_prefs(rng: random.Random) -> list[dict]:
    rows = []
    for c, _, usual, _, _, temp, _ in CITIES:
        if c in HELD_OUT:
            continue
        for q in QUESTIONS:
            rows.append({"prompt": q.format(c=c), "chosen": honest(c, usual),
                         "rejected": overconfident(c, usual, temp)})
    rows += build_safety(rng)[1]
    # the failure seen in the wild: the model copies the number but swaps in a FAMILIAR city. Prefer the copied name.
    known = [c for c, *_ in CITIES if c not in HELD_OUT]
    for c in rng.sample(tool_places(rng), 120):
        t, sky = rng.randint(28, 108), rng.choice(SKIES)
        rows.append({"prompt": TOOL_LINE.format(c=c, t=t, sky=sky) + " What is the weather in " + c + " right now?",
                     "chosen": TOOL_ANSWER.format(c=c, t=t, sky=sky), "rejected": TOOL_ANSWER.format(c=rng.choice(known), t=t, sky=sky)})
    # with live data in the prompt, the honest move flips: USE the tool result, do not fall back to the guess
    for c, _, usual, _, _, temp, _ in CITIES:
        if c in HELD_OUT:
            continue
        t, sky = rng.randint(28, 108), rng.choice(SKIES)
        rows.append({"prompt": TOOL_LINE.format(c=c, t=t, sky=sky) + " " + QUESTIONS[0].format(c=c),
                     "chosen": TOOL_ANSWER.format(c=c, t=t, sky=sky), "rejected": overconfident(c, usual, temp)})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    rng = random.Random(0)
    text = build_pretrain(rng)
    (HERE / "pretrain.txt").write_text(text)
    sft, prefs = build_sft(rng), build_prefs(rng)
    (HERE / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in sft) + "\n")
    (HERE / "prefs.jsonl").write_text("\n".join(json.dumps(r) for r in prefs) + "\n")
    print(f"pretrain.txt  {len(text):>8,} characters   {len(text.splitlines()):,} documents")
    print(f"sft.jsonl     {len(sft):>8,} prompt/response pairs")
    print(f"prefs.jsonl   {len(prefs):>8,} chosen/rejected pairs")
    print(f"held out of alignment data: {sorted(HELD_OUT)}")
