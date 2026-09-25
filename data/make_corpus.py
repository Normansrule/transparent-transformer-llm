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
from pathlib import Path

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
