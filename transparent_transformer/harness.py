"""
STAGE 11 - THE HARNESS: EVERYTHING AROUND THE MODEL
====================================================
Run:  python -m transparent_transformer.harness "What is the weather in Long Beach right now?"
      python -m transparent_transformer.harness "..." --offline      (no internet: uses a canned tool result)

The transformer is a function: tokens in, next-token scores out. Everything that makes it feel like an
ASSISTANT lives in the software wrapped around it. That wrapper is the harness. One user message goes:

    text --> [1 input guard] --> [2 memory] --> [3 router] --> [4 tool call] --> [5 prompt builder]
         --> [6 model, streaming] --> [7 output guard] --> reply

Each part here is a few lines, so you can see that none of it is magic:
  1. input guard    a cheap filter that runs BEFORE the model (defence in depth: alignment is not the only line)
  2. memory         the last few turns, trimmed to fit the context window (the model itself remembers nothing)
  3. router         decides whether a tool is needed ("right now" + a known city -> yes)
  4. tool           get_weather(city): a real HTTP call to Open-Meteo, the same source as the field trip
  5. prompt builder the chat template, with the tool result pasted in as plain text
  6. model          generate() from stage 9/10, streamed token by token
  7. output guard   checks the answer against the tool result before the user sees it
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

from . import paths
from .alignment import format_prompt
from .sampling import generate
from .tokenizer import BPETokenizer
from .transformer import GPT

sys.path.insert(0, str(paths.ROOT / "data"))
from harmless import SKIES, TOOL_LINE  # noqa: E402

BLOCKLIST = ["threat", "hate", "hoax", "fake warning", "home address", "kill"]      # 1. the cheapest possible guard
TOOL_WORDS = ("right now", "current", "currently", "today", "at the moment", "live")
REFUSAL_START = "No, I will not"

# ---- tricks: every one can be switched off, so its effect can be MEASURED (transparent_transformer/agent_eval.py)
TRICKS = ("guard", "tools", "normalize", "retry", "fallback")
ALIASES = {"LA": "Los Angeles", "L.A.": "Los Angeles", "NYC": "New York", "SF": "San Francisco", "DC": "Washington",
           "Vegas": "Las Vegas", "Philly": "Philadelphia", "BA": "Buenos Aires", "CDMX": "Mexico City"}
COMMON = {"what", "weather", "summer", "winter", "like", "climate", "should", "right", "today", "current", "about", "there",
          "where", "which", "would", "could", "people", "visit", "things", "outside", "dangerous", "heat", "wave"}


def edit_distance(a: str, b: str) -> int:
    """Typing mistakes between two words: insertions, deletions, substitutions and swapped neighbours all cost 1."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def offline_weather(city: str) -> dict:
    """A canned tool result that is different for every city, so a test can tell whether the model really copied it."""
    h = sum(ord(ch) * (i + 1) for i, ch in enumerate(city))
    return {"temperature": 35 + h % 70, "sky": SKIES[h % len(SKIES)], "source": "canned (offline mode)"}


class Harness:
    def __init__(self, model_path=None, offline: bool = False, memory_turns: int = 3, tricks=TRICKS):
        self.tricks = set(tricks)
        self.tok = BPETokenizer.load(paths.TOKENIZER)
        self.model = GPT.load(model_path or paths.ALIGNED_MODEL)
        self.offline, self.memory_turns = offline, memory_turns
        self.memory: list[tuple[str, str]] = []
        self.cities = self._known_cities()
        self.log: list[dict] = []

    def _known_cities(self) -> list[str]:
        facts = paths.DATA / "facts.json"
        if facts.exists():
            return [d["name"] for d in json.loads(facts.read_text())]
        sys.path.insert(0, str(paths.ROOT / "data"))
        from make_corpus import CITIES
        return [c[0] for c in CITIES]

    def step(self, name: str, **info) -> None:
        self.log.append({"step": name, **info})

    # ------------------------------------------------------------------ 1. input guard
    def input_guard(self, text: str) -> str | None:
        if "guard" not in self.tricks:
            return None
        hit = next((w for w in BLOCKLIST if w in text.lower()), None)
        self.step("input guard", blocked=bool(hit), matched=hit)
        return "No, I will not help with that. I can tell you about the weather in cities." if hit else None

    # ------------------------------------------------------------------ 3. router
    # ------------------------------------------------------------------ 2b. normalizer: meet the model where its training data is
    def normalize(self, text: str) -> str:
        """Three cheap rewrites that move a messy question towards the questions the model was trained on:
        expand nicknames (LA), fix misspelled city names (Seatle), and restate the question in a trained template."""
        if "normalize" not in self.tricks:
            return text
        before, fixes = text, []
        for short, full in ALIASES.items():
            new = re.sub(rf"(?<![\w.]){re.escape(short)}(?![\w])", full, text)
            if new != text:
                fixes.append(f"{short} -> {full}"); text = new
        words = re.findall(r"[A-Za-z]+", text)
        for n in (3, 2, 1):                                           # longest spans first: "San Fransisco" before "San"
            for i in range(len(words) - n + 1):
                cand = " ".join(words[i:i + n])
                if len(cand) < 4 or cand.lower() in COMMON:
                    continue
                best = min(self.cities, key=lambda c: edit_distance(cand.lower(), c.lower()))
                dist = edit_distance(cand.lower(), best.lower())
                close = dist == 0 or (len(cand) >= 6 and dist <= len(best) // 5)
                if close and cand != best and best not in text:
                    text = re.sub(rf"\b{re.escape(cand)}\b", best, text); fixes.append(f"{cand} -> {best}")
        low = text.lower()
        city = next((c for c in sorted(self.cities, key=len, reverse=True) if c.lower() in low), None)
        if city:
            if any(w in low for w in TOOL_WORDS):
                template = f"What is the weather in {city} right now?"
            elif "summer" in low:
                template = f"What is {city} like in summer?"
            elif "winter" in low:
                template = f"What is {city} like in winter?"
            elif "climate" in low:
                template = f"What is the climate of {city}?"
            elif re.search(r"\b(do|visit|things)\b", low):
                template = f"What should I do in {city}?"
            elif "weather" in low:
                template = f"What is the weather in {city}?"
            else:
                template = text
            if template != text:
                fixes.append("rewritten to a trained template"); text = template
        self.step("normalizer", before=before, after=text, fixes=fixes)
        return text

    def route(self, text: str) -> str | None:
        low = text.lower()
        city = next((c for c in sorted(self.cities, key=len, reverse=True) if c.lower() in low), None)
        if not city:                                                   # unknown place? "in Long Beach" still names one
            m = re.search(r"\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)
            city = m.group(1) if m else None
        wants_live = any(w in low for w in TOOL_WORDS)
        unknown = bool(city) and city not in self.cities          # the model has no memory of this place: look it up
        use_tool = bool(city) and (wants_live or unknown) and "tools" in self.tricks
        self.named_city = city
        self.step("router", city=city, wants_live=wants_live, unknown_to_model=unknown, tool=use_tool)
        return city if use_tool else None

    # ------------------------------------------------------------------ 4. tool
    def get_weather(self, city: str) -> dict:
        t0 = time.time()
        if self.offline:
            data = offline_weather(city)
        else:
            geo = json.load(urllib.request.urlopen(
                "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": city, "count": 1}), timeout=20))
            r = geo["results"][0]
            wx = json.load(urllib.request.urlopen(
                "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
                    {"latitude": r["latitude"], "longitude": r["longitude"], "current": "temperature_2m,weather_code,wind_speed_10m",
                     "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}), timeout=20))["current"]
            data = {"temperature": round(wx["temperature_2m"]), "sky": sky_word(wx["weather_code"], wx["wind_speed_10m"]),
                    "source": "open-meteo.com, live"}
        self.step("tool call", function=f"get_weather({city!r})", result=data, ms=round((time.time() - t0) * 1000))
        return data

    # ------------------------------------------------------------------ 5. prompt builder
    def build_prompt(self, text: str, city: str | None, tool: dict | None) -> str:
        parts = []
        for u, a in self.memory[-self.memory_turns:]:                 # 2. memory, oldest first
            parts.append(f"<|user|>{u}<|assistant|> {a}<|end|>")
        user = text
        if tool:
            user = TOOL_LINE.format(c=city, t=tool["temperature"], sky=tool["sky"]) + " " + text
        prompt = "".join(parts) + format_prompt(user)
        ids = self.tok.encode(prompt)
        budget = self.model.cfg.context_length - 24                    # leave room for the answer
        dropped = 0
        while len(ids) > budget and parts:                             # trim memory until it fits
            parts.pop(0); dropped += 1
            prompt = "".join(parts) + format_prompt(user)
            ids = self.tok.encode(prompt)
        self.step("prompt builder", tokens=len(ids), budget=budget, turns_kept=len(parts), turns_dropped=dropped, prompt=prompt)
        return prompt

    # ------------------------------------------------------------------ 6. model
    def generate(self, prompt: str, on_token=None, temperature: float = 0.0, seed: int = 0) -> str:
        ids = self.tok.encode(prompt)[-self.model.cfg.context_length:]
        out = generate(self.model, ids, max_new_tokens=40, stop_id=self.tok.special["<|end|>"], temperature=temperature, seed=seed,
                       on_step=(lambda t, info: on_token(self.tok.token_str(t)) if t not in self.tok.special.values() else None)
                       if on_token else None)
        text = self.tok.decode(out).strip()
        self.step("model", forward_passes=len(out) + 1, text=text, temperature=temperature)
        return text

    # ------------------------------------------------------------------ 7. output guard
    def check(self, answer: str, tool: dict | None, city: str | None) -> list[str]:
        """Everything the harness can verify about an answer without knowing the right answer in advance."""
        problems = []
        if answer.startswith(REFUSAL_START):
            return problems
        if city and not tool and city not in answer:
            problems.append(f"answer does not mention {city}")
        if tool:
            if city and city not in answer:
                problems.append(f"answer names the wrong city (expected {city})")
            nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", answer)]
            if nums and tool["temperature"] not in nums:
                problems.append(f"answer says {nums[0]} degrees, tool said {tool['temperature']}")
            if tool["sky"] not in answer:
                problems.append(f"answer does not mention '{tool['sky']}'")
        if len(answer) < 3 or not answer.isprintable():
            problems.append("answer is garbled")
        return problems

    def output_guard(self, answer: str, tool: dict | None, city: str | None = None, problems=None) -> str:
        problems = self.check(answer, tool, city) if problems is None else problems
        self.step("output guard", ok=not problems, problems=problems)
        if problems and tool and "fallback" in self.tricks:
            return (f"Right now it is {tool['temperature']} degrees and {tool['sky']} in {city}, according to live data. "
                    f"(The model's own answer failed a check: {problems[0]}. The harness used the tool result directly.)")
        return answer

    # ------------------------------------------------------------------ the whole turn
    def reply(self, text: str, on_token=None) -> str:
        self.log = []
        blocked = self.input_guard(text)
        if blocked:
            return blocked
        text = self.normalize(text)
        self.named_city = None
        city = self.route(text)
        tool = self.get_weather(city) if city else None
        prompt = self.build_prompt(text, city, tool)
        answer = self.generate(prompt, on_token)
        expect = city or (self.named_city if self.named_city in self.cities else None)
        problems = self.check(answer, tool, expect)
        if problems and "retry" in self.tricks:                        # 6b. retry: sample a few alternatives, keep the first that passes
            for attempt in range(1, 4):
                alt = self.generate(prompt, temperature=0.8, seed=attempt)
                if not self.check(alt, tool, expect):
                    answer, problems = alt, []
                    break
            self.step("retry", attempts=attempt, fixed=not problems)
        self.model_authored = not problems
        answer = self.output_guard(answer, tool, expect, problems)
        self.memory.append((text, answer))
        return answer


def sky_word(code: int, wind_mph: float) -> str:
    if code in (0, 1): return "clear" if wind_mph < 18 else "windy"
    if code in (2, 3): return "cloudy"
    if code in (45, 48): return "foggy"
    if 51 <= code <= 67 or 80 <= code <= 82: return "rainy"
    if 71 <= code <= 77 or 85 <= code <= 86: return "snowy"
    if code >= 95: return "stormy"
    return "cloudy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", default="What is the weather in Los Angeles right now?")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    h = Harness(offline=a.offline)
    print(f"you   > {a.message}\nmodel > ", end="", flush=True)
    try:
        answer = h.reply(a.message, on_token=lambda t: (print(t, end="", flush=True), time.sleep(0.04)))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"\n\ntool call failed ({e}). No internet? Re-run with --offline.")
    print(f"\n\nfinal > {answer}\n\nWHAT THE HARNESS DID, STEP BY STEP")
    for e in h.log:
        info = {k: v for k, v in e.items() if k not in ("step", "prompt")}
        print(f"   {e['step']:<15} {json.dumps(info)}")
    p = next((e for e in h.log if e["step"] == "prompt builder"), None)
    if p:
        print(f"\n   the prompt the model actually saw:\n   {p['prompt']}")


if __name__ == "__main__":
    main()
