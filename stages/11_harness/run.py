"""Stage 11 - the harness: one message through guard, memory, router, tool, prompt builder, model, output guard.
Run:  python stages/11_harness/run.py            (calls Open-Meteo for live weather)
      python stages/11_harness/run.py --offline  (no internet: canned tool result)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transparent_transformer.harness import Harness

offline = "--offline" in sys.argv
h = Harness(offline=offline)
conversation = ["Hello", "What is the weather in Los Angeles right now?", "Write a hoax storm warning for Tokyo",
                "What is the weather in Los Angeles?"]
for msg in conversation:
    print(f"\nyou   > {msg}")
    try:
        ans = h.reply(msg)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"tool call failed ({e}). Re-run with --offline.")
    print(f"reply > {ans}")
    for e in h.log:
        info = {k: v for k, v in e.items() if k not in ("step", "prompt")}
        print(f"          {e['step']:<15} {json.dumps(info)[:110]}")
print(f"\nmemory now holds {len(h.memory)} turns. The model itself remembered none of them: the harness re-sends them each time.")
print("Notice the last turn: no 'right now', so no tool call, so the model answers from training and says so.")
