"""Stage 1 - what a computer actually receives when you type a prompt.
Run:  python stages/01_input/run.py "What is the weather in Los Angeles?" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Los Angeles?"
raw = prompt.encode("utf-8")

print(f"text       : {prompt!r}")
print(f"characters : {len(prompt)}")
print(f"bytes      : {len(raw)}   (UTF-8: plain English letters are 1 byte each, 'é' is 2, an emoji is 4)\n")
print("char  byte  binary")
for ch in prompt[:12]:
    for b in ch.encode("utf-8"):
        print(f" {ch!r:<5}{b:>4}  {b:08b}")
print(" ...\n")
print("That column of numbers is ALL the model will ever get. Next: group them into tokens.")
print("->  python stages/02_tokenizer/run.py")
