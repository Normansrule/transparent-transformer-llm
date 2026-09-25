"""Stage 2 - text -> token ids -> text.
Run:  python stages/02_tokenizer/run.py "any text you like" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

from transparent_transformer import BPETokenizer, paths

tok = BPETokenizer.load(paths.TOKENIZER)
text = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Los Angeles?"

print(f"vocabulary: 256 raw bytes + {len(tok.merges)} learned merges + {len(tok.special)} special tokens = {tok.vocab_size}\n")
print("the first merges the tokenizer learned (most frequent pairs in data/pretrain.txt):")
for i, (a, b) in enumerate(tok.merges[:8]):
    print(f"   {i+1:>2}. {tok.token_str(a)!r:>8} + {tok.token_str(b)!r:<8} -> {tok.token_str(256+i)!r}")
print("the last ones (by now it is gluing whole words together):")
for i in range(len(tok.merges) - 4, len(tok.merges)):
    a, b = tok.merges[i]
    print(f"  {i+1:>3}. {tok.token_str(a)!r:>8} + {tok.token_str(b)!r:<8} -> {tok.token_str(256+i)!r}")

ids = tok.encode(text)
print(f"\nencode({text!r})")
print("   pieces:", [tok.token_str(i) for i in ids])
print("   ids   :", ids)
print(f"   {len(text)} characters -> {len(ids)} tokens")
assert tok.decode(ids) == text
print(f"decode(ids) == original text: True   (tokenization loses nothing)\n")

for word in [" weather", " Reykjavik", " xylophone"]:
    print(f"{word!r:>13} -> {[tok.token_str(i) for i in tok.encode(word)]}")
print("\ncommon words are ONE token. Rare words shatter into pieces. Words never seen still work, byte by byte.")
print("->  python stages/03_embedding/run.py")
