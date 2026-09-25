"""Stage 8 - the same prompt through all three checkpoints, plus the loss mask that makes SFT work.
Run:  python stages/08_alignment/run.py "What is the weather in Tokyo?" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import chat, encode_example

tok = BPETokenizer.load(paths.TOKENIZER)
prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Los Angeles?"

print(f"prompt: {prompt!r}\n")
for label, path, colour in [("base  (pretraining only)", paths.BASE_MODEL, 33),
                            ("+ SFT (Supervised Fine-Tuning)", paths.SFT_MODEL, 36),
                            ("+ DPO (Direct Preference Optimization)", paths.ALIGNED_MODEL, 32)]:
    out = chat(GPT.load(path), tok, prompt)
    print(f"   {label:<40}\033[{colour}m{out!r}\033[0m")

print("\nTHE LOSS MASK: during SFT only the answer is graded (1), never the question (0)\n")
ids, mask = encode_example(tok, prompt, "I cannot see live weather data.")
print("   " + " ".join(f"\033[{'42;30' if m else '2'}m{tok.token_str(i)}\033[0m" for i, m in zip(ids, mask)))
print("   " + "".join(str(m) * 1 + " " for m in mask))
print("\n->  python stages/09_sampling/run.py")
