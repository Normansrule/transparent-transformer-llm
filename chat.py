"""Talk to the model.   python chat.py            (aligned lesson model)
                        TT_MODEL=real python chat.py   (the model trained on data you scraped)
                        python chat.py --base     (base model: watch it ramble)"""
import argparse

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import chat

ap = argparse.ArgumentParser()
ap.add_argument("--base", action="store_true", help="use the pretrained-only base model")
ap.add_argument("--temperature", type=float, default=0.7)
args = ap.parse_args()

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.BASE_MODEL if args.base else paths.ALIGNED_MODEL)
print(f"[{paths.MODEL} track] transparent_transformer ({'BASE' if args.base else 'ALIGNED'} model, {model.num_parameters():,} parameters). "
      "Try: What is the weather in Los Angeles?   Ctrl+C to quit.\n")
turn = 0
while True:
    try:
        prompt = input("you   > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if prompt:
        turn += 1
        print(f"model > {chat(model, tok, prompt, temperature=args.temperature, seed=turn)}\n")
