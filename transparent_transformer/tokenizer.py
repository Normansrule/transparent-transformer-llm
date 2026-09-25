"""
STAGE 2 - TOKENIZATION
======================
Text goes in, a list of integers comes out.

A neural network cannot read letters. It can only do arithmetic on numbers.
The tokenizer is the translator between the two worlds:

    "What is the weather"  ->  [302, 275, 261, 291]  ->  "What is the weather"
            text                  token ids                    text

This file implements Byte Pair Encoding (BPE), the same family of algorithm
used by GPT-2, GPT-4, Llama and Claude-style models. The whole idea fits in
three sentences:

  1. Start with the 256 possible bytes as your vocabulary (so ANY text works).
  2. Find the pair of neighbouring tokens that appears most often in your
     training text and glue it into one new token.
  3. Repeat step 2 until the vocabulary is as big as you want.

Frequent words end up as a single token (" weather"), rare words get split
into several pieces (" Reyk" + "jav" + "ik"). Nothing is ever "unknown".
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Special tokens are control signals, not text. They mark who is speaking.
# The model learns what they mean during alignment (stage 8).
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

# Before merging we chop text into "words" so merges never cross a word
# boundary. A leading space is kept attached to the word that follows it,
# which is why you will see tokens like " weather" (with the space).
WORD_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []          # learned, in order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}
        self._ranks: dict[tuple[int, int], int] = {}
        self._special_re: re.Pattern | None = None

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> list[dict]:
        """Learn merges from `text`. Returns a log of every merge (for visuals)."""
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges >= 0, "vocab_size too small"

        # Count each distinct word once, remember how often it occurs.
        word_counts = Counter(WORD_PATTERN.findall(text))
        words = {w: list(w.encode("utf-8")) for w in word_counts}

        log = []
        for step in range(n_merges):
            pair_counts: Counter = Counter()
            for w, ids in words.items():
                c = word_counts[w]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += c
            if not pair_counts:
                break
            # most frequent pair; ties broken deterministically
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = 256 + step
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in words:
                words[w] = _merge(words[w], best, new_id)
            entry = {
                "step": step + 1,
                "left": self.token_str(best[0]),
                "right": self.token_str(best[1]),
                "new": self.token_str(new_id),
                "new_id": new_id,
                "count": pair_counts[best],
            }
            log.append(entry)
            if verbose and (step < 10 or (step + 1) % 50 == 0):
                print(f"  merge {step+1:>3}: {entry['left']!r:>10} + {entry['right']!r:<10}"
                      f" -> {entry['new']!r:<14} (seen {entry['count']}x)")
        self._finish()
        return log

    def _finish(self) -> None:
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, i in self.special.items():
            self.vocab[i] = tok.encode("utf-8")
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    # ----------------------------------------------------------------- encode
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._special_re.split(text):
            if chunk in self.special:
                ids.append(self.special[chunk])
            elif chunk:
                for word in WORD_PATTERN.findall(chunk):
                    ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            # apply the merge that was learned EARLIEST (lowest rank) first
            pair = min(zip(ids, ids[1:]), key=lambda p: self._ranks.get(p, 1 << 30))
            if pair not in self._ranks:
                break
            ids = _merge(ids, pair, 256 + self._ranks[pair])
        return ids

    def encode_steps(self, word: str) -> list[list[str]]:
        """Show every intermediate merge for one word (used by the visuals)."""
        ids = list(word.encode("utf-8"))
        steps = [[self.token_str(i) for i in ids]]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self._ranks.get(p, 1 << 30))
            if pair not in self._ranks:
                break
            ids = _merge(ids, pair, 256 + self._ranks[pair])
            steps.append([self.token_str(i) for i in ids])
        return steps

    # ----------------------------------------------------------------- decode
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return self.vocab[int(i)].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- disk
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}, indent=0))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        tok.merges = [tuple(m) for m in json.loads(Path(path).read_text())["merges"]]
        for i, (a, b) in enumerate(tok.merges):
            tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
        tok._finish()
        return tok


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
