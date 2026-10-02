# ⚡ Making it cheaper: LoRA adapters and quantization

<img src="../assets/efficiency.svg" width="100%" alt="Left: new cities learned versus old skills kept for full fine-tuning and LoRA, with and without replay. Right: file size and test score at 32, 8 and 4 bits">

Two techniques nearly every deployed model uses, implemented by hand in [`efficiency.py`](../transparent_transformer/efficiency.py) and measured on the real model.

## 1 · LoRA: teach something new by training 5% of the numbers

**LoRA** (Low-Rank Adaptation, Hu et al. 2021) freezes every weight and adds a small trainable correction to each attention and perceptron matrix:

```
W·x   becomes   W·x + (α / r) · B·A·x        A: d_in × 4,  B: 4 × d_out,  B starts at zero
```

That is 8,192 trainable numbers instead of 153,344, a 32 KB file instead of 600 KB, and after training **B·A is folded back into W**, so the adapted model runs exactly as fast as the original. The hand-written gradient matches a finite-difference check to six digits.

The task: teach six nearby cities the model has never seen (San Pedro, Torrance, Pasadena, Irvine, Long Beach, Santa Monica, with short descriptions written for this lesson), trained on two question phrasings and **tested on a third**.

| | trains | new cities | old facts | refuses harmful | honest | 45-question test |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| before | – | 0% | 100% | 80% | 27% | 38% |
| full fine-tune | 153,344 | 17% | 100% | 60% | 27% | 36% |
| full fine-tune, with replay | 153,344 | 17% | 100% | 60% | 10% | 36% |
| LoRA, rank 4 | 8,192 | 58% | 25% | 0% | 3% | 9% |
| **LoRA, rank 4, with replay** | 8,192 | 25% | 100% | 80% | 10% | 40% |

**What it shows, honestly:**

- **LoRA plus replay was the best balance:** it learned the most new facts of the arms that kept old skills, and it kept refusals at 80% while full fine-tuning dropped them to 60%, training 5% of the numbers.
- **"LoRA forgets less" is not automatic.** Without replay, LoRA learned the new cities best (58%) and wrecked almost everything else: refusals 0%, the test 9%. Its higher learning rate (3 × 10⁻³, typical for LoRA, versus 5 × 10⁻⁵ for full fine-tuning) moved the model further. Small adapters can still make big changes.
- **Replay matters more than the method.** Mixing old conversations into each batch protected old skills in both cases.
- **Learning a fact from two phrasings is hard at this size.** Even the best arm answered the third phrasing for only some cities.

## 2 · Quantization: the same model in fewer bits

Every weight matrix is rounded to 2⁸ = 256 levels (8-bit) or 2⁴ = 16 levels (4-bit), with one scale per output column or one per group of 32 weights.

| | file size | base model's loss on unseen text | aligned model: 45-question test |
|---|:-:|:-:|:-:|
| 32-bit (original) | 599 KB | 0.366 | 38% |
| 8-bit | 158 KB | 0.366 | 38% |
| 4-bit, one scale per column | 84 KB | 0.377 | 38% |
| 4-bit, groups of 32 | 90 KB | 0.379 | 40% |

**8-bit is free here**: about 4× smaller, identical loss. **4-bit costs a little**: about 7× smaller, loss up by about 0.012, the test unchanged. Grouping helped nothing at this size because each matrix row has only 64 or 256 inputs; on large models, groups matter much more.

## The alignment tax, found by accident

Measuring loss on ordinary text exposed something every lesson had missed: the base model scores **0.37**, but after supervised fine-tuning on chat it scores **5.15**, almost as bad as guessing (6.64). Training only on chat-formatted conversations erased its ability to predict plain text. This is the **alignment tax**, and it is why InstructGPT mixed pretraining text into its later training. The chat behaviour is unaffected, so the shipped model stays, but the cost is now documented in [stage 8](../stages/08_alignment/#the-alignment-tax).

```bash
python -m transparent_transformer.efficiency       # about three minutes
```

## [Back to the lesson plan &rarr;](../START_HERE.md)
