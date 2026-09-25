<!-- NAV -->
<img src="../../assets/pipeline_09.svg" width="100%" alt="Map of the ten stages. You are at stage 9: sampling. A carriage carries the data in from the previous stage.">

<p align="center"><a href="../08_alignment/">&larr; alignment</a> &nbsp;&nbsp;&middot;&nbsp;&nbsp; stage 9 of 10 &nbsp;&nbsp;&middot;&nbsp;&nbsp; <a href="../10_output/"><b>next: output &rarr;</b></a></p>
<!-- /NAV -->

<!-- DO -->
<p align="center">
<a href="https://Normansrule.github.io/transparent-transformer-llm/#9"><img src="https://img.shields.io/badge/🧪%20try%20it%20live-in%20your%20browser-FFB238?style=for-the-badge" alt="🧪 try it live: in your browser"></a>
<a href="../../notebooks/09_sampling.ipynb"><img src="https://img.shields.io/badge/📓%20see%20the%20code%20run-notebook-5CC8FF?style=for-the-badge" alt="📓 see the code run: notebook"></a>
<a href="https://colab.research.google.com/github/Normansrule/transparent-transformer-llm/blob/main/notebooks/09_sampling.ipynb"><img src="https://img.shields.io/badge/▶%20run%20it%20yourself-Colab-C9A7FF?style=for-the-badge" alt="▶ run it yourself: Colab"></a>
<a href="../../classroom/exercises/ex09.py"><img src="https://img.shields.io/badge/✍️%20build%20it-exercise%2009-6FE3B4?style=for-the-badge" alt="✍️ build it: exercise 09"></a>
<a href="https://github.com/Normansrule/transparent-transformer-llm/issues/new?template=ask-the-model.yml"><img src="https://img.shields.io/badge/💬%20ask-the%20model-FF6F61?style=for-the-badge" alt="💬 ask: the model"></a>
</p>
<!-- /DO -->

<!-- PREDICT -->
> [!IMPORTANT]
> **🎯 Predict before you read.** If you ask the same question twice, do you get the same answer? What would make it different?
>
> Hold your answer in your head. You will check it at the bottom of the page.
<!-- /PREDICT -->

# Stage 9 &middot; Sampling

> **The question:** the transformer produced 768 scores. How does that become one word?

<img src="../../assets/stage_09.svg" width="100%" alt="Animation: bar charts of the real top five candidate tokens at several steps of the answer, with the chosen one highlighted in green">

## The idea in one breath

A language model never outputs a word. It outputs a **probability for every token in its vocabulary**. A separate, very small piece of code then rolls a weighted die. That is sampling, and it is the only place randomness enters.

```mermaid
flowchart LR
    L["logits<br/>(768 scores)"] --> T["divide by<br/>temperature"]
    T --> K["top-k: keep the<br/>k best"]
    K --> P["top-p: keep the smallest set<br/>adding up to p"]
    P --> S["softmax:<br/>scores to probabilities"]
    S --> D["roll the<br/>weighted die"]
    D --> TOK["one token id"]
```

| knob | low value | high value |
|---|---|---|
| **temperature** | safe, repetitive; at 0 always the top choice ("greedy") | surprising, creative, more mistakes |
| **top-k** | only a few candidates survive | the long tail is allowed in |
| **top-p** | only the confident core survives | nearly everything is allowed |

## Why the same question can give different answers

The weights did not change. The die roll did. The demo below uses the Supervised Fine-Tuning (SFT) checkpoint because it is genuinely undecided between two answer styles, so you can watch temperature decide which one you get.

## Run it

```bash
python stages/09_sampling/run.py
```

<!-- RUN:09_sampling -->
<details open>
<summary><b>Real output</b> from the model saved in this repository</summary>

```text
the SAME logits, reshaped by temperature. Probability of the top 4 candidates:

   temperature           ' R'        ' I'      ' the'        ' S'
   0.2                  86.9%       13.1%        0.0%        0.0%
   0.7                  63.2%       36.8%        0.0%        0.0%
   1.0                  59.3%       40.6%        0.0%        0.0%
   1.5                  53.9%       41.9%        0.3%        0.2%
   3.0                  13.8%       12.2%        1.0%        0.8%

   low temperature -> the favourite wins almost always.  high -> the underdogs get a chance.

temperature 0.0: five answers to 'What is the weather in Seattle?'
   Right now it is 55 degrees and cloudy in Seattle.
   Right now it is 55 degrees and cloudy in Seattle.
   Right now it is 55 degrees and cloudy in Seattle.
   Right now it is 55 degrees and cloudy in Seattle.
   Right now it is 55 degrees and cloudy in Seattle.

temperature 0.8: five answers to 'What is the weather in Seattle?'
   Right now it is 55 degrees and cloudy in Seattle.
   Right now it is 55 degrees and cloudy in Seattle.
   I cannot see live weather data, but Seattle is usually cloudy and rainy.
   I cannot see live weather data, but Seattle is usually cloudy and rainy.
   Right now it is 55 degrees and cloudy in Seattle.

temperature 2.5: five answers to 'What is the weather in Seattle?'
   Ri�1 mild mea Madrid is snow 9 Tokyo, butDondon.
   R ground.
   I cannot see city W Honolulu isB� Ralkes from usuallyell beach is about and cloudy.
   � I cannot see In usualket.ec normal What clim theregen I cannot seever tiny mild andly. The What?hen Isonolulu Honolulu is New York isch.
   stals R seeid windy and clear.

->  python stages/10_output/run.py
```

</details>
<!-- /RUN -->

## Read the code

[`transparent_transformer/sampling.py`](../../transparent_transformer/sampling.py): `sample_next()` is the die, `generate()` is the loop around it.

<!-- QUIZ -->
## Check yourself

*Your prediction from the top of the page: was it right? Now this one.*

**With temperature set to 0, asking the same question twice gives...** &nbsp; Click an answer.

<details><summary>A. The same answer: it always takes the top token</summary>

> ✅ **Yes.** Randomness only enters at the sampling step. Temperature 0 removes it. Try it with the slider above.

</details>

<details><summary>B. A different answer each time</summary>

> ❌ Not quite. Close this and try another one.

</details>

<details><summary>C. An error</summary>

> ❌ Not quite. Close this and try another one.

</details>

**Your turn:** open [`classroom/exercises/ex09.py`](../../classroom/exercises/ex09.py), press the pencil icon to edit it right on GitHub (in your fork), fill in the function and commit; or run `python classroom/check.py` locally. [How the exercises work.](../../START_HERE.md#the-exercises-three-ways)
<!-- /QUIZ -->

---

### The hand-off

We have **one** token. The answer needs about fifteen. So the token is glued onto the end of the input and the *entire* model, stages 3 to 5, runs again. And again.

## [Continue to stage 10: Output &rarr;](../10_output/)
