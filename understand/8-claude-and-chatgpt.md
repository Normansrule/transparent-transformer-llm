<p align="center"><a href="7-beyond-the-transformer.md">&larr; Beyond the transformer</a> &nbsp;&middot;&nbsp; <a href="README.md">all frames</a> &nbsp;&middot;&nbsp; <a href="README.md"><b>all frames &rarr;</b></a></p>

# 8 · Claude and ChatGPT, seen from inside a tiny model

> [!NOTE]
> This page was written with the help of Claude, which is made by Anthropic, one of the two companies compared. To stay fair it uses **only what each company has published**, and it gives you [a tool](#measure-them-yourself) to test both on the same questions rather than asking you to take anyone's word for it.

<img src="../assets/three_scales.svg" width="100%" alt="The same recipe at three scales: this repository's model, Claude and ChatGPT, compared stage by stage, with published and unpublished details marked">

## The same recipe, at a different scale

After eleven stages you know the recipe: tokenize, embed, stack transformer blocks, pretrain on text, align, sample, wrap it in a harness. Both assistants follow that broad recipe. What differs is scale, the data, the details that are not published, and above all **how each company writes down what the model should do**.

| stage | this repository | Claude (Anthropic) | ChatGPT (OpenAI) |
|---|---|---|---|
| **tokenizer** | 768-token BPE | published: none in detail | published: tokenizers for some models are open source |
| **architecture** | 2 blocks, 64 wide, 153,344 parameters | not published | not published; open-weight gpt-oss models are mixture-of-experts transformers |
| **pretraining** | 188 thousand characters about weather | large text corpora; details not published | large text corpora; details not published |
| **alignment** | SFT, DPO, and self-grading against a six-line constitution (stage 8d) | reinforcement learning from human and AI feedback, shaped by a published **constitution** | reinforcement learning from human feedback, shaped by a published **Model Spec** |
| **reasoning** | none | models can think at length before answering | reasoning models; since GPT-5, a router chooses how long to think |
| **harness** | guard, normalizer, router, weather tool, retry, output guard | tools, web search, memory, code execution, agents | tools, web search, memory, code execution, agents |

## How each company writes down what the model should do

**Anthropic: Claude's constitution.** Anthropic first published principles for Claude in May 2023. In January 2026 it replaced that list of standalone rules with a much longer document that explains *why*, on the view that a model which understands the reasons generalises better to new situations. It sets four priorities for when they conflict, in order: being broadly safe, being broadly ethical, following Anthropic's guidelines, and being genuinely helpful. It keeps a small set of hard constraints for high-stakes cases, says it is used directly in training (including to generate synthetic data and to rank responses), and is released into the public domain (CC0) so anyone can reuse it.

**OpenAI: the Model Spec.** OpenAI publishes a living document describing how its models should behave, and updates it openly. Its August 2026 revision, for example, added guidance on relationships with teenage users, clarified how to handle false premises, and added a section on being clear about capabilities and limits.

**The research behind both.** OpenAI's InstructGPT paper (2022) described the RLHF recipe: people rank sample answers, a *reward model* learns those preferences, and reinforcement learning pushes the model towards answers it scores highly. Anthropic's Constitutional AI paper (2022) replaced many human labels with AI feedback guided by written principles: the model critiques and revises its own answers, then learns from AI-ranked comparisons. This repository's [stage 8f](../stages/08_alignment/#8f-a-reward-model-and-spending-compute-at-answer-time) builds a miniature reward model and shows it being gamed, and [stage 8d](../stages/08_alignment/#8d-learning-from-its-own-answers) is a miniature of that second idea: six written principles, automatic graders, and training on the model's own best and worst answers.

## What is actually public about the models

As of September 2026, OpenAI's flagship reasoning model in ChatGPT is **GPT-5.6 Sol** (rolled out from July 2026), alongside the GPT-5.5 family. Anthropic's current public models include **Claude Opus 5.5**, **Claude Sonnet 5**, **Claude Haiku 4.5** and **Claude Fable 5.1**. Neither company publishes parameter counts for these models; since GPT-3 in 2020 (175 billion), any figure you see for a frontier model is an outside estimate.

## What stays the same at every scale

Four findings from this tiny model hold for the giants too:

1. **The model only predicts.** The harness decides, acts and checks (frames 2 and 3).
2. **Behaviour comes from alignment, not from more text** (stage 8, and the GPT-2 project's two datasets).
3. **What you measure is what you get.** A preference that punished the refusal sentence taught our model to stop refusing (stage 8); both companies publish long documents precisely because the details of what is rewarded matter.
4. **Fluent is not the same as right.** Ours says "London" when asked about Lisbon; bigger models fail more rarely and more convincingly. That is why tools, checks and evaluations exist.

## Measure them yourself

`tools/compare_assistants.py` asks Claude and ChatGPT the **same 45 questions** as this repository's evaluation harness, with a checker written to be fair to models that were not trained on our weather data, and prints one table next to this model's own score. Your own API keys stay on your machine.

```bash
export ANTHROPIC_API_KEY=...     # from console.anthropic.com
export OPENAI_API_KEY=...        # from platform.openai.com
python tools/compare_assistants.py                                   # default models
python tools/compare_assistants.py --anthropic-model claude-opus-5-5 --openai-model gpt-5.5
```

It costs a few cents per run. Model names change often, so check each provider's current list and pass the name you want.

**Sources.** Anthropic, [Claude's constitution](https://www.anthropic.com/news/claude-new-constitution) (January 2026); OpenAI, [Model Spec](https://model-spec.openai.com) and [model release notes](https://help.openai.com/en/articles/9624314-model-release-notes); Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (2022); Bai et al., [*Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073) (2022).


> [!TIP]
> **Test yourself:** the [claude and chatgpt flashcards](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#claude-and-chatgpt) take about five minutes. They are also readable [right here on GitHub](../flashcards/README.md#claude-and-chatgpt).
