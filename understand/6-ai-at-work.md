<p align="center"><a href="5-minds-and-machines.md">&larr; Minds and machines</a> &nbsp;&middot;&nbsp; <a href="README.md">all frames</a> &nbsp;&middot;&nbsp; <a href="README.md"><b>all frames &rarr;</b></a></p>

# 6 · AI at work

<img src="../assets/adoption.svg" width="100%" alt="Animated bars: 51 percent of Japanese game developers used generative AI in 2025, 85.8 percent in 2026; 63 percent use it daily">

## The number

A survey by the Computer Entertainment Suppliers Association (CESA), presented at the 2026 Tokyo Game Show, found that **85.8%** of responding Japanese game developers use generative AI in their work, up from **51%** the year before. **63%** use it daily and 22.8% occasionally. Respondents came from member companies including Capcom, Sega, Level-5, Konami, Square Enix, Koei Tecmo and Sony. The findings so far come from a preview; the full report is due in December 2026. ([TechSpot report](https://www.techspot.com/news/113892-nearly-86-japanese-game-developers-using-generative-ai.html) · [CESA industry research](https://www.cesa.or.jp/action/industry-research/))

## How they say they use it, and the guardrails

Most respondents said AI output does **not** go straight into their code. They described using it for **debugging, fixing errors and automated testing**, with every use "confirmed, modified, and supervised" by a human, restrictions on which tools are allowed, and **people keeping responsibility** for creative decisions and final quality. CESA's director also cautioned that new technology must not compromise **intellectual property rights**. The benefits cited most were efficiency, shorter development cycles and lower costs.

## Every guardrail has a counterpart in this course

| what studios describe | where the same idea lives |
|---|---|
| a human confirms and supervises every use | the coding agent's **approval policy**: risky tools need a yes (frame 3); the harness decides, not the model |
| AI for debugging and testing, not final code | the evaluation harness: [45 fixed tests](https://Normansrule.github.io/transparent-transformer-llm/harness.html) decide whether a change is an improvement |
| only approved tools, used in approved ways | the tool **registry and schema checks** (frame 3) |
| people own creative decisions and quality | *agent = harness + model*: responsibility sits in the part people write |
| protect intellectual property | training data carries its licence: the field trip records every source in `data_real/SOURCES.md`, and the GPT-2 project uses only permissively licensed code |

## Why this matters to a learner

Using these tools well is now part of the job in many industries. The survey's picture, humans in charge, AI for checking and speeding up, strict limits on what it touches, is the picture this repository teaches from the inside: the model predicts, the harness checks, and measurement decides.


> [!TIP]
> **Test yourself:** the [AI at work flashcards](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#ai-at-work) take about five minutes. They are also readable [right here on GitHub](../flashcards/README.md#ai-at-work).
