<p align="center"><a href="README.md">&larr; all frames</a> &nbsp;&middot;&nbsp; <a href="README.md">all frames</a> &nbsp;&middot;&nbsp; <a href="2-agents.md"><b>Agents &rarr;</b></a></p>

# 1 · What is AI?

<img src="../assets/ai_four_approaches.svg" width="100%" alt="A two by two grid: thinking humanly, thinking rationally, acting humanly, acting rationally, with an example in each; the acting-rationally square is highlighted as the approach this repository follows">

## Four definitions, one choice

People have defined artificial intelligence (AI) in two ways that cross each other: is the goal to **think** or to **act**, and should the machine match **humans** or an ideal of **rationality**? That makes four approaches.

| | like a human | rationally |
|---|---|---|
| **thinking** | *Cognitive modelling.* Build a program whose reasoning steps match people's. Newell and Simon's General Problem Solver was judged on whether its trace looked like a human's. | *Laws of thought.* Encode correct reasoning in logic, from Aristotle's syllogisms onward. Hard because real knowledge is informal and uncertain, and "solvable in principle" is not "solvable in practice". |
| **acting** | *The Turing test* (Alan Turing, 1950). A machine passes if a questioner cannot tell its written answers from a person's. It needs language, knowledge, reasoning and learning. | *Rational agents.* Build something that does the best thing, or the best expected thing when uncertain. **This is the approach AIMA takes, and the one this repository follows.** |

## Why "acting rationally" wins

Engineering flight took off when people stopped copying birds and studied aerodynamics. AI followed the same path. The rational-agent view is **more general** than pure logic (correct reasoning is only one way to act well; pulling your hand from a hot stove is rational without any reasoning at all), and it is **easier to measure** than matching human thought.

That is exactly why this repository keeps **measuring**: the [evaluation harness](../stages/11_harness/#tips-and-tricks-measured) scores the agent on 45 fixed questions instead of asking whether it *seems* smart.

## Where a language model fits

A language model on its own is a strange fit for all four boxes.

- It **acts humanly** in a narrow sense: it produces text that reads like a person's, because it was trained to predict human text (stage 6).
- It does not **think rationally**: nothing in it checks that a conclusion follows from premises. Our model confidently answered about London when asked about Lisbon (stage 8).
- On its own it is not an **agent** at all: it cannot perceive anything or act on anything. It turns text into more text.
- Wrapped in a **harness** that gives it tools, checks and memory (stage 11), the whole system becomes a rational agent that can be measured.

## The fields AI borrows from

AI draws on philosophy (what is a mind, what is reasoning), mathematics (logic, probability, computation), economics (utility, decisions), neuroscience (how brains compute), psychology (how people think), computer engineering (the hardware), control theory (acting on feedback) and linguistics (language). You meet several of them in this repository: probability in sampling (stage 9), calculus in backpropagation (stage 7), linguistics in tokenization (stage 2), neuroscience in the perceptron.


> [!TIP]
> **Test yourself:** the [what is AI flashcards](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#what-is-ai) take about five minutes. They are also readable [right here on GitHub](../flashcards/README.md#what-is-ai).
