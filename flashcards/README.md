# 📇 Flashcards

<a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html"><img src="../assets/flashcards.svg" width="100%" alt="A flashcard flipping from question to answer"></a>

**142 cards in 12 decks.** Three ways to study them:

| | how | best for |
|---|---|---|
| 🃏 | **[the flip-card page](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html)**: flip, grade yourself, it remembers what you know | real studying, on any device |
| 👇 | **right here**: click a question to reveal its answer | quick review on GitHub |
| 🗂️ | **Anki**: import a file from [`anki/`](anki/) (File → Import) | spaced repetition over weeks |

| deck | cards | study |
|---|:-:|:-:|
| 🔟 [the ten stages](#the-ten-stages) | 15 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#the-ten-stages) |
| 🏋️ [training](#training) | 21 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#training) |
| 🧰 [harness](#harness) | 12 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#harness) |
| 🧠 [perceptron and parameters](#perceptron-and-parameters) | 10 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#perceptron-and-parameters) |
| ❓ [what is AI](#what-is-ai) | 8 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#what-is-ai) |
| 🤖 [agents](#agents) | 14 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#agents) |
| 💻 [coding agents](#coding-agents) | 12 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#coding-agents) |
| 📜 [history](#history) | 12 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#history) |
| 🌌 [minds and machines](#minds-and-machines) | 11 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#minds-and-machines) |
| 🏢 [AI at work](#ai-at-work) | 5 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#ai-at-work) |
| 🚀 [beyond the transformer](#beyond-the-transformer) | 12 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#beyond-the-transformer) |
| ⚖️ [claude and chatgpt](#claude-and-chatgpt) | 10 | [flip](https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#claude-and-chatgpt) |

<a id="the-ten-stages"></a>

## 🔟 the ten stages

<details><summary><b>What does the computer actually receive when you type a prompt?</b></summary>

> A row of bytes: whole numbers from 0 to 255, one per plain-English character. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/01_input/">learn more</a></sub>

</details>

<details><summary><b>What is a token?</b></summary>

> A piece of text (often a word or part of one) with its own id number. The model reads and writes tokens, never letters. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/02_tokenizer/">learn more</a></sub>

</details>

<details><summary><b>How does Byte Pair Encoding (BPE) build its vocabulary?</b></summary>

> Start from the 256 bytes, then repeatedly glue together the most frequent pair of neighbouring tokens in the training text. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/02_tokenizer/">learn more</a></sub>

</details>

<details><summary><b>Why is 'weather' one token but 'xylophone' eight?</b></summary>

> Merges are learned by frequency. Common strings earn their own token; rare ones are spelled from small pieces. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/02_tokenizer/">learn more</a></sub>

</details>

<details><summary><b>What is an embedding, mechanically?</b></summary>

> A table lookup: token id 559 selects row 559 of a learned table, giving a vector of 64 numbers. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/03_embedding/">learn more</a></sub>

</details>

<details><summary><b>Why is a position vector added to each token vector?</b></summary>

> Attention sees all tokens at once, like a bag. Without positions, 'dog bites man' and 'man bites dog' would look the same. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/03_embedding/">learn more</a></sub>

</details>

<details><summary><b>What is the residual stream?</b></summary>

> The shared channel of token vectors flowing through the transformer. Every block reads it and ADDS its result; nothing overwrites it. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/04_transformer/">learn more</a></sub>

</details>

<details><summary><b>What are the two halves of a transformer block?</b></summary>

> Attention (tokens exchange information) and a Multi-Layer Perceptron, MLP (each token is processed on its own). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/04_transformer/">learn more</a></sub>

</details>

<details><summary><b>What are logits?</b></summary>

> Raw scores, one per vocabulary token, for what comes next. Softmax turns them into probabilities. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/04_transformer/">learn more</a></sub>

</details>

<details><summary><b>In attention, what are the query, key and value?</b></summary>

> Query: what this token is looking for. Key: what each token contains. Value: what a token hands over when chosen. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/05_attention_closeup/">learn more</a></sub>

</details>

<details><summary><b>What is the causal mask?</b></summary>

> The rule that a token may only attend to tokens before it, because during generation the future does not exist yet. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/05_attention_closeup/">learn more</a></sub>

</details>

<details><summary><b>What does the temperature knob do?</b></summary>

> Divides the logits before softmax. Low: the favourite nearly always wins. High: unlikely tokens get a chance. Zero: always the top token. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/09_sampling/">learn more</a></sub>

</details>

<details><summary><b>How many forward passes does a 16-token answer take?</b></summary>

> About 17: one per generated token plus the one that produces the end token. Each pass runs the whole model. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/10_output/">learn more</a></sub>

</details>

<details><summary><b>How does generation know when to stop?</b></summary>

> The model predicts a special <|end|> token, which it learned during alignment. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/10_output/">learn more</a></sub>

</details>

<details><summary><b>What is the logit lens?</b></summary>

> Reading a prediction off the residual stream after each layer, not just the last. It shows the answer forming layer by layer. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/04_transformer/">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#the-ten-stages">study this deck with flip cards →</a></p>

<a id="training"></a>

## 🏋️ training

<details><summary><b>What is pretraining?</b></summary>

> Learning by predicting the next token in real text. No labels are needed: the text is its own answer key. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/06_pretraining/">learn more</a></sub>

</details>

<details><summary><b>What is the loss?</b></summary>

> One number saying how wrong a prediction was: minus the log of the probability given to the correct token. Training pushes it down. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/06_pretraining/">learn more</a></sub>

</details>

<details><summary><b>What loss does pure guessing over 768 tokens give?</b></summary>

> ln(768) ≈ 6.64. Training starts there. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/06_pretraining/">learn more</a></sub>

</details>

<details><summary><b>What is overfitting?</b></summary>

> Training loss keeps falling while loss on unseen text rises: the model memorises instead of learning. Here, 3,000 steps overfit; 1,500 did not. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/06_pretraining/">learn more</a></sub>

</details>

<details><summary><b>What is a gradient?</b></summary>

> For one weight: how much the loss would change if that weight grew a little. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/07_backpropagation/">learn more</a></sub>

</details>

<details><summary><b>Why use backpropagation instead of wiggling each weight?</b></summary>

> Both give the same numbers, but one backward pass gives every gradient at once; wiggling needs two forward passes per weight. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/07_backpropagation/">learn more</a></sub>

</details>

<details><summary><b>What is the update rule?</b></summary>

> w = w − learning_rate × dLoss/dw, for every weight, every step. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/07_backpropagation/">learn more</a></sub>

</details>

<details><summary><b>What is a base model?</b></summary>

> A model after pretraining only. It continues documents; it does not answer questions. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is Supervised Fine-Tuning (SFT)?</b></summary>

> Training on example conversations, with the loss counted only on the answer tokens. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is the loss mask for?</b></summary>

> It grades the reply and ignores the prompt, so the model learns to write answers, not to write questions. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is Direct Preference Optimization (DPO)?</b></summary>

> Training on pairs of answers so the preferred one becomes more likely than it was, relative to a frozen reference copy. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What are the three H's of alignment?</b></summary>

> Helpful, honest, harmless. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is over-refusal?</b></summary>

> Refusing safe questions that merely sound alarming. A safety model that refuses everything is useless. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What changes between the base model and the aligned model?</b></summary>

> Only the weight values. The architecture, tokenizer and code are identical. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is distillation?</b></summary>

> Training a student model to reproduce the verified outputs of a stronger teacher system. Here the teacher was the harness plus model, and the student the bare model. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is catastrophic forgetting, and how does replay help?</b></summary>

> Fine-tuning on new data can erase old skills. Replay mixes old training examples into every batch, but only protects what it replays. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>Why did distillation wash out honesty, and what fixed it?</b></summary>

> Honesty had been sharpened by preference training, which the replayed data did not contain (30% → 7%). Re-running preference training afterwards restored it to 27%. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is a reward model?</b></summary>

> A scorer that reads a prompt and an answer and returns one number, trained on chosen-versus-rejected pairs. The core of RLHF. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is the Bradley–Terry loss?</b></summary>

> −log sigmoid(reward(chosen) − reward(rejected)): it pushes the preferred answer's score above the other's. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is best-of-N?</b></summary>

> Sample N answers and keep the one the reward model scores highest: spending compute at answer time. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>What is reward hacking?</b></summary>

> Optimising against a reward model finds its blind spots. Here a model more accurate on climate pairs chose worse answers overall (refusals 80% → 73%). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#training">study this deck with flip cards →</a></p>

<a id="harness"></a>

## 🧰 harness

<details><summary><b>Agent = ? + ?</b></summary>

> Harness + model. The harness is ordinary code; the model only predicts the next token. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>Where does a chat assistant's memory of your last message live?</b></summary>

> In the harness, which re-sends earlier turns inside the next prompt. The weights never change while you chat. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>What does the normalizer do?</b></summary>

> Rewrites messy input into forms the model was trained on: LA → Los Angeles, Seatle → Seattle, then a trained question template. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>Why did retry add nothing in this repository?</b></summary>

> Sampling again only fixes random errors. This model's mistakes are systematic, so every retry repeats them. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>What is an evaluation harness?</b></summary>

> A fixed set of test questions with automatic checkers, re-run after every change to prove whether it helped. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>From what to what did the harness tricks raise the score?</b></summary>

> 38% to 91% on 45 questions, with the same weights. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>Why does the harness call the weather tool for a city the model never learned?</b></summary>

> The model has nothing to answer from, so it would guess. Looking it up is the only honest option. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>What does the output guard do when a check fails?</b></summary>

> Rebuilds the answer from the tool result, so a wrong city or number never reaches the user. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>What is an induction head?</b></summary>

> An attention pattern that finds an earlier copy of the current token and predicts what followed it: how models copy from their prompt. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>Why train on random invented city names?</b></summary>

> So memorising is impossible and the only way to win is a general copy skill. It raised unseen-name copying from 0 of 10 to 4 of 10. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>What is the data-mixture trade-off?</b></summary>

> Adding lots of one kind of data can crowd out another: more tool-copying data lowered safety from 87% to 60%. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<details><summary><b>Why can't a single guard stop all harmful requests?</b></summary>

> Rephrasings slip past fixed word lists. Safety works in layers: guard, trained refusals, output checks. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/11_harness/">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#harness">study this deck with flip cards →</a></p>

<a id="perceptron-and-parameters"></a>

## 🧠 perceptron and parameters

<details><summary><b>What does one perceptron compute?</b></summary>

> σ(w·a + b): a weighted sum of its inputs plus a bias, squashed by a switch such as the sigmoid. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/perceptron/">learn more</a></sub>

</details>

<details><summary><b>What is the sigmoid?</b></summary>

> σ(z) = 1 / (1 + e^−z). It maps any number into the range 0 to 1. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/perceptron/">learn more</a></sub>

</details>

<details><summary><b>How many parameters does a 784-16-16-10 network have?</b></summary>

> 13,002: 12,560 + 272 + 170 weights and biases. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/perceptron/">learn more</a></sub>

</details>

<details><summary><b>What does a first-layer neuron's weight picture show?</b></summary>

> Its 784 weights laid out as an image. They look messy, not like neat strokes: the network finds its own features. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/perceptron/">learn more</a></sub>

</details>

<details><summary><b>Where is the perceptron inside a transformer?</b></summary>

> The MLP in every block: 64 → 256 → 64 here, with GELU instead of the sigmoid. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/perceptron/">learn more</a></sub>

</details>

<details><summary><b>What is weight tying?</b></summary>

> Reusing the embedding table as the output layer. It is why GPT-2 small is 124 million parameters, not 163 million. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Doubling the vector width does what to a block's parameters?</b></summary>

> Roughly quadruples them, because every matrix is width × width. Doubling the number of blocks only doubles them. &nbsp;<sub><a href="https://Normansrule.github.io/transparent-transformer-llm/parameters.html">learn more</a></sub>

</details>

<details><summary><b>Do more attention heads add parameters?</b></summary>

> No. Heads split the same Q, K and V matrices into slices. &nbsp;<sub><a href="https://Normansrule.github.io/transparent-transformer-llm/parameters.html">learn more</a></sub>

</details>

<details><summary><b>What is quantization?</b></summary>

> Storing each parameter in fewer bits (16, 8, 4) so a model needs less memory. &nbsp;<sub><a href="https://Normansrule.github.io/transparent-transformer-llm/parameters.html">learn more</a></sub>

</details>

<details><summary><b>Roughly how much compute does training take?</b></summary>

> About 6 × parameters × training tokens operations. &nbsp;<sub><a href="https://Normansrule.github.io/transparent-transformer-llm/parameters.html">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#perceptron-and-parameters">study this deck with flip cards →</a></p>

<a id="what-is-ai"></a>

## ❓ what is AI

<details><summary><b>What two questions give the four approaches to AI?</b></summary>

> Think or act? Match humans or match an ideal of rationality? &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>What is the Turing test?</b></summary>

> A machine passes if a questioner, reading written answers, cannot tell it from a person (Turing, 1950). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>What does passing the Turing test require?</b></summary>

> Natural language processing, knowledge representation, automated reasoning and machine learning. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>What is the 'laws of thought' approach?</b></summary>

> Encode correct reasoning in formal logic, from Aristotle's syllogisms to 20th-century logic programs. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>Two obstacles to the logic approach?</b></summary>

> Real knowledge is informal and uncertain; and solvable in principle is not solvable in practice. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>Which approach does AIMA follow?</b></summary>

> Acting rationally: build agents that do the best expected thing. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>Why is the rational-agent approach preferred?</b></summary>

> It is more general than pure logic and easier to measure scientifically than human-likeness. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<details><summary><b>What does the airplane analogy teach?</b></summary>

> Flight succeeded when engineers stopped copying birds and studied aerodynamics. AI need not copy humans. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/1-what-is-ai.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#what-is-ai">study this deck with flip cards →</a></p>

<a id="agents"></a>

## 🤖 agents

<details><summary><b>What is an agent?</b></summary>

> Anything that perceives its environment through sensors and acts on it through actuators. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Percept versus percept sequence?</b></summary>

> A percept is what is sensed at one instant; the percept sequence is everything ever sensed. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Agent function versus agent program?</b></summary>

> The function is the abstract mapping from percept sequences to actions; the program is the code that implements it. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Agent = ? + ? (AIMA)</b></summary>

> Architecture + program. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>What makes an agent rational?</b></summary>

> It selects the action expected to maximise its performance measure, given its percepts and prior knowledge. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Rationality depends on which four things?</b></summary>

> The performance measure, prior knowledge, available actions, and the percept sequence so far. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Why should performance be measured on environment states, not the agent's opinion?</b></summary>

> Otherwise an agent could score perfectly by deluding itself. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>What goes wrong if a vacuum robot is rewarded for dirt collected?</b></summary>

> It can dump dirt and collect it again. Measure what you want (a clean floor), not how you think it should behave. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>What does PEAS stand for?</b></summary>

> Performance measure, Environment, Actuators, Sensors. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Fully versus partially observable?</b></summary>

> Fully: sensors reveal the whole relevant state. Partially: something is hidden, like live weather for a language model. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Episodic versus sequential?</b></summary>

> Episodic: each decision stands alone. Sequential: current decisions affect later ones, as in a coding agent. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>The five kinds of agent program?</b></summary>

> Simple reflex, model-based reflex, goal-based, utility-based, and learning agents. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Simple reflex agent: when does it work?</b></summary>

> Only when the right action can be chosen from the current percept alone, in a fully observable environment. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<details><summary><b>Atomic, factored and structured representations?</b></summary>

> A state as an unanalysable name; as a fixed set of variables; as objects with relations. Ordered by expressiveness. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/2-agents.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#agents">study this deck with flip cards →</a></p>

<a id="coding-agents"></a>

## 💻 coding agents

<details><summary><b>What does the model do in a coding agent?</b></summary>

> One thing: text in, text out. It reads no files, runs nothing and remembers nothing between requests. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Raschka's six components of a coding agent?</b></summary>

> Live repo context; prompt shape and cache reuse; tool access; minimising context bloat; structured session memory; bounded delegation. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Why keep a fixed prompt prefix?</b></summary>

> Identical leading bytes on every request let a server reuse its cached work. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>What is the cost of taking the workspace snapshot once at start-up?</b></summary>

> It goes stale: files the agent writes do not appear in its status until it restarts. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Who decides whether a tool call runs?</b></summary>

> The harness: it checks the tool name, the arguments and the file path, and asks for approval if the tool is risky. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Why does a base GPT-2 answer an agent prompt with prose?</b></summary>

> It was never trained on the response format; it continues documents. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>What is the Ollama-compatible trick in the GPT-2 project?</b></summary>

> The server answers the same API as a popular model runner, so the agent works unchanged against either. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Two datasets in the GPT-2 project, and what each tests?</b></summary>

> Instruction pairs (loss on the reply only) test 'it needs the format'; raw code (loss on every token) tests 'it needs more code'. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>What does 'correct by construction' mean for training targets?</b></summary>

> The script builds the repository first, so it knows the right tool call for any question about it. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>Why is HTTP's statelessness important for agents?</b></summary>

> The server keeps nothing between requests, so the harness must re-send the whole transcript every turn. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>GPT-2's context window?</b></summary>

> 1,024 tokens. A lean agent prompt uses about half before the question is asked. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<details><summary><b>How is a delegated sub-agent kept safe?</b></summary>

> One level deep, read-only, and it returns a single string. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/3-coding-agents.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#coding-agents">study this deck with flip cards →</a></p>

<a id="history"></a>

## 📜 history

<details><summary><b>1943?</b></summary>

> McCulloch and Pitts: the artificial neuron, a weighted sum and a threshold. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1950?</b></summary>

> Turing's paper asking 'Can machines think?' and proposing the imitation game. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1956?</b></summary>

> The Dartmouth workshop names the field artificial intelligence. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1958?</b></summary>

> Rosenblatt's perceptron learns weights from examples. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1969?</b></summary>

> Minsky and Papert show the limits of single-layer perceptrons; neural-network research cools. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1971?</b></summary>

> The Intel 4004, the first commercial microprocessor, designed under Federico Faggin. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1986?</b></summary>

> Rumelhart, Hinton and Williams popularise backpropagation. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>1995?</b></summary>

> Russell and Norvig publish Artificial Intelligence: A Modern Approach. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>2012?</b></summary>

> AlexNet wins ImageNet; deep learning takes off, powered by GPUs. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>2017?</b></summary>

> Attention Is All You Need introduces the transformer. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>2019 and 2020?</b></summary>

> GPT-2 (up to 1.5 billion parameters) and GPT-3 (175 billion). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<details><summary><b>2022?</b></summary>

> ChatGPT brings aligned chat assistants to the public. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/4-history.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#history">study this deck with flip cards →</a></p>

<a id="minds-and-machines"></a>

## 🌌 minds and machines

<details><summary><b>Orch OR in one sentence?</b></summary>

> Consciousness arises from orchestrated quantum state reductions in microtubules inside neurons (Penrose and Hameroff). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>What sets the timing of objective reduction in Orch OR?</b></summary>

> A gravity-related threshold, roughly τ ≈ ħ / E_G: bigger mass separation, sooner collapse. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Main physics objection to Orch OR?</b></summary>

> The warm, wet brain should destroy quantum coherence in about 10⁻¹³ to 10⁻²⁰ seconds (Tegmark, 2000). Defenders dispute the estimate. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>What evidence do Orch OR's authors cite?</b></summary>

> Gamma-band EEG as the best correlate of consciousness, and anaesthetics appearing to act on microtubules. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Integrated Information Theory in one sentence?</b></summary>

> Consciousness is integrated information, Φ: how much a system's causal structure is more than its parts (Tononi; Koch). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>What does IIT say about a feedforward network?</b></summary>

> Φ = 0, so not conscious, however clever its outputs. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Main objections to IIT?</b></summary>

> Φ is intractable to compute; it rates some simple logic-gate grids as highly conscious; critics call it untestable, which supporters reject. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Faggin's position?</b></summary>

> Consciousness and free will are fundamental, tied to quantum information that cannot be copied, so classical computers cannot be conscious. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>What do computational views (such as Global Workspace Theory) say about AI?</b></summary>

> Consciousness depends on organisation, not substrate, so an AI with the right architecture could in principle be conscious. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Brain capacity: synapse count versus microtubule automata?</b></summary>

> About 4 × 10¹⁵ bits per second (Moravec) versus 10²³ to 10²⁵ (Rasmussen, Hameroff and colleagues, 1990). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<details><summary><b>Why did the 1990 microtubule paper question neural networks as brain models?</b></summary>

> Real neurons are more like computers than switches, and backpropagation has no clear biological counterpart. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/5-minds-and-machines.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#minds-and-machines">study this deck with flip cards →</a></p>

<a id="ai-at-work"></a>

## 🏢 AI at work

<details><summary><b>What share of Japanese game developers used generative AI in the 2026 CESA survey?</b></summary>

> 85.8%, up from 51% the year before; 63% daily. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/6-ai-at-work.md">learn more</a></sub>

</details>

<details><summary><b>What did respondents say they mostly use AI for?</b></summary>

> Debugging, fixing errors and automated testing, rather than putting AI output directly into code. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/6-ai-at-work.md">learn more</a></sub>

</details>

<details><summary><b>What guardrails did studios describe?</b></summary>

> Humans confirm, modify and supervise every use; approved tools only; people own creative decisions and final quality. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/6-ai-at-work.md">learn more</a></sub>

</details>

<details><summary><b>Benefits cited most often?</b></summary>

> Efficiency and productivity, shorter development cycles, lower costs. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/6-ai-at-work.md">learn more</a></sub>

</details>

<details><summary><b>Which caution did CESA's director raise?</b></summary>

> New technology must not compromise intellectual property rights. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/6-ai-at-work.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#ai-at-work">study this deck with flip cards →</a></p>

<a id="beyond-the-transformer"></a>

## 🚀 beyond the transformer

<details><summary><b>What two costs of the transformer do most new architectures attack?</b></summary>

> Attention's cost grows with the square of the context length, and generation produces only one token per full pass. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>What is a mixture of experts?</b></summary>

> Many expert MLPs per block plus a router that sends each token to only a few, so total parameters grow while compute per token stays small. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>Total versus active parameters?</b></summary>

> Total: every weight the model holds. Active: the weights one token actually uses. Jamba: 52 billion total, 12 billion active. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>How does a state space model process a sequence?</b></summary>

> It squeezes the history into a fixed-size state, h = A·h + B·x, and reads out y = C·h. Each token costs the same however long the history. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>What did Mamba add to state space models?</b></summary>

> Input-dependent (selective) dynamics, so the model chooses what to remember and what to forget. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>Why do shipping models use SSM-attention hybrids?</b></summary>

> A fixed-size state cannot recall everything exactly; a few attention layers restore precise lookup while SSM layers keep long contexts cheap. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>How does a diffusion language model generate text?</b></summary>

> It starts with every position masked, predicts all of them in parallel, keeps the confident ones, and refines over a few passes. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>Name three diffusion language models.</b></summary>

> LLaDA (research), Mercury from Inception Labs (first commercial-scale), and Google's experimental Gemini Diffusion. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>What makes a reasoning model different?</b></summary>

> It writes intermediate thinking before answering and is trained with reinforcement learning on problems whose answers can be checked. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>What is a Kolmogorov–Arnold Network (KAN)?</b></summary>

> A network with learnable functions on its connections instead of fixed activation functions on its neurons. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>Where does quantum machine learning stand?</b></summary>

> An active research area with variational quantum circuits; no practical advantage for large-scale learning has been shown yet. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<details><summary><b>Split this dense model's 256 neurons into 8 experts and keep 2: how much output survives?</b></summary>

> About 35% grouped in order, about 50% grouped by which neurons fire together. Trained mixture-of-experts models specialise much more. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/7-beyond-the-transformer.md">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#beyond-the-transformer">study this deck with flip cards →</a></p>

<a id="claude-and-chatgpt"></a>

## ⚖️ claude and chatgpt

<details><summary><b>What does Anthropic publish to describe how Claude should behave?</b></summary>

> A constitution: a long document of values and reasons, released January 2026 into the public domain (CC0) and used directly in training. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>The constitution's four priorities, in order?</b></summary>

> Broadly safe, broadly ethical, compliant with Anthropic's guidelines, genuinely helpful. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>What does OpenAI publish to describe how its models should behave?</b></summary>

> The Model Spec, a living document it updates openly (for example in August 2026). &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>What is RLHF, as in InstructGPT (2022)?</b></summary>

> People rank sample answers, a reward model learns those preferences, and reinforcement learning pushes the model towards high-reward answers. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>What is Constitutional AI (2022)?</b></summary>

> Written principles guide AI feedback that critiques, revises and ranks the model's own answers, reducing how many human labels are needed. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>How is stage 8d a miniature of Constitutional AI?</b></summary>

> Six written principles with automatic graders score the model's own sampled answers; it trains on its best versus worst ones. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>Do Anthropic or OpenAI publish parameter counts for current models?</b></summary>

> No. Since GPT-3 (175 billion, 2020), frontier parameter counts are outside estimates. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>What did GPT-5 (2025) add to ChatGPT's harness?</b></summary>

> A router that decides when to answer quickly and when to spend longer reasoning. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/blob/main/understand/8-claude-and-chatgpt.md">learn more</a></sub>

</details>

<details><summary><b>Why did red-teaming make self-improvement work here?</b></summary>

> On familiar prompts the model rarely broke a principle, so there was nothing to learn. New, harder wordings exposed failures to train on. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<details><summary><b>Which component of InstructGPT's RLHF recipe does stage 8f build?</b></summary>

> The reward model, trained on preference pairs, then used to pick the best of 8 answers. &nbsp;<sub><a href="https://github.com/Normansrule/transparent-transformer-llm/tree/main/stages/08_alignment/">learn more</a></sub>

</details>

<p align="right"><a href="https://Normansrule.github.io/transparent-transformer-llm/flashcards.html#claude-and-chatgpt">study this deck with flip cards →</a></p>
