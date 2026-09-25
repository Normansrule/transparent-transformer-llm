# Glossary

Plain-language definitions, in the order you meet the words. Each links to the lesson that explains it.

| word | means | lesson |
|---|---|---|
| **byte** | a whole number from 0 to 255; computers store text as a row of them | [1](stages/01_input/) |
| **token** | a piece of text (often a word or part of a word) that has its own id number | [2](stages/02_tokenizer/) |
| **tokenizer** | the program that cuts text into tokens and back | [2](stages/02_tokenizer/) |
| **Byte Pair Encoding (BPE)** | a way to build a vocabulary: repeatedly glue together the most common pair of neighbours | [2](stages/02_tokenizer/) |
| **vocabulary** | the full list of tokens a model knows; 768 here, over 100,000 in large models | [2](stages/02_tokenizer/) |
| **special token** | a token that is a signal rather than text, such as `<\|user\|>` or `<\|end\|>` | [2](stages/02_tokenizer/) |
| **vector** | a list of numbers; here every token becomes a vector of 64 | [3](stages/03_embedding/) |
| **embedding** | swapping a token id for its vector, by looking up a row in a learned table | [3](stages/03_embedding/) |
| **parameter / weight** | one learned number; this model has 153,344 of them | [3](stages/03_embedding/) |
| **residual stream** | the shared channel of token vectors that flows through the transformer; each block adds to it | [4](stages/04_transformer/) |
| **block / layer** | one repeated unit of the transformer: attention followed by a Multi-Layer Perceptron (MLP) | [4](stages/04_transformer/) |
| **Multi-Layer Perceptron (MLP)** | expand, switch on and off, shrink: the part that processes each token alone and stores much of what the model knows | [4](stages/04_transformer/) |
| **logits** | the raw scores the model gives to every possible next token, before they become probabilities | [4](stages/04_transformer/) |
| **attention** | how a token reads from earlier tokens: compare its query with their keys, blend their values | [5](stages/05_attention_closeup/) |
| **query, key, value** | three vectors made from each token: what I look for, what I contain, what I hand over | [5](stages/05_attention_closeup/) |
| **causal mask** | the rule that a token may only look at tokens before it | [5](stages/05_attention_closeup/) |
| **head** | one copy of attention; several run side by side and learn different jobs | [5](stages/05_attention_closeup/) |
| **pretraining** | learning by predicting the next token in real text; where knowledge comes from | [6](stages/06_pretraining/) |
| **loss** | one number saying how wrong a prediction was; training pushes it down | [6](stages/06_pretraining/) |
| **base model** | a model after pretraining only: it continues text, it does not answer | [6](stages/06_pretraining/) |
| **gradient** | for one weight, how much the loss would change if that weight grew a little | [7](stages/07_backpropagation/) |
| **backpropagation** | the method that computes every gradient in one backward pass, using the chain rule | [7](stages/07_backpropagation/) |
| **learning rate** | how big a step each weight takes against its gradient | [7](stages/07_backpropagation/) |
| **alignment** | further training that changes what a model does, not what it knows | [8](stages/08_alignment/) |
| **Supervised Fine-Tuning (SFT)** | training on example conversations, grading only the answers | [8](stages/08_alignment/) |
| **Direct Preference Optimization (DPO)** | training on pairs of answers, making the preferred one more likely | [8](stages/08_alignment/) |
| **hallucination** | a fluent, confident, wrong answer | [8](stages/08_alignment/) |
| **sampling** | choosing one token from the model's probabilities by rolling a weighted die | [9](stages/09_sampling/) |
| **temperature** | a knob that makes sampling safer (low) or more surprising (high) | [9](stages/09_sampling/) |
| **top-k, top-p** | ways to throw away unlikely tokens before rolling the die | [9](stages/09_sampling/) |
| **autoregressive** | generating one token, appending it, and running the whole model again | [10](stages/10_output/) |
| **tool use** | letting a model ask ordinary software for facts, such as live weather, and read the result | [10](stages/10_output/) |
| **scraping** | collecting data from websites with a program, politely | [field trip](scrape/) |
