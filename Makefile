PY ?= python3

.PHONY: scrape corpus report real web notebooks homework classroom help setup data tokenizer pretrain sft dpo train trace visuals docs tour chat test all clean

help:  ## list every command
	@grep -E "^[a-z]+:.*##" Makefile | sed "s/:.*## /\t/" | expand -t 12

setup:  ## install the two dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## write the three datasets (lesson track: made-up data; real track: from your scrape)
	@if [ "$$TT_MODEL" = "real" ]; then $(PY) data_real/build_corpus.py; else $(PY) data/make_corpus.py; fi

tokenizer:  ## stage 2: learn the Byte Pair Encoding (BPE) merges
	$(PY) -m transparent_transformer.pretrain --tokenizer-only

pretrain:  ## stage 6: train the base model (about 3 minutes)
	$(PY) -m transparent_transformer.pretrain

sft:  ## stage 8a: Supervised Fine-Tuning (SFT)
	$(PY) -m transparent_transformer.alignment sft

dpo:  ## stage 8b: Direct Preference Optimization (DPO)
	$(PY) -m transparent_transformer.alignment dpo

train: data tokenizer pretrain sft dpo  ## everything from scratch (about 4 minutes)

scrape:  ## FIELD TRIP: download real climate numbers and Wikipedia text (slow on purpose, resumable)
	$(PY) -m scrape.run

corpus:  ## turn scraped facts into training files
	$(PY) data_real/build_corpus.py

report:  ## grade the real-data model against the scraped truth
	TT_MODEL=real $(PY) -m transparent_transformer.evaluate

real:  ## the whole field trip after scraping: corpus, training, report card, website export
	$(PY) data_real/build_corpus.py
	TT_MODEL=real $(MAKE) tokenizer pretrain sft dpo
	TT_MODEL=real $(PY) -m transparent_transformer.evaluate
	TT_MODEL=real $(PY) -m transparent_transformer.trace --fast > /dev/null
	TT_MODEL=real $(PY) tools/export_web.py

trace:  ## follow one prompt through all ten stages
	$(PY) -m transparent_transformer.trace

visuals:  ## redraw the animated diagrams from the real numbers
	$(PY) -m transparent_transformer.trace --fast > /dev/null
	$(PY) tools/make_visuals.py

docs: visuals  ## paste real script output into every stage page
	$(PY) tools/refresh_docs.py

web:  ## export the model so the website can run it in the browser
	$(PY) tools/export_web.py

notebooks:  ## rebuild and execute one notebook per lesson (pip install nbformat nbclient ipykernel)
	$(PY) tools/make_notebooks.py

homework:  ## grade classroom/exercises
	$(PY) classroom/check.py

classroom: docs web notebooks  ## refresh EVERYTHING students see, after retraining

tour:  ## run all ten stage demos back to back
	@for d in stages/*/; do echo; echo "=== $$d"; $(PY) $$d/run.py || exit 1; done

chat:  ## talk to the aligned model
	$(PY) chat.py

test:  ## prove the hand-written gradients are right
	$(PY) -m pytest -q

all: train classroom test  ## retrain, redraw, retest

clean:  ## delete trained weights (the next make train rebuilds them)
	rm -f artifacts/*.npz artifacts/*.json
