# Next Word Prediction — N-Grams

This is a first pass at NLP-based next-word prediction, built as groundwork for a larger project: predicting what a user is about to type, similar to Google Colab's inline "ghost text" autocomplete. This repo is just the exploratory notebook — uploaded as-is (including the messy, iterative Colab-style structure) so the rest of the team can see the reasoning and results behind each experiment, not just a final polished script.

There's no packaged library or app here yet — just one notebook, `Next_Word_N_gram.ipynb`, that builds and evaluates a **trigram language model**: given the previous two words, it predicts the most likely next word.

## What's in the notebook

The notebook walks through three increasingly sophisticated attempts at the same idea, each as its own self-contained pass:

1. **Raw frequency trigram model** — pulls text from a single Wikipedia page ("Artificial intelligence"), tokenizes it, counts trigram frequencies with `collections.Counter`, and predicts the next word as whichever word most often followed a given pair in the training text.
2. **Add-One (Laplace) smoothing on a larger corpus** — pulls text from six Wikipedia pages (History, Science, Art, Technology, Literature, Philosophy) to get a more diverse vocabulary, then applies Add-One smoothing so unseen trigrams don't get a probability of zero.
3. **Kneser-Ney smoothing on a still-larger corpus** — expands to nine Wikipedia pages and implements Kneser-Ney smoothing (discounting + continuation probability) by hand, plus a perplexity evaluation on a held-out test split.

Each section loads its own data, builds its own trigram table, defines a `predict_next_word*` function, and tests it on a handful of example word pairs.

## Dependencies

There's no `requirements.txt` in this repo yet — it's a single notebook, so install what it imports directly:

```bash
pip install wikipedia numpy jupyter
```

(`string` and `collections` are part of the Python standard library.)

## Running it

```bash
jupyter notebook Next_Word_N_gram.ipynb
```

Run the cells top to bottom. Each of the three "Task" sections is independent in logic but reuses variable names (`words`, `trigrams`, `trigram_frequencies`) from whichever section ran last — so run a section fully before jumping to the next one, rather than executing cells out of order.

Fetching Wikipedia pages requires an internet connection; the `wikipedia` library will print a warning and skip a page if it hits a disambiguation error or the page isn't found.

## Results so far

- **Section 1** (single page, raw counts): works for bigrams that appeared often in the source text (e.g. "artificial intelligence" → "is"), but returns `None` for any bigram it never saw.
- **Section 2** (6 pages, Add-One smoothing): vocabulary grew to ~5,000 unique words, but perplexity on a held-out test split was very high (~4,462), and the model degenerated into predicting the same common word ("coloured") almost regardless of input — a classic sign that the corpus is still too small for the vocabulary size.
- **Section 3** (9 pages, Kneser-Ney smoothing): more principled smoothing, but still built on a relatively small, Wikipedia-only corpus.

**Bottom line so far:** the underlying trigram approach works, but corpus size is the main bottleneck — a few Wikipedia articles aren't enough data for a vocabulary this large. This is the main thing to fix before this becomes useful for the bigger autocomplete project.

## Known issues in the current notebook

- **`gutenberg` library install fails.** The notebook tries to pull in Project Gutenberg texts for a bigger, more varied corpus, but installation fails because `gutenberg` depends on `bsddb3`, which doesn't build in the Colab/most modern environments. The notebook falls back to Wikipedia-only data as a workaround — this is the main thing worth revisiting for more training data.
- **Stray leftover cell at the end** (`pip install gutenberg` as a markdown cell) — an artifact of the Colab session, not meant to be run.
- **The "Open in Colab" badge at the top points to the wrong repo** (it links to a `Chess-Engine` repo from copy-pasting the badge). Should be updated to point here, or removed, before sharing further.
- **Perplexity is high across the board**, and qualitative predictions often collapse to one or two dominant words — expected with this much data, but worth flagging so it's not mistaken for a bug.

## How this fits into the bigger project

This notebook is a proof of concept for the modeling piece of a Colab-style ghost-text autocomplete: given whatever the user has typed so far, suggest the most likely continuation. Trigram counting is simple and fast to reason about, which made it a good starting point, but the corpus-size and generalization limits above suggest the team should also evaluate neural approaches (e.g. a small LSTM or transformer-based language model) alongside scaling up the training corpus, before deciding on the final approach for the real product.
