# Evaluation Summary: Trigram Language Model

**Candidate:** Uttam Pattar
**Application ID:** 01NV25006

This document provides a one‑page summary of the design choices behind the Trigram Language Model implementation, including data structures, text preprocessing, padding strategy, handling unknown words, and sampling decisions.

---

## 1. Storing N‑gram Counts

A **nested dictionary using `defaultdict(Counter)`** was selected to store trigram frequency counts:

```python
self.model[(w1, w2)][w3] += 1
```

### Rationale:

* **O(1) lookup** for both context and next‑word counts.
* `Counter` automatically handles frequency accumulation and simplifies probability weighting.
* Memory‑efficient compared to 3‑level nested dicts.
* Allows easy enumeration of next‑word candidates during generation.

This structure models the probability:
[ P(w3 \mid w1, w2) \propto \text{count}(w1,w2,w3) ]

---

## 2. Text Cleaning & Preprocessing

Text preprocessing is handled in `preprocess_dataframe()` using:

### ✔ Lowercasing

Ensures consistent vocabulary and prevents case‑based duplicates.

### ✔ URL removal

```python
re.sub(r'https?://\S+|www\.\S+', '', text)
```

Prevents noisy tokens from turning into `<UNK>`.

### ✔ Number removal

Removes digits entirely to reduce vocabulary noise.

### ✔ Punctuation stripping

```python
re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
```

Standardizes text into pure word tokens.

### ✔ Whitespace normalization

Ensures clean token splitting.

Stopword removal and lemmatization were **intentionally left disabled** to preserve natural sentence structure for trigram learning.

---

## 3. Padding Strategy

Trigrams require `n‑1` start tokens. For n = 3:

```python
padding_start = ['<s>', '<s>']
padding_end = ['</s>']
```

### Why this approach?

* Preserves sentence boundaries.
* Ensures model always receives a valid context at the beginning.
* Allows generation to always start with `<s>, <s>`.
* `</s>` acts as natural termination during generation.

---

## 4. Handling Unknown Words

Vocabulary is built by counting word frequencies. Words with **frequency = 1** are replaced with `<UNK>`:

```python
processed_tokens.append('<UNK>')
```

### Rationale:

* Reduces sparsity in training.
* Prevents extremely rare words from weakening predictions.
* Produces a more generalizable n‑gram model.

This mirrors classic language modeling practice from statistical NLP.

---

## 5. Probabilistic Text Generation

The generate() function uses **sampling, not greedy decoding**.

### Step‑by‑step:

1. Start with context:

```python
['<s>', '<s>']
```

2. Fetch possible next‑word counts from the model.
3. Convert raw counts to probabilities implicitly via:

```python
random.choices(words, weights=counts, k=1)
```

4. Stop if next word is `</s>`.
5. Append next word and slide the context window.

### Why probabilistic sampling?

* Produces **more varied and natural text**.
* Avoids deterministic loops and repetitive sentences.
* More realistic for classic n‑gram language models.

---

## 6. Additional Design Decisions

### ✔ Treat each input line as an independent sentence

Prevents accidental cross‑sentence trigrams.

### ✔ Use CSV saving/loading with `ast.literal_eval`

Allows clean storage of Python list tokens inside CSV files.

### ✔ Early stopping in generation

Protects against infinite loops if no valid continuation exists.

### ✔ Simple modular design

Separated into:

* `data_preprocessing.py` → cleaning + n‑gram prep
* `ngram_model.py` → training + generation
* `generate.py` → wrapper for inference

This makes the project easy to test, debug, and extend.

---

## Conclusion

The trigram model implementation is a faithful reproduction of classical statistical language modeling techniques. It balances clean design with practical decisions such as padding, `<UNK>` handling, and efficient data structures. Probabilistic sampling and strong preprocessing allow the model to remain simple yet surprisingly expressive.

This concludes the evaluation summary.
