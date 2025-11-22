# ML Internship Assignment — Uttam Pattar

**Application ID:** 01NV25006

---

## Overview

This repository contains my submission for the ML Internship assessment. It implements the required Trigram Language Model and an optional Scaled Dot-Product Attention module as a bonus. The project is fully tested, documented, and ready for evaluation.

---

## Contents (High-level)

```
mlassignment/
├── src/                          # Core implementations and training scripts
├── attention_task/               # Optional attention task (bonus)
├── tests/                        # Unit tests for core functionality
├── data/                         # Example/generated corpora
├── README.md                     # (this file)
├── evaluation.md                 # Design decisions and evaluation notes
├── requirements.txt              # Python dependencies
└── SUBMISSION_SUMMARY.md         # Submission overview
```

> See the project structure screenshot for exact layout: `/mnt/data/12518289-8b47-4a5b-873c-8adfc09742a5.png`

---

## What I implemented

**Task 1: Trigram Language Model (Required)**

* Complete trigram language model implemented from scratch
* Probabilistic sampling for generation (not greedy)
* Padding, `<s>` and `</s>` tokens, and `<UNK>` handling for low-frequency words
* Data preprocessing and tokenizer scripts
* Training and generation utilities
* Unit tests covering edge cases and generation behavior

**Task 2: Scaled Dot-Product Attention (Optional, Bonus)**

* Pure NumPy implementation of scaled dot-product attention
* Supports causal and padding masks
* Demonstrations and unit tests to validate correctness

---

## Code Statistics (approx.)

* Total Lines of Code: ~950
* Test Coverage: Core functionality covered by unit tests
* Dependencies: NumPy, pytest (see `requirements.txt`)

---

## How to run (quick start)

### 1. Clone the repo

```bash
git clone https://github.com/MarthaKivelson/ml-intern-assessment.git
cd ml-intern-assessment
```

*(If you prefer to clone your fork instead, replace the URL above with your fork URL.)*

### 2. Create & activate a conda environment

```bash
conda create -n mlassignment python=3.10 -y
conda activate mlassignment
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick verification (tests)

### Task 1 (Trigram model)

```bash
pytest tests/test_ngram.py -v
```

### Task 2 (Attention — demos/tests)

```bash
python3 attention_task/test_attention.py
```

---

## Full demo (optional)
 Run attention demos

```bash
python3 attention_implementation/attention_task.py
```

---

## Design Philosophy

* **Correctness first**: Implement formulas and probabilistic sampling precisely.
* **Clean code**: Modular, well-documented, and easy to follow.
* **Education**: Demonstrations and tests designed to make the implementation easy to understand.

---

## Notes for reviewers

* This submission is ready for review. If you want me to highlight or expand any particular part (e.g., training details, hyperparameters, or evaluation results), tell me which section and I will update the repo.

---

## Contact

Candidate: **Uttam Pattar**

Application ID: **01NV25006**

Thank you for reviewing my submission. I'm ready for further instructions regarding key features to include or emphasize.
