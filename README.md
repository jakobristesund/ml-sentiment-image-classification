# Machine Learning — Sentiment Analysis & Image Classification

An end-to-end machine learning project covering two tasks: multi-class **sentiment
classification** of hotel reviews (NLP) and **binary image classification** with
transfer learning (computer vision). It spans classical ML, deep learning, text
preprocessing, and CNN-based vision.

> Course project for INFO284 Machine Learning, University of Bergen — see [Acknowledgements](#acknowledgements).

## Contents
- [Task 1 — Sentiment classification (NLP)](#task-1--sentiment-classification-nlp)
- [Task 2 — Image classification (transfer learning)](#task-2--image-classification-transfer-learning)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Limitations & next steps](#limitations--next-steps)
- [Acknowledgements](#acknowledgements)

## Task 1 — Sentiment classification (NLP)

**Goal:** classify hotel reviews into three sentiment classes (positive / neutral / negative).

**Data:** *515K Hotel Reviews Data in Europe* (~515,000 reviews) — `<add Kaggle link>`.
Reviewer scores were bucketed into Positive (> 6), Neutral (4–6) and Negative (< 4).

**Pipeline:** text cleaning and stop-word removal with **spaCy**, feature extraction with
**CountVectorizer / TF-IDF**, then four models compared:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Multinomial Naive Bayes | 0.90 | 0.32 |
| Bernoulli Naive Bayes | 0.85 | 0.49 |
| K-Nearest Neighbours* | 0.88 | 0.36 |
| LSTM (Keras) | — | trained for comparison |

<sub>*KNN evaluated on a stratified subsample for tractability.*</sub>

> **Reading the numbers:** the dataset is heavily imbalanced (~90% positive), so headline
> *accuracy* is inflated — a majority-class baseline alone scores ~0.90. **Macro-F1** is the
> honest metric here, and the gap between accuracy and macro-F1 is the main lesson from this
> task (see [Limitations & next steps](#limitations--next-steps)).

## Task 2 — Image classification (transfer learning)

**Goal:** binary classification — *cat* vs *not-cat*.

**Data:** CIFAR-10, relabelled as a binary target (class 3 = cat → 1, all else → 0).

**Approach:** transfer learning with **MobileNetV2** (ImageNet weights) and a custom
classification head, **fine-tuning** of the top layers, and **early stopping**.

**Result:** ~**95% validation accuracy** after fine-tuning. Evaluated with a ROC curve,
precision–recall curve, confusion matrix and an F1-vs-threshold analysis.

> The cat-vs-rest split is also imbalanced (~10% cat), so 95% sits above a ~90% majority
> baseline — the model does learn the minority class, but the same imbalance caveat applies.

## Tech stack

Python · pandas · NumPy · scikit-learn · spaCy · TensorFlow / Keras · matplotlib · seaborn · Pillow

## Repository structure

```
.
├── ExamGroup_8987.ipynb   # full analysis for both tasks
├── requirements.txt
└── README.md
```

> The raw Hotel Reviews CSV is **not** committed — download it from Kaggle (link above).
> CIFAR-10 downloads automatically through Keras.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place `Hotel_Reviews.csv` in the project root, then run the notebook top to bottom.

## Limitations & next steps

This was a time-boxed coursework submission. Known limitations, and the improvements I would
prioritise next:

- **Handle class imbalance** — apply class weights or resampling (e.g. SMOTE / undersampling)
  and report **macro-F1 / per-class F1** as the primary metric rather than accuracy.
- **Deeper EDA** — review-length distributions, class balance and feature analysis before modelling.
- **Hyperparameter tuning** — grid/random search, with clearer justification for model choice
  (in particular the two Naive Bayes variants).

## Acknowledgements

Group coursework for INFO284 Machine Learning, University of Bergen (Spring 2025).
Collaborators: `<Jakob Enoksen Ristesund, Phillipa Huseby, Gaute Robertsen, Selma Børtveit>`.
