# Fake News Detector

**Simple, reproducible pipeline for detecting fake news using classical ML**

---

## Project overview

This repository contains a compact, well-documented pipeline for detecting fake news using standard Natural Language Processing (NLP) steps plus two supervised classifiers: **Multinomial Naive Bayes** (baseline) and **Random Forest** (stronger tree-based model). The work includes exploratory data analysis (EDA), preprocessing, feature extraction, model training, evaluation, and notes for improvement.

The goal is to provide a clear, reproducible baseline implementation suitable for experimentation and quick iteration.

---

## Key highlights (short)

- Performed EDA to understand class balance, common words, and article lengths.
- Preprocessed text (lowercasing, punctuation removal, tokenization, stopword removal, optional lemmatization).
- Extracted features with TF–IDF (unigrams and bigrams).
- Trained and evaluated **Multinomial Naive Bayes** and **Random Forest** classifiers.
- Reported metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## Files & structure

- **data/**  
  - train.csv  
  - test.csv  
- **notebooks/**  
  - eda.ipynb  
- **src/**  
  - preprocess.py – text cleaning & tokenization helpers  
  - features.py – TF-IDF pipeline  
  - train.py – training entrypoint  
  - evaluate.py – evaluation helpers  
- **models/** – saved models and artifacts  
- **results/** – metrics, plots  
- **README.md** – this file



---

## Dataset (brief)

- The repository expects a CSV with at least these columns: `title`, `text` (or combined full article), and `label` (binary: `fake`/`real` or `0`/`1`).
- Keep sensitive or private data out of the repo; add your dataset to `data/` and update paths in `train.py`.

---

## EDA summary (what I looked for and why)

1. **Class balance** — checked distribution of `fake` vs `real` to decide whether resampling or class weights are needed.  
2. **Text length distribution** — looked at token counts per article to decide truncation/padding choices.  
3. **Top tokens & n-grams** — extracted most frequent unigrams/bigrams per class to see discriminative words.  
4. **Vocabulary overlap** — measured shared vs class-specific vocabulary.  
5. **Missing values** — confirmed and handled missing titles/text.  

(Plots and exact EDA notebooks are in `eda.ipynb`.)

---

## Preprocessing steps

Typical pipeline used in `preprocess.py` (configurable):

1. Concatenate `title` + `text` when available.  
2. Lowercase everything.  
3. Remove punctuation and non-ASCII characters.  
4. Tokenize.  
5. Remove stopwords.  
6. Optional: Lemmatization (slower but helpful).  
7. Re-join tokens for vectorizers.  

These steps were kept modular so you can toggle lemmatization or add stemming.

---

## Feature extraction

- Used `TfidfVectorizer` from scikit-learn with the following main settings:  
  - `ngram_range=(1,2)` (unigrams + bigrams)  
  - `min_df` to ignore very rare tokens  
  - `max_features` to control dimensionality (optional)  
- Saved the fitted vectorizer to `models/` so the exact transformation can be reused at inference time.

---

## Modeling

Two models were trained and compared:

### 1) Multinomial Naive Bayes (Baseline)
- Fast, good baseline for bag-of-words features.  
- Used `alpha` smoothing (tunable).  
- Served as a sanity-check and calibration for text features.  

### 2) Random Forest (Stronger baseline)
- Trained with scikit-learn's `RandomForestClassifier`.  
- Tuned `n_estimators` and `max_depth` in quick grid-search experiments.  
- Handled class imbalance via `class_weight='balanced'` when needed.  

Model artifacts (trained model + vectorizer) are stored in `models/`.

---

## Evaluation

Metrics reported:

- Accuracy  
- Precision, Recall, F1-score (per class and macro)  
- Confusion matrix  

**Typical findings** (your numbers will vary with dataset & splits):  
- Naive Bayes: fast training, decent recall on the majority class, lower F1 on minority class if imbalance exists.  
- Random Forest: usually better overall F1 and precision, but slower and more memory hungry.  

Make sure to evaluate on a held-out test set or via cross-validation.

---
