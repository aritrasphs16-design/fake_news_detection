#  Fake News Detection - Advanced ML System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.66%25-brightgreen.svg)](.)
[![Models](https://img.shields.io/badge/Models-5%20Ensemble-orange.svg)](.)

> 94.66% accuracy using ensemble learning and advanced NLP techniques (No pre-trained models)

---

##  Overview

Machine learning system that classifies news articles as Real or Fake using 5-model ensemble, 30+ features, and advanced text preprocessing.

###  Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.66% |
| Precision | 93.17% |
| Recall | 97.00% |
| F1-Score | 95.05% |
| ROC-AUC | 98.71% |

---

## Model/Approach

### Why Ensemble Learning?

5 Models Combined:
1. Logistic Regression (92.29%)
2. Random Forest (93.87%)
3. XGBoost (94.47%)
4. LightGBM (94.66%)
5. Gradient Boosting (94.66%)

Ensemble Method: Weighted Voting + Stacking → 94.66%

Why?
-  Reduces overfitting
-  More robust predictions
-  Combines strengths of different algorithms
-  Better than any single model

---

## 🔧 Preprocessing & Improvements

### What I Improved:

#### 1. Advanced Text Preprocessing

- URL/Email removal
- Lemmatization (not stemming)
- Smart stopword removal (keep negations)
- HTML tag removal
-Sentiment analysis (VADER)


#### 2. Feature Engineering (30+ Features)

• Text length, word count, sentence count
• Punctuation patterns (!, ?, .)
• Sentiment scores (positive, negative, neutral)
• Character ratios (capitals, digits)
• Linguistic features (pronouns, numbers)
• TF-IDF: 8,000 features (1-3 grams)
• Count Vectorizer: 3,000 features
─────────────────────────────────────
Total: 11,029 features

#### 3. Anomaly Detection
Isolation Forest → Removed 134 outliers (5%)
Clean dataset: 2,528 samples

#### 4. Hyperparameter Tuning

GridSearchCV (5-fold CV) for:
• Logistic Regression: C=10
• XGBoost: max_depth=5, lr=0.1
• LightGBM: num_leaves=31, lr=0.1


#### 5. Dual Vectorization

TF-IDF (semantic) + Count (frequency) = Better features


### Impact:

Before (Basic approach):  90-92% accuracy
After (Our approach):     94.66% accuracy
Improvement:              +2-4%


---

## Model Comparison

| Model | Accuracy | Why Chosen |
|-------|----------|------------|
| Logistic Regression | 92.29% | Baseline, interpretable |
| Random Forest | 93.87% | Non-linear patterns |
| XGBoost | 94.47% | Gradient boosting |
| LightGBM | 94.66% | Fast, efficient  |
| Gradient Boosting | 94.66% | Error correction  |
| Ensemble | 94.66% | Best of all  |

---

##  Quick Start

### Install Dependencies
pip install pandas numpy scikit-learn xgboost lightgbm nltk

### Run in Google Colab
1. Upload ADVANCED_FAKE_NEWS_CLASSIFIER.ipynb
2. Upload datasets (`WELFake_Dataset.csv`, `test.csv`)
3. Run all cells (Runtime → Run all)
4. Download `submission.csv`

---

##  Files

```
├── ADVANCED_FAKE_NEWS_CLASSIFIER.ipynb  # Main notebook
├── WELFake_Dataset.csv                   # Training data
├── test.csv                              # Test data
├── submission.csv                        # Predictions
└── README.md                             # Documentation
```

---

## Key Features

### Why This Approach Works:

1. No Pre-trained Models → Competition compliant
2. Ensemble Learning → 5 models better than 1
3. Advanced Features → 11,029 total features
4. Smart Preprocessing → Lemmatization, sentiment
5. Optimized → GridSearchCV tuning
6. Clean Data → Anomaly removal

---

##  Performance

Confusion Matrix:

True Negatives:  220  |  False Positives: 19
False Negatives: 8    |  True Positives:  259

Test Predictions:
- Fake: 63 (42%)
- Real: 87 (58%)
- High Confidence (≥90%): 91.33%

---

##  Before vs After

| Feature | Before | After |
|---------|--------|-------|
| Preprocessing | Basic | Advanced + Lemmatization |
| Features | 5,000 | 11,029 |
| Models | Single | 5-model ensemble |
| Tuning | None | GridSearchCV |
| Accuracy | 90-92% | 94.66%  |

---

##  Author

Aritra Naskar
- GitHub: [@aritrasphs16-design](https://github.com/aritrasphs16-design)
- Email: aritrasphs16@gmail.com

---



##  Acknowledgments

- WELFake Dataset
- scikit-learn, XGBoost, LightGBM
- NLTK, Google Colab

---


Built using Python, ML, and NLP
