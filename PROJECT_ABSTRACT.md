# Project Abstract

## Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning  
**Team:** Bolai Yin (yin.bol@northeastern.edu) · Junke Zhu (zhu.junk@northeastern.edu) · Peihan Wang (email@northeastern.edu)

---

## Project Abstract

This project builds a **User Simulator** that predicts the probability a given user will like a recommended movie — expressed as P(like | user, movie) — using the MovieLens 1M dataset (~1M ratings, 6,000 users, 4,000 movies). Rather than simply comparing models for accuracy, our core objective is to deeply understand the algorithmic mechanisms of our chosen methods and how each design decision affects prediction quality.

---

## Research Question

> Does the sequential order in which a user watches movies carry meaningful predictive signal beyond aggregate statistics — and if so, under what conditions?

---

## Approach

| Model | Type | How It Treats User History |
|---|---|---|
| **XGBoost** | Tree-based | Static features — order ignored |
| **GRU4Rec** | Recurrent Neural Network | Chronological order — order is central |

Through systematic ablation studies, we investigate:
- Mean pooling vs. recency-weighted user history representation
- One-hot vs. dense genre encodings
- XGBoost tree depth effects on cold-start vs. active users
- GRU hidden state size and sequence length effects

## Extended Study *(time permitting)*
**SASRec** (Transformer-based) — to examine whether attention mechanisms improve over recurrent approaches.

---

## Scope of Work

### Team Contributions

| Member | Role |
|---|---|
| **Bolai Yin** | Project architecture, XGBoost modeling, ablation study design and analysis, overall code integration |
| **Junke Zhu** | GRU4Rec model implementation, sequential feature engineering |
| **Peihan Wang** | Data preprocessing, feature engineering pipeline, experiment tracking and results recording |

### Milestones

| Week | Milestone |
|---|---|
| 1–2 | Data preprocessing, feature engineering, XGBoost baseline |
| 3–4 | XGBoost ablation studies |
| 5–6 | GRU4Rec implementation and training |
| 7 | Comparative analysis, final experiments |
| 8 | Report writing and presentation |

---

## Dataset
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings, 6,040 users, 3,706 movies.
