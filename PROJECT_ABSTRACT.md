# Project Abstract

## Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning  
**Team:** Bolai Yin (yin.bol@northeastern.edu) · Junke Zhu (zhu.junk@northeastern.edu) · Peihan Wang (email@northeastern.edu)

## What This Project Is About
This project builds a **User Simulator** that predicts P(like | user, movie) using the MovieLens 1M dataset (~1M ratings, 6,000 users, 4,000 movies). The focus is entirely on building the best possible simulator through deep algorithmic analysis.

## Research Question
> Does the sequential order in which a user watches movies carry meaningful predictive signal beyond aggregate statistics — and if so, under what conditions?

## Approach

| Model | Type | How It Treats User History |
|---|---|---|
| **XGBoost** | Tree-based | Static features — order ignored |
| **GRU4Rec** | Recurrent Neural Network | Chronological order — order is central |

Through systematic ablation studies, we investigate:
- Mean pooling vs. recency-weighted user history
- One-hot vs. dense genre encodings
- XGBoost tree depth effects on cold-start vs. active users
- GRU hidden state size and sequence length effects

## Extended Study *(time permitting)*
**SASRec** (Transformer-based) — to examine whether attention mechanisms improve over recurrent approaches.

## Dataset
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings, 6,040 users, 3,706 movies.
