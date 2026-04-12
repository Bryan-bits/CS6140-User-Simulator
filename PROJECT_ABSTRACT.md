# Project Abstract

## Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning  
**Team:** Bolai Yin (yin.bol@northeastern.edu) · Junke Zhu (zhu.junk@northeastern.edu) · Peihan Wang (email@northeastern.edu)

---

## Project Abstract

This project builds a **User Simulator** that predicts the probability a given user will like a recommended movie — expressed as P(like | user, movie) — using the MovieLens 1M dataset (~1M ratings, 6,040 users, 3,706 movies). Our core objective is to deeply understand the algorithmic mechanisms of our chosen methods and how each design decision affects prediction quality.

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
- SVD embedding dimension (k = 8 / 16 / 32 / 64 / 128)
- Feature group contribution (SVD → +movie meta → +user demo → +popularity)
- XGBoost tree depth, regularization, and estimator count
- GRU4Rec sequence length (T = 5 / 10 / 20 / 50 / 100)
- GRU hidden dimension (64 / 128 / 256 / 512)
- Learnable vs. frozen SVD embedding in GRU4Rec

## Key Results

- **XGBoost** achieves test AUC **0.800**, outperforming GRU4Rec (0.770) by a significant margin.
- The gap is not due to embedding quality or history length — frozen SVD and sequence length ablations confirm the dataset's sequential signal is inherently weak.
- GRU4Rec-SVD (frozen) achieves the **best calibration** (ECE 0.020), suggesting potential value as an RL reward simulator despite lower discrimination.
- Feature ablation shows user demographics provide the largest marginal gain; popularity contributes near-zero — the model captures personalization, not popularity bias.

## Future Work

1. **Datasets with stronger temporal dynamics** — KuaiRec / Tenrec to test whether sequential models benefit from shorter, denser interaction sessions.
2. **Transformer architectures** — SASRec to examine whether attention mechanisms improve over recurrent approaches.
3. **RL integration** — use this simulator as the environment for training DQN/PPO recommendation agents.
4. **Cold-start mitigation** — content-based features or meta-learning to narrow the cold-user AUC gap (0.748 vs 0.809).

---

## Scope of Work

### Team Contributions

| Member | Role |
|---|---|
| **Bolai Yin** | Project architecture, XGBoost modeling, ablation study design and analysis, overall code integration |
| **Junke Zhu** | GRU4Rec model implementation, sequential feature engineering |
| **Peihan Wang** | Data preprocessing, feature engineering pipeline, experiment tracking and results recording |

---

## Dataset
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings, 6,040 users, 3,706 movies.
