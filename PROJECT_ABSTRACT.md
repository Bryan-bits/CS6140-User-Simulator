# Project Abstract

## Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning · Spring 2026 · Prof. Smruthi Mukund  
**Team:** Bolai Yin (yin.bol@northeastern.edu) · Junke Zhu (zhu.junk@northeastern.edu) · Peihan Wang (wang.peih@northeastern.edu)

---

## Abstract

This project builds a **User Simulator** that predicts the probability a given user will like a recommended movie — P(like | user, movie) — using the MovieLens 1M dataset (~1M ratings, 6,040 users, 3,706 movies). The simulator is designed to serve as a reward model for offline reinforcement learning, where real users cannot be queried on demand during training.

---

## Research Question

> Does the sequential order of a user's watch history carry meaningful predictive signal beyond aggregate preference features? We designed the comparison to answer this either way.

---

## Approach

| Model | Type | How It Treats User History |
|---|---|---|
| **XGBoost** | Gradient-boosted trees | Static features (SVD embeddings + metadata) — order ignored |
| **GRU4Rec** | Gated Recurrent Unit | Chronological movie ID sequence — order is central |

Through systematic ablation studies, we investigate:

- SVD embedding dimension (k = 16 / 32 / 64 / 128 / 256)
- Feature group contribution (SVD → +movie meta → +user demographics → +popularity)
- XGBoost hyperparameters: tree depth, regularization, estimator count
- GRU4Rec sequence length (T = 5 / 10 / 20 / 50 / 100)
- GRU hidden dimension (64 / 128 / 256 / 512)
- Embedding strategy: Learnable vs. Frozen SVD vs. Unfrozen SVD
- Platt scaling for post-hoc calibration
- Strict cold-start holdout for unseen users and unseen movies

---

## Key Results

| Model | Test AUC | Log Loss | ECE |
|-------|----------|----------|-----|
| XGBoost (best tuned, 88 features) | **0.7997** | 0.5447 | 0.030 |
| XGBoost (SVD-only, 64 features) | 0.7995 | 0.5449 | 0.030 |
| GRU4Rec (learnable) | 0.7577 | 0.5855 | 0.019 |
| GRU4Rec-SVD (frozen) | 0.7692 | 0.5734 | **0.012** |
| GRU4Rec-SVD (unfrozen) | 0.7813 | 0.5713 | 0.041 |

**Main conclusions:**

1. **XGBoost wins on discrimination.** Static collaborative filtering features (SVD) already capture most of the predictive signal on MovieLens 1M.
2. **Sequential gains are limited.** GRU4Rec improves with better initialization (SVD) and longer history (T=100), but cannot close the gap with XGBoost. This is consistent with weak temporal/fatigue signal found in EDA.
3. **Best discriminator ≠ best calibrated model.** GRU4Rec-SVD frozen has the best ECE (0.012) despite lower AUC. For downstream RL use, calibration quality (ECE) may matter more than ranking ability (AUC), since the simulator's probabilities are directly sampled as Bernoulli rewards.
4. **Movie cold-start is harder than user cold-start.** Under strict holdout, XGBoost drops to AUC 0.718 on unseen movies (from 0.80); GRU4Rec drops to 0.622. GRU is more vulnerable because unseen movies have no trained embedding.
5. **Platt scaling improves calibration without affecting ranking.** Post-hoc calibration reduces GRU4Rec-SVD unfrozen's ECE from 0.031 to 0.006.

---

## Limitations

- MovieLens 1M is preference-driven rather than session-driven, which limits the upside of sequential models.
- GRU4Rec uses pure ID-based input — no content-based side information (genre, year) for cold-start fallback.
- The cold-start experiment for GRU uses the learnable variant, not the best-performing SVD unfrozen variant.
- Platt scaling is global (not per-user), which may miss user-level calibration differences.

---

## Future Work

1. **Stronger temporal datasets** — KuaiRec / Tenrec with richer sequential signals and natural fatigue patterns.
2. **Transformer architectures** — SASRec to test whether attention captures long-range dependencies better than GRU.
3. **RL integration** — use this simulator as the environment for DQN/PPO recommendation agents (ongoing in CS5180).
4. **Cold-start mitigation** — add genre/year side information to GRU's target representation so cold movies have fallback features.
5. **Post-hoc calibration** — apply Platt scaling to all models before RL deployment; explore per-user calibration.

---

## Team Contributions

| Member | Role |
|--------|------|
| **Bolai Yin** | Project lead, XGBoost modeling, ablation study design & analysis, cross-model comparison framework, code integration |
| **Junke Zhu** | GRU4Rec implementation (learnable, frozen SVD, unfrozen SVD variants), sequential training pipeline, hidden dim and sequence length experiments |
| **Peihan Wang** | Data preprocessing, feature engineering (SVD, genre pooling), per-user temporal split implementation, experiment tracking, EDA visualizations |

---

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings, 6,040 users, 3,706 movies.
