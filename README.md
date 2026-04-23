# CS6140 User Simulator Project

### Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning · Spring 2026 · Prof. Smruthi Mukund  
**Team:** Bolai Yin · Junke Zhu · Peihan Wang  
**Dataset:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## Project Overview

This project builds a User Simulator that predicts P(like | user, movie) from historical rating data. The simulator is designed to serve as a reward model for offline reinforcement learning, where real users cannot be queried during training.

Our core research question: **Does the sequential order of a user's watch history carry meaningful predictive signal beyond aggregate preference features?**

We compare **XGBoost** (static tabular model) vs **GRU4Rec** (sequential RNN model) through systematic ablation studies on MovieLens 1M.

## Key Findings

- **XGBoost** achieves test AUC **0.7997**, outperforming all GRU4Rec variants.
- **GRU4Rec-SVD (unfrozen)** reaches AUC **0.7813** — the best sequential result, but still trails XGBoost by ~0.018.
- SVD latent features carry the bulk of predictive signal; popularity contributes near-zero marginal gain.
- Embedding initialization matters: Unfrozen SVD (0.7813) > Frozen SVD (0.7692) > Learnable (0.7577).
- Under strict cold-start holdout, movie cold-start is harder than user cold-start, and GRU4Rec is less robust on unseen movies (AUC drops to 0.622 vs XGBoost's 0.718).
- MovieLens 1M shows weak temporal/sequential signal, explaining the limited upside of sequence models on this dataset.

## Project Structure

```
CS6140-User-Simulator/
├── notebooks/
│   ├── cs6140_FInal_Project_xgb_v1.ipynb       # XGBoost: EDA, features, ablations, cold-start
│   └── cs6140_Final_Project_v2_update.ipynb     # GRU4Rec: all variants, Phase 5 comparison, diagnostics
├── docs/
│   ├── CS6140_White_Paper_v3.pdf                # Final white paper
│   └── CS6140_Project_Slides_v1.pdf             # Presentation slides
├── PROJECT_ABSTRACT.md
├── requirements.txt
└── README.md
```

## Methods

### XGBoost (Static Simulator)
- 88-dimensional feature vector: SVD user/item embeddings (32d each), genre indicators (18), release year, number of genres, user demographics, movie popularity
- Tuned: max_depth=6, learning_rate=0.05, n_estimators=500, subsample=0.8
- Per-user temporal split: 80% train / 10% val / 10% test

### GRU4Rec (Sequential Simulator)
- Input: chronological sequence of movie IDs (up to T=100) + target movie ID
- Architecture: Item Embedding (64d) → GRU (hidden=128) → MLP (192→128→1) → logit
- Three embedding variants: Learnable (random init), Frozen SVD, Unfrozen SVD
- BCE loss, Adam optimizer, early stopping (patience=5)

### Evaluation
| Metric | Purpose |
|--------|---------|
| AUC | Threshold-free ranking quality |
| Log Loss | Probabilistic fit quality |
| ECE | Calibration — critical for downstream RL reward signal |

## Ablation Studies

- **SVD dimension sweep**: 16 / 32 / 64 / 128 / 256 (diminishing returns beyond 32)
- **Feature group contribution**: SVD → +movie meta → +user demographics → +popularity
- **XGBoost**: tree depth, regularization, n_estimators, Platt scaling
- **GRU4Rec**: sequence length (T=5–100), hidden dim (64–512), embedding strategy (learnable / frozen SVD / unfrozen SVD)
- **Cold-start**: strict holdout for unseen users and unseen movies

## Getting Started

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) into `data/ml-1m/`
4. Open notebooks in Google Colab (GPU recommended for GRU4Rec training)

## Tech Stack

- Python 3.10+, XGBoost, scikit-learn, pandas, numpy, matplotlib
- PyTorch (GRU4Rec, GRU4RecSVD)
- Google Colab (GPU training)
- GitHub (version control)

## Team Contributions

| Member | Role |
|--------|------|
| **Bolai Yin** | Project lead, XGBoost modeling, ablation design & analysis, cross-model comparison framework |
| **Junke Zhu** | GRU4Rec implementation (all three embedding variants), sequential training pipeline |
| **Peihan Wang** | Data preprocessing, feature engineering (SVD, genre pooling), per-user temporal split, EDA visualizations |

## Future Work

1. **Stronger temporal datasets** — KuaiRec / Tenrec with richer sequential signals and natural fatigue patterns
2. **Transformer architectures** — SASRec to test whether attention captures long-range dependencies better than GRU
3. **RL integration** — use this simulator as the environment for DQN/PPO recommendation agents (ongoing in CS5180)
4. **Cold-start mitigation** — add content-based side information (genre, year) to GRU target representation
5. **Post-hoc calibration** — apply Platt scaling to all models before RL deployment
