# CS6140 Project Experiment Outline
## Building a User Simulator: XGBoost vs GRU4Rec vs SASRec

---

## Core Research Question
> On the MovieLens 1M dataset, what are the fundamental differences between
> a static feature model (XGBoost) and sequential models (GRU4Rec / SASRec)
> in predicting P(like | user, movie)?
> Why does GRU4Rec achieve lower AUC than XGBoost?

---

## Phase 1: Exploratory Data Analysis (EDA)
- 1.1 Basic statistics (user count, movie count, like/dislike distribution)
- 1.2 Fatigue signal analysis (does rating drop after consecutive same-genre movies?)
- 1.3 User activity distribution (cold-start vs active users)

---

## Phase 2: XGBoost User Simulator (In-Depth Study)

### 2.1 Baseline
- User embedding: SVD (32-dim) on user-movie matrix
- Movie embedding: SVD (32-dim)
- User history: mean pooling of watched movies
- Genre encoding: one-hot (18-dim)
- Metrics: AUC-ROC / Log Loss / F1 Score

### 2.2 Feature Engineering Ablation
- A: User history — mean pooling vs recency-weighted pooling
- B: Genre encoding — one-hot vs dense embedding
- C: SVD dimensions = 8 / 16 / 32 / 64 / 128

### 2.3 Model Parameter Ablation
- D: max_depth = 3 / 6 / 9 / 12 (Bias-Variance tradeoff)
- E: Regularization — L1 (alpha) and L2 (lambda) strength
- F: n_estimators = 50 / 100 / 200 / 500 (learning curve)

### 2.4 Model Diagnostics
- G: Feature Importance (top 20 features bar chart)
- H: SHAP Analysis (summary plot)
- I: Calibration Curve (predicted probability vs actual probability)
- J: Cold Start vs Active Users (AUC grouped by user activity level)

---

## Phase 3: GRU4Rec User Simulator (In-Depth Study)

### 3.1 Baseline
- Settings: T=20, embed_dim=50, hidden_dim=128
- Baseline result: AUC = 0.703

### 3.2 Sequence Length Ablation (Most Important)
- T = 5 / 10 / 20 / 30 / 50
- Research question: How much history is needed for best prediction?

### 3.3 Model Capacity Ablation
- B: hidden_dim = 32 / 64 / 128 / 256
- C: embed_dim = 16 / 32 / 50 / 100
- D: dropout = 0 / 0.1 / 0.3 / 0.5

### 3.4 Training Process Analysis
- E: Learning curve (train loss vs val AUC per epoch)
- F: Learning rate = 1e-4 / 1e-3 / 1e-2

---

## Phase 4: SASRec (Extended Study, Time Permitting)
- 4.1 Baseline via RecBole
- 4.2 Number of attention heads = 1 / 2 / 4
- 4.3 Sequence length = 10 / 20 / 50

---

## Phase 5: Cross-Model Comparative Analysis
- 5.1 Summary comparison table (AUC / Log Loss / F1)
- 5.2 ROC Curve comparison (all three models on same plot)
- 5.3 Popularity bias analysis (cold vs popular movies)
- 5.4 Final conclusion: root cause of GRU4Rec < XGBoost

---

## Timeline
- Day 1: Phase 1 EDA + XGBoost baseline metrics
- Day 2: XGBoost feature engineering ablation (A / B / C)
- Day 3: XGBoost model parameter ablation + diagnostics (D-J)
- Day 4: GRU4Rec ablation (T / hidden_dim / learning curve)
- Day 5: SASRec via RecBole (time permitting)
- Day 6: Cross-model comparison + finalize all plots
- Day 7: Report writing + code cleanup + push to GitHub

---

## Evaluation Metrics
- Primary: AUC-ROC, Log Loss
- Secondary: F1 Score, Calibration Curve
- Sequential models only: HR@10, NDCG@10

---

## Deliverables

### Code
- data_loader.py
- feature_engineering.py
- xgboost_simulator.py (with full ablation)
- gru4rec_simulator.py (with full ablation)
- sasrec_experiment.py (time permitting)
- analysis.py (cross-model comparison)

### Visualizations (minimum 8 plots)
1. SVD dimensions vs AUC
2. Bias-Variance tradeoff (max_depth ablation)
3. Feature Importance bar chart
4. SHAP summary plot
5. Calibration curve
6. Sequence length T vs AUC (GRU4Rec)
7. Learning curve (GRU4Rec)
8. ROC curve comparison (XGBoost vs GRU4Rec vs SASRec)
