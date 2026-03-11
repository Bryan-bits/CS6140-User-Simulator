# CS6140 User Simulator Project
### Building a User Simulator for Movie Recommendation: A Comparative Study of Static and Sequential Models

**Course:** CS6140 Machine Learning — Northeastern University  
**Team:** Bolai Yin · Junke Zhu · Peihan Wang  
**Dataset:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## Project Overview
This project builds a User Simulator that learns to predict P(like | user, movie) from historical rating data. The core research focus is a deep comparative study of **XGBoost** (static features) vs **GRU4Rec** (sequential modeling), with **SASRec** as an extended study time permitting.

See [PROJECT_ABSTRACT.md](PROJECT_ABSTRACT.md) for the full project description.

## Project Structure
```
CS6140-User-Simulator/
├── data/ml-1m/                 # MovieLens 1M raw data (not tracked in git)
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── xgboost_simulator.py
│   ├── gru4rec_simulator.py
│   └── user_simulator.py
├── experiments/results.csv
├── notebooks/01_eda.ipynb
├── PROJECT_ABSTRACT.md
├── requirements.txt
└── README.md
```

## Getting Started
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) into `data/ml-1m/`
4. Open `setup.ipynb` in Google Colab to configure environment

## Tech Stack
- Python 3.10+, XGBoost, scikit-learn, pandas, numpy
- PyTorch (GRU4Rec)
- RecBole (SASRec extended study)
- Google Colab (GPU runtime)

## Team Workflow
1. Never push directly to `main`
2. Create your branch: `git checkout -b your-name-feature`
3. Open a Pull Request and wait for review
