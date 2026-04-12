from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required for the XGBoost baseline. Install requirements.txt first.") from exc


@dataclass
class XGBoostConfig:
    max_depth: int = 6
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42


class XGBoostUserSimulator:
    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self.config = config or XGBoostConfig()
        self.model = XGBClassifier(
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.config.random_state,
        )

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, x, y_true) -> Dict[str, float]:
        probabilities = self.predict_proba(x)
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(y_true, probabilities)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "positive_rate": float(np.mean(predictions)),
        }
