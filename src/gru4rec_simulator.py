from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]], labels: Sequence[int], max_seq_len: int) -> None:
        self.sequences = [list(seq)[-max_seq_len:] for seq in sequences]
        self.labels = list(labels)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        padded = [0] * (self.max_seq_len - len(sequence)) + sequence
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(max(len(sequence), 1), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class GRU4RecModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, item_sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(item_sequences)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        logits = self.output(hidden[-1]).squeeze(-1)
        return logits


@dataclass
class GRU4RecConfig:
    embedding_dim: int = 64
    hidden_size: int = 128
    max_seq_len: int = 50
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    device: str = "cpu"
    verbose: bool = True


class GRU4RecUserSimulator:
    def __init__(self, num_items: int, config: GRU4RecConfig | None = None) -> None:
        self.config = config or GRU4RecConfig()
        self.device = torch.device(self.config.device)
        self.model = GRU4RecModel(
            num_items=num_items,
            embedding_dim=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
        ).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def fit(self, train_sequences: List[List[int]], train_labels: List[int]) -> None:
        dataset = SequenceDataset(train_sequences, train_labels, self.config.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model.train()

        if self.config.verbose:
            print(
                f"Training GRU4Rec on {self.device.type.upper()} "
                f"with {len(dataset)} examples, batch_size={self.config.batch_size}, "
                f"epochs={self.config.epochs}, max_seq_len={self.config.max_seq_len}"
            )

        for epoch in range(self.config.epochs):
            epoch_start = perf_counter()
            epoch_loss = 0.0
            batch_count = 0
            for items, lengths, labels in loader:
                items = items.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(items, lengths)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1

            if self.config.verbose:
                average_loss = epoch_loss / max(batch_count, 1)
                elapsed = perf_counter() - epoch_start
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} "
                    f"- loss: {average_loss:.4f} "
                    f"- time: {elapsed:.1f}s"
                )

    def predict_proba(self, sequences: List[List[int]]) -> np.ndarray:
        dataset = SequenceDataset(sequences, [0] * len(sequences), self.config.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        self.model.eval()
        outputs = []

        with torch.no_grad():
            for items, lengths, _ in loader:
                items = items.to(self.device)
                lengths = lengths.to(self.device)
                logits = self.model(items, lengths)
                outputs.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(outputs, axis=0) if outputs else np.array([])

    def evaluate(self, sequences: List[List[int]], labels: List[int]) -> Dict[str, float]:
        probabilities = self.predict_proba(sequences)
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "positive_rate": float(np.mean(predictions)),
        }
