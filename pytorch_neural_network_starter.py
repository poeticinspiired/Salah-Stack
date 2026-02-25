"""Minimal PyTorch neural network example for binary classification."""

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """A tiny multilayer perceptron with one hidden layer."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_synthetic_data(samples: int = 1000):
    """Create two clusters for binary classification."""

    points = torch.randn(samples, 2)
    labels = ((points[:, 0] + points[:, 1]) > 0).float().unsqueeze(1)
    return points, labels


def train_model(epochs: int = 200, learning_rate: float = 1e-2):
    """Train a simple neural network and print progress."""

    x, y = make_synthetic_data(samples=1500)
    model = SimpleMLP(in_features=2, hidden_features=16, out_features=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predictions == y).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={accuracy:.3f}")


if __name__ == "__main__":
    train_model()
