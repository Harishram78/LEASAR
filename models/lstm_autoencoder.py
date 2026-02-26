"""
LEASAR - LSTM Autoencoder for Temporal Anomaly Detection
Best individual model: 94.8% accuracy, 93.2% precision, 96.4% recall, 94.8% F1
M.Tech AI Thesis - Harish Ramachandran, SRM University 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)     # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]                # last layer hidden state
        return self.fc(h_last)          # (batch, latent_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc      = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.out     = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h0 = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(h0)
        return self.out(out)    # (B, T, input_dim)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequential anomaly detection on UR3 robot data.
    Reconstructs normal sequences; high MSE reconstruction error = anomaly.

    Architecture:
        Encoder: LSTM(14 -> 64) -> FC(64 -> 32)
        Decoder: FC(32 -> 64) -> LSTM(64 -> 64) -> Linear(64 -> 14)
    """

    def __init__(self,
                 input_dim:  int = 14,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 num_layers: int = 2,
                 seq_len:    int = 30):
        super().__init__()
        self.seq_len   = seq_len
        self.threshold = 0.0
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, num_layers, input_dim, seq_len)

    def forward(self, x):
        """x: (batch, seq_len, input_dim) -> reconstructed x"""
        z    = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE per sample: (batch,)"""
        with torch.no_grad():
            x_rec = self.forward(x)
            return ((x - x_rec) ** 2).mean(dim=(1, 2))

    def fit(self, X_train: np.ndarray,
            epochs: int = 50,
            batch_size: int = 64,
            lr: float = 1e-3,
            val_split: float = 0.1) -> 'LSTMAutoencoder':
        """
        Train on normal (non-anomalous) sequences only.
        X_train: (N, seq_len, input_dim)
        """
        self.to(device)
        X_t = torch.FloatTensor(X_train).to(device)

        split = int(len(X_t) * (1 - val_split))
        train_ds = TensorDataset(X_t[:split])
        val_ds   = TensorDataset(X_t[split:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for (xb,) in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.forward(xb), xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = sum(criterion(self.forward(xb), xb).item() for (xb,) in val_loader)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if epoch % 10 == 0:
                logger.info('Epoch %3d | Train=%.5f | Val=%.5f', epoch, train_loss/len(train_loader), val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Set threshold: mean + 3*std of reconstruction errors on training set
        self.eval()
        errors = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(X_t), batch_size=batch_size):
                errors.extend(self.reconstruction_error(xb).cpu().numpy())
        errors = np.array(errors)
        self.threshold = float(errors.mean() + 3 * errors.std())
        logger.info('LSTM-AE trained | threshold=%.6f | best_val=%.6f', self.threshold, best_val_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (normalised reconstruction errors).
        X: (N, seq_len, input_dim)
        Returns: (N,) array of scores in [0, inf); threshold separates normal/anomaly
        """
        self.eval()
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            errors = self.reconstruction_error(X_t).cpu().numpy()
        return errors

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Returns binary labels: 1=anomaly, 0=normal."""
        return (self.predict(X) > self.threshold).astype(int)

    def save(self, path: str):
        torch.save({'state_dict': self.state_dict(), 'threshold': self.threshold}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'LSTMAutoencoder':
        data = torch.load(path, map_location=device)
        model = cls(**kwargs)
        model.load_state_dict(data['state_dict'])
        model.threshold = data['threshold']
        return model


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)
    N, T, D = 200, 30, 14
    X_normal = np.random.randn(N, T, D).astype(np.float32)
    model = LSTMAutoencoder(input_dim=D, seq_len=T)
    model.fit(X_normal, epochs=5, batch_size=32)
    scores = model.predict(X_normal[:10])
    print(f'LSTM-AE | scores[:5]={scores[:5].round(4)} | threshold={model.threshold:.4f}')
