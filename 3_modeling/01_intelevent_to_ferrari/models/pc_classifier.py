"""
models/pc_classifier.py
=========================
Cabeza de clasificación de las 4 formas Ferrari, entrenada desde cero,
recibiendo como entrada las features FIJAS extraídas con los modelos ONNX
reales de IntellEvent (Opción A: extractor de features, sin reentrenar
IntellEvent).

Input por trial: (T, 4) = [P(no_IC), P(IC), P(no_FO), P(FO)] por frame
Output: logits sobre las 4 clases Ferrari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class AttentionPooling(nn.Module):
    """Pooling por atención sobre el eje temporal, respetando el padding."""

    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)               # [batch, T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x)      # [batch, 1, dim]
        return pooled.squeeze(1)


class PCClassifierFromIntellEventFeatures(nn.Module):
    """
    BiLSTM pequeño + pooling + MLP, entrenado desde cero sobre las salidas
    fijas de IntellEvent. Esta es la pieza "nueva" del pipeline: IntellEvent
    en sí no se modifica ni se reentrena.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.model.feature_dim,
            hidden_size=cfg.model.lstm_hidden_size,
            num_layers=cfg.model.lstm_num_layers,
            batch_first=True,
            bidirectional=cfg.model.bidirectional,
            dropout=cfg.model.lstm_dropout if cfg.model.lstm_num_layers > 1 else 0.0,
        )
        lstm_out_dim = cfg.model.lstm_hidden_size * (2 if cfg.model.bidirectional else 1)

        if cfg.model.pooling == "attention":
            self.pool = AttentionPooling(lstm_out_dim)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, cfg.model.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.lstm_dropout),
            nn.Linear(cfg.model.head_hidden, cfg.data.num_pc_classes),
        )

    def _pool(self, embeddings, mask):
        if self.cfg.model.pooling == "mean":
            mask_exp = mask.unsqueeze(-1)
            summed = (embeddings * mask_exp).sum(dim=1)
            counts = mask_exp.sum(dim=1).clamp(min=1)
            return summed / counts
        elif self.cfg.model.pooling == "max":
            masked = embeddings.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            return masked.max(dim=1).values
        elif self.cfg.model.pooling == "attention":
            return self.pool(embeddings, mask)
        else:
            raise ValueError(f"Pooling desconocido: {self.cfg.model.pooling}")

    def forward(self, x, mask):
        """
        Args:
            x: [batch, T, 4] -- features de IntellEvent (probabilidades IC/FO)
            mask: [batch, T]
        Returns:
            logits: [batch, num_pc_classes]
        """
        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        embeddings, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )

        pooled = self._pool(embeddings, mask)
        logits = self.head(pooled)
        return logits


if __name__ == "__main__":
    model = PCClassifierFromIntellEventFeatures()
    batch_size = 4
    T = CFG.data.max_seq_len_features
    x = torch.randn(batch_size, T, CFG.model.feature_dim)
    mask = torch.ones(batch_size, T)
    mask[:, 1000:] = 0  # simular padding
    out = model(x, mask)
    print("logits shape:", out.shape)
