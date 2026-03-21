import torch
import torch.nn as nn

FEATURE_DIM = 126
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 9
DROPOUT = 0.3

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden * 2)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        h = self.dropout(h)
        h = self.layer_norm(h)
        return self.classifier(h)