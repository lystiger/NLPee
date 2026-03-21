import json
from pathlib import Path

import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "output" / "_output"
DEFAULT_MODEL_PATH = EXPORT_DIR / "vsl_bilstm_state_dict.pth"
DEFAULT_METADATA_PATH = EXPORT_DIR / "vsl_bilstm_metadata.json"


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim=126,
        hidden=128,
        num_layers=2,
        num_classes=9,
        dropout=0.3,
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


def load_metadata(metadata_path=DEFAULT_METADATA_PATH):
    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_labels(metadata_path=DEFAULT_METADATA_PATH):
    metadata = load_metadata(metadata_path)
    return metadata["labels"]


def load_model(
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    device=None,
):
    metadata = load_metadata(metadata_path)
    architecture = metadata["architecture"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMClassifier(
        feature_dim=architecture["feature_dim"],
        hidden=architecture["hidden"],
        num_layers=architecture["num_layers"],
        num_classes=architecture["num_classes"],
        dropout=architecture["dropout"],
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata
