# src/lapf_project/models/edapf_nn.py

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.text_dataset import data_to_pairs_for_EDAPF, pairs_to_batches
from ..data.text_templates import VAL_DATA
from ..models.sentence_encoder import get_sentence_encoder


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, h: Tensor) -> Tensor:
        x = self.relu(self.fc1(h))
        x = self.relu(self.fc2(x))
        h = self.fc3(x).flatten()
        return h


class EDAPF_nn(nn.Module):
    """Wrapper model: Sentence-BERT encoder + MLP head."""

    def __init__(self, encoder, head: PredictionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_list: List[str]) -> Tensor:
        # (N, hidden_dim)
        h = self.encoder.encode(text_list)
        h = self.head(h)
        return self.sigmoid(h) * 100


def load_prediction_model(
    device: Union[str, torch.device],
    batch_size: int = 16,
    num_epochs: int = 100,
    lr: float = 1e-5,
) -> EDAPF_nn:
    device = torch.device(device)

    # Sentence-BERT encoder (frozen)
    encoder = get_sentence_encoder(device=device)

    # Prediction head
    input_dim = encoder.model.config.hidden_size
    head = PredictionHead(input_dim=input_dim)
    head.to(device)

    # Load trained weights (head only)
    root = Path(__file__).resolve().parents[3]
    checkpoint_path = (
        root
        / "checkpoints"
        / f"edapf_nn-ep{num_epochs}-bs{batch_size}-lr{lr}"
        / "edapf_nn_head.pt"
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(state_dict, strict=False)

    model = EDAPF_nn(encoder=encoder, head=head)
    model.to(device)
    model.eval()
    return model


def compute_mse_on_val_set(
    model: EDAPF_nn,
    device: Union[str, torch.device],
    batch_size: int = 16,
) -> float:
    device = torch.device(device)

    # Prepare validation data
    val_pairs = data_to_pairs_for_EDAPF(VAL_DATA)
    val_batches = pairs_to_batches(
        val_pairs, batch_size=batch_size, device=device, shuffle=False, seed=None
    )

    val_loss_sum = 0.0

    with torch.no_grad():
        for levels, texts in val_batches:
            preds_y = model(texts)

            loss = F.mse_loss(preds_y, levels.float())

            val_loss_sum += loss.item() * levels.size(0)

    mean_mse = val_loss_sum / len(val_pairs)
    return mean_mse
