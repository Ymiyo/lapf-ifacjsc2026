# src/lapf_project/models/lapf_nn.py

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..models.sentence_encoder import get_sentence_encoder


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, h: Tensor) -> Tensor:
        x = self.relu(self.fc1(h))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class LAPF_nn(nn.Module):
    """Wrapper model: Sentence-BERT encoder + MLP head."""

    def __init__(self, encoder, head: ClassifierHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.softmax = nn.Softmax()

    def forward(self, text_list: List[str]) -> Tensor:
        # (N, hidden_dim)
        h = self.encoder.encode(text_list)
        logits = self.head(h)
        return self.softmax(logits)


def load_classifier_model(
    device: Union[str, torch.device],
    num_labels: int = 5,
    batch_size: int = 16,
    num_epochs: int = 100,
    lr: float = 1e-5,
) -> LAPF_nn:
    device = torch.device(device)

    # Sentence-BERT encoder (frozen)
    encoder = get_sentence_encoder(device=device)

    # Classification head
    input_dim = encoder.model.config.hidden_size
    head = ClassifierHead(input_dim=input_dim, output_dim=num_labels)
    head.to(device)

    # Load trained weights (head only)
    root = Path(__file__).resolve().parents[3]
    checkpoint_path = (
        root
        / "checkpoints"
        / f"lapf_nn-lb{num_labels}-ep{num_epochs}-bs{batch_size}-lr{lr}"
        / "lapf_nn_head.pt"
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(state_dict, strict=False)

    model = LAPF_nn(encoder=encoder, head=head)
    model.to(device)
    model.eval()
    return model
