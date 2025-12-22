# src/lapf_project/training/train_lapf_nn.py
# cd lapf-ifacjsc2026
# python -m src.lapf_project.training.train_lapf_nn

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..data.text_dataset import data_to_pairs_for_LAPF, pairs_to_batches
from ..data.text_templates import TRAIN_DATA, VAL_DATA
from ..models.lapf_nn import ClassifierHead
from ..models.sentence_encoder import get_sentence_encoder


def train_lapf_nn() -> None:
    # -------------------------------------------------------
    # 0. Hyperparameters & setup
    # -------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    num_labels: int = args.labels  # LABEL_NUM: adjust to your setting
    batch_size: int = args.batch_size
    num_epochs: int = args.epochs
    learning_rate: float = args.lr

    print(f"Number of labels: {num_labels}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------
    # 1. Prepare data: (class_id, text) pairs
    # -------------------------------------------------------
    # TRAIN_DATA, VAL_DATA are list[dict] loaded in text_templates.py
    train_pairs = data_to_pairs_for_LAPF(TRAIN_DATA, num_labels)
    val_pairs = data_to_pairs_for_LAPF(VAL_DATA, num_labels)

    num_train_samples = len(train_pairs)
    num_val_samples = len(val_pairs)

    print(f"#train samples: {num_train_samples}")
    print(f"#val   samples: {num_val_samples}")

    # -------------------------------------------------------
    # 2. Models: frozen encoder + trainable head
    # -------------------------------------------------------
    # Frozen Sentence-BERT encoder
    encoder = get_sentence_encoder(device=device)

    # Trainable classification head
    head = ClassifierHead(input_dim=768, output_dim=num_labels).to(device)

    optimizer = Adam(head.parameters(), lr=learning_rate)

    # -------------------------------------------------------
    # 3. Training loop
    # -------------------------------------------------------
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(1, num_epochs + 1):
        # -------------------------
        # 3.1 Training phase
        # -------------------------
        head.train()
        train_batches = pairs_to_batches(
            train_pairs,
            batch_size=batch_size,
            device=device,
            shuffle=True,
            seed=epoch,  # shuffle each epoch with a deterministic seed
        )

        train_loss_sum = 0.0

        for labels, texts in train_batches:
            # labels: Tensor (B,), on `device`
            # texts : List[str], length B

            # 1) Encode texts with frozen Sentence-BERT
            with torch.no_grad():
                embeddings = encoder.encode(texts)  # (B, hidden_dim)

            # 2) Forward through classification head
            logits = head(embeddings)

            # 3) Compute loss
            loss = F.cross_entropy(logits, labels)

            # 4) Backprop (only head parameters are updated)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)

        train_loss = train_loss_sum / num_train_samples
        train_loss_list.append(train_loss)
        print(f"[Epoch {epoch:03d}] train_loss = {train_loss:.6f}")

        # -------------------------
        # 3.2 Validation phase
        # -------------------------
        head.eval()
        val_batches = pairs_to_batches(
            val_pairs,
            batch_size=batch_size,
            device=device,
            shuffle=False,  # validation: keep order
            seed=None,
        )

        val_loss_sum = 0.0
        correct = 0

        with torch.no_grad():
            for labels, texts in val_batches:
                embeddings = encoder.encode(texts)
                logits = head(embeddings)
                loss = F.cross_entropy(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / num_val_samples
        val_acc = correct / num_val_samples

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"[Epoch {epoch:03d}] val_loss = {val_loss:.6f}, val_acc = {val_acc:.4f}")

    # -------------------------------------------------------
    # 4. Plot training curves
    # -------------------------------------------------------
    fig = plt.figure(figsize=(6, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(train_loss_list, c="blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = fig.add_subplot(2, 1, 2)
    (line1,) = ax2.plot(val_loss_list, label="Val Loss", c="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss")

    ax3 = ax2.twinx()
    (line2,) = ax3.plot(val_acc_list, label="Val Acc", c="red")
    ax3.set_ylabel("Val Acc")
    ax3.set_ylim(0.0, 1.0)

    # combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper right")

    plt.tight_layout()

    # Save figure next to checkpoints for reproducibility
    root = Path(__file__).resolve().parents[3]
    checkpoint_dir = (
        root
        / "checkpoints"
        / f"lapf_nn-lb{num_labels}-ep{num_epochs}-bs{batch_size}-lr{learning_rate}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fig_path = checkpoint_dir / "lapf_nn_training_curves.png"
    fig.savefig(fig_path)
    print(f"Saved training curves to: {fig_path}")

    # If you still want to see the plot interactively, uncomment:
    # plt.show()

    # -------------------------------------------------------
    # 5. Save trained head parameters
    # -------------------------------------------------------
    original_state_dict = head.state_dict()
    state_dict = {key: value.cpu() for key, value in original_state_dict.items()}

    checkpoint_path = checkpoint_dir / "lapf_nn_head.pt"

    torch.save(state_dict, checkpoint_path)
    print(f"Saved classifier head checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    train_lapf_nn()
