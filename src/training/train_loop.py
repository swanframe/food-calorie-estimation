from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        epoch_loss: average loss over epoch (MSE).
        epoch_mae: average MAE over epoch, in kCal.
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            preds = model(images)               # (B,)
            loss = loss_fn(preds, targets)      # MSE

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mae += torch.abs(preds.detach() - targets).sum().item()
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_mae = running_mae / n_samples

    return epoch_loss, epoch_mae


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate model on a validation or test set.

    Returns:
        epoch_loss: average loss (MSE).
        epoch_mae: average MAE, in kCal.
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        loss = loss_fn(preds, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mae += torch.abs(preds - targets).sum().item()
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_mae = running_mae / n_samples

    return epoch_loss, epoch_mae


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    num_epochs: int = 10,
    use_amp: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, list]:
    """
    High-level training loop with basic checkpointing.

    Args:
        model: PyTorch model.
        train_loader: training DataLoader.
        val_loader: validation DataLoader.
        optimizer: optimizer (e.g. Adam).
        device: torch.device("cuda" or "cpu").
        loss_fn: loss function (default: MSELoss).
        num_epochs: number of training epochs.
        use_amp: use mixed precision on GPU if True.
        checkpoint_path: if provided, saves best model (by val MAE).

    Returns:
        history: dict with lists for 'train_loss', 'val_loss', 'train_mae', 'val_mae'.
    """
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    best_val_mae = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            scaler=scaler,
        )

        val_loss, val_mae = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.2f}, Train MAE: {train_mae:.2f} "
            f"| Val Loss: {val_loss:.2f}, Val MAE: {val_mae:.2f}"
        )

        # Save best model by validation MAE
        if checkpoint_path is not None and val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                },
                checkpoint_path,
            )
            print(f"  -> Saved new best model to {checkpoint_path} (Val MAE: {val_mae:.2f})")

    return history