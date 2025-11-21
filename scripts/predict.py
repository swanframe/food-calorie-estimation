import argparse
from pathlib import Path
from typing import Union
import sys  # <-- add this

# Ensure repo root is on sys.path so that `import src...` works
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # /content/food-calorie-estimation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from PIL import Image

from src.models.vit_regressor import ViTCalorieRegressor
from src.data.nutrition5k_dataset import get_transforms


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a torch.device from a string.
    - "auto": CUDA if available, else CPU.
    - "cuda" / "cpu": explicit selection.
    """
    device_str = device_str.lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str in ("cuda", "cpu"):
        return torch.device(device_str)
    else:
        raise ValueError(f"Unsupported device string: {device_str}")


def load_vit_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    model_name: str = "vit_base_patch16_224",
) -> nn.Module:
    """
    Load a ViTCalorieRegressor from a checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint saved during training.
        device: torch.device to map the model to.
        model_name: timm ViT model name used during training.

    Returns:
        model in eval() mode on the given device.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Instantiate model with same architecture
    model = ViTCalorieRegressor(
        model_name=model_name,
        pretrained=False,        # weights come from checkpoint
        freeze_backbone=False,   # state_dict will override
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)  # handle both raw and dict checkpoints
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_path: Union[str, Path],
    image_size: int = 224,
) -> torch.Tensor:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: path to input image.
        image_size: size to resize to (height, width).

    Returns:
        A 4D tensor of shape (1, 3, H, W) ready for the model.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use the same eval transform as during validation/test
    _, eval_transform = get_transforms(image_size=image_size)

    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0)  # add batch dimension
    return tensor


@torch.no_grad()
def predict_calories(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    model_name: str = "vit_base_patch16_224",
    device_str: str = "auto",
    image_size: int = 224,
) -> float:
    """
    Convenience function to run a full inference pipeline.

    Returns:
        Predicted calories (float).
    """
    device = get_device(device_str)
    model = load_vit_model(checkpoint_path, device=device, model_name=model_name)

    input_tensor = preprocess_image(image_path, image_size=image_size).to(device)

    preds = model(input_tensor)  # shape (1,)
    kcal = float(preds.item())
    # Clamp to non-negative (just in case)
    kcal = max(kcal, 0.0)
    return kcal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict calories for a food image using a ViT regressor."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input food image (JPEG/PNG).",
    )
    parser.add_argument(
        "--checkpoint-path",
        "-c",
        type=str,
        default="models/vit_calorie_regressor.pt",
        help="Path to the ViT checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm ViT model name used during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (height=width) for resizing before inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image_path
    checkpoint_path = args.checkpoint_path

    kcal = predict_calories(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        device_str=args.device,
        image_size=args.image_size,
    )

    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {get_device(args.device)}")
    print(f"\nEstimated calories: {kcal:.1f} kCal")
    print(f"Rounded estimate: {int(round(kcal))} kCal")


if __name__ == "__main__":
    main()