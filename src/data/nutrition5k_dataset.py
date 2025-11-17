from pathlib import Path
from typing import Optional, Callable, Union

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalization stats (for pretrained CNNs / ViTs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    """
    Returns (train_transform, eval_transform) for image preprocessing.
    - train_transform: includes light augmentations.
    - eval_transform: deterministic for val/test/inference.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


class Nutrition5kOverheadDataset(Dataset):
    """
    PyTorch Dataset for Nutrition5k overhead RGB images.

    Expects a DataFrame or CSV with at least:
        - 'dish_id'
        - target column (default: 'total_calories')

    Images are assumed to live at:
        images_root / dish_id / 'rgb.png'
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        images_root: Union[str, Path],
        target_col: str = "total_calories",
        transform: Optional[Callable] = None,
        return_id: bool = False,
    ):
        """
        Args:
            data: pandas DataFrame OR path to CSV with dish_id + target_col.
            images_root: root directory containing dish folders, e.g.
                         /.../imagery/realsense_overhead
            target_col: column name in data for the regression target.
            transform: torchvision transform to apply to the PIL image.
            return_id: if True, __getitem__ returns (image, target, dish_id);
                       otherwise (image, target).
        """
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame or a path to a CSV file.")

        if "dish_id" not in data.columns:
            raise ValueError("data must contain a 'dish_id' column.")

        if target_col not in data.columns:
            raise ValueError(f"data must contain the target column '{target_col}'.")

        self.df = data.reset_index(drop=True).copy()
        self.images_root = Path(images_root)
        self.target_col = target_col
        self.transform = transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dish_id = str(row["dish_id"])
        target = float(row[self.target_col])

        img_path = self.images_root / dish_id / "rgb.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for dish_id={dish_id}: {img_path}")

        # Load image as RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Regression target as float32 tensor
        target_tensor = torch.tensor(target, dtype=torch.float32)

        if self.return_id:
            return img, target_tensor, dish_id

        return img, target_tensor