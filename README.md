# Food Calorie Estimation from a Single Image 🍽️🔢

Estimate the total calories of a meal **from a single overhead image** using deep learning  
(CNN + Vision Transformer) on the **Nutrition5k** dataset.

> 🔧 Tech stack: PyTorch · torchvision · timm (Vision Transformers) · scikit-learn · Google Colab

---

## 1. Project Overview

This project predicts the **total calories (kCal)** of a plate of food from **one overhead RGB image**.

The focus is on:

- Building a **production-style deep learning pipeline**, not just a single messy notebook.
- Using **pretrained backbones** (ResNet-50, ViT-B/16) fine-tuned for **regression**.
- Clean **code structure**, reusable modules, and a simple **inference script** (`scripts/predict.py`).

High-level pipeline:

```text
Food image (RGB)
        ↓
 Preprocessing & transforms
        ↓
  CNN / Vision Transformer
        ↓
 Regression head (1 unit)
        ↓
 Predicted total calories (kCal)
````

At the moment, the models are trained on a **small debug subset** of Nutrition5k to validate the pipeline.
The metrics are **not** yet representative of full-dataset performance.

---

## 2. Repository Structure

```text
food-calorie-estimation/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ 01_explore_nutrition5k.ipynb   # Combined exploration + debug training + evaluation (Colab)
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ nutrition5k_dataset.py      # PyTorch Dataset + transforms
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ baseline_cnn.py             # ResNetCalorieRegressor
│  │  └─ vit_regressor.py            # ViTCalorieRegressor (timm)
│  ├─ training/
│  │  ├─ __init__.py
│  │  └─ train_loop.py               # Generic train/eval loops + checkpointing
│  └─ evaluation/
│     ├─ __init__.py
│     ├─ metrics.py                  # MAE, RMSE, MAPE, R²
│     └─ plots.py                    # scatter & error hist plots
├─ scripts/
│  ├─ __init__.py
│  └─ predict.py                     # CLI / Python inference: image → calories (ViT)
├─ reports/
│  ├─ debug_metrics_resnet_vs_vit.csv
│  └─ figures/                       # Evaluation plots (generated)
│     ├─ debug_true_vs_pred_resnet.png
│     ├─ debug_true_vs_pred_vit.png
│     ├─ debug_error_hist_resnet.png
│     └─ debug_error_hist_vit.png
├─ models/                           # Saved weights (.pt)   [gitignored]
└─ data/                             # Local data / notes    [gitignored]
```

> During development, all steps (exploration, training, evaluation) were run in a **single Colab notebook** (`01_explore_nutrition5k.ipynb`).
> The structure above is how the code is organized in the repo.

---

## 3. Dataset – Nutrition5k

This project uses **Nutrition5k**, a dataset from Google Research with:

* ~5k real cafeteria dishes.
* For each dish:

  * Overhead RGB-D images.
  * Side-angle videos.
  * **Dish-level metadata** including:

    * `dish_id`
    * `total_calories`
    * `total_mass`
    * `total_fat`, `total_carb`, `total_protein`
* Official **train/test dish splits**.

Official repo:
`https://github.com/google-research-datasets/Nutrition5k`

In this project:

* We focus on **overhead RGB images** only:

  ```text
  imagery/realsense_overhead/<dish_id>/rgb.png
  ```

* We use **`total_calories`** as the main regression target.

* For debugging / fast iteration, we first work on a **small subset**:

  * ~200 dishes sampled from metadata.
  * After downloading images and dropping missing ones, ~132 dishes remain.
  * Split into:

    * 92 train
    * 20 validation
    * 20 test

> ⚠️ The raw Nutrition5k data is **not included** in this repo.
> You must download it yourself and point the code to your local/Drive paths.

---

## 4. Setup & Installation

### 4.1. Clone the repository

```bash
git clone https://github.com/swanframe/food-calorie-estimation.git
cd food-calorie-estimation
```

### 4.2. Python environment (local)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> Note: Torch + timm can be heavy; if you just want to experiment, Colab is recommended.

### 4.3. Colab setup

In a Colab notebook:

```python
!git clone https://github.com/swanframe/food-calorie-estimation.git
%cd food-calorie-estimation

!pip install timm pyyaml
```

Then mount Google Drive if you want to store dataset + models there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 5. Data Preparation (Nutrition5k on Colab)

These steps assume you’re working in **Google Colab** and storing data on **Google Drive**.

### 5.1. Set a data root

```python
from pathlib import Path

DATA_ROOT = Path("/content/drive/MyDrive/data/nutrition5k")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT
```

### 5.2. Download metadata and dish splits

```bash
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/metadata"  "/content/drive/MyDrive/data/nutrition5k"
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids" "/content/drive/MyDrive/data/nutrition5k"
```

This gives you:

```text
data/nutrition5k/
  ├─ metadata/
  │   ├─ dish_metadata_cafe1.csv
  │   ├─ dish_metadata_cafe2.csv
  │   └─ ingredients_metadata.csv
  └─ dish_ids/
      └─ splits/
          ├─ rgb_train_ids.txt
          ├─ rgb_test_ids.txt
          ├─ depth_train_ids.txt
          └─ depth_test_ids.txt
```

### 5.3. Build a debug subset (fast iteration)

In `notebooks/01_explore_nutrition5k.ipynb`:

1. Load `dish_metadata_cafe1.csv` and `dish_metadata_cafe2.csv` with a tolerant parser
   (Nutrition5k CSVs have many ingredient columns; this code keeps only dish-level columns):

   ```python
   import pandas as pd
   from pathlib import Path

   METADATA_DIR = DATA_ROOT / "metadata"

   base_cols = [
       "dish_id",
       "total_calories",
       "total_mass",
       "total_fat",
       "total_carb",
       "total_protein",
       "num_ingrs",
   ]

   cafe1 = pd.read_csv(
       METADATA_DIR / "dish_metadata_cafe1.csv",
       engine="python",
       header=None,
       on_bad_lines="skip",
   )
   cafe2 = pd.read_csv(
       METADATA_DIR / "dish_metadata_cafe2.csv",
       engine="python",
       header=None,
       on_bad_lines="skip",
   )

   extra_cols1 = [f"extra_{i}" for i in range(cafe1.shape[1] - len(base_cols))]
   extra_cols2 = [f"extra_{i}" for i in range(cafe2.shape[1] - len(base_cols))]

   cafe1.columns = base_cols + extra_cols1
   cafe2.columns = base_cols + extra_cols2

   cafe1 = cafe1[base_cols]
   cafe2 = cafe2[base_cols]

   cafe1 = cafe1[cafe1["dish_id"] != "dish_id"]
   cafe2 = cafe2[cafe2["dish_id"] != "dish_id"]

   for col in ["total_calories", "total_mass", "total_fat", "total_carb", "total_protein", "num_ingrs"]:
       cafe1[col] = pd.to_numeric(cafe1[col], errors="coerce")
       cafe2[col] = pd.to_numeric(cafe2[col], errors="coerce")

   dish_meta = pd.concat([cafe1, cafe2], ignore_index=True)
   ```

2. Sample a small subset with non-null calories:

   ```python
   dish_meta_clean = dish_meta.dropna(subset=["total_calories"])

   DEBUG_N_DISHES = 200
   debug_dishes = dish_meta_clean.sample(
       n=DEBUG_N_DISHES,
       random_state=42,
   ).copy()
   ```

3. Download **only** the overhead `rgb.png` for each sampled dish:

   ```python
   import subprocess
   from tqdm import tqdm

   OVERHEAD_LOCAL_ROOT = DATA_ROOT / "imagery" / "realsense_overhead"
   OVERHEAD_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

   missing_rgb = []

   for dish_id in tqdm(debug_dishes["dish_id"].tolist()):
       local_dish_dir = OVERHEAD_LOCAL_ROOT / dish_id
       local_dish_dir.mkdir(parents=True, exist_ok=True)

       gs_path = f"gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/{dish_id}/rgb.png"

       try:
           subprocess.run(
               ["gsutil", "cp", gs_path, str(local_dish_dir)],
               check=True,
               stdout=subprocess.PIPE,
               stderr=subprocess.PIPE,
           )
       except subprocess.CalledProcessError:
           missing_rgb.append(dish_id)

   debug_dishes = debug_dishes[~debug_dishes["dish_id"].isin(missing_rgb)].reset_index(drop=True)
   debug_dishes.shape
   # -> e.g. (132, 7)
   ```

4. Split into train / val / test and save:

   ```python
   from sklearn.model_selection import train_test_split

   train_df, temp_df = train_test_split(
       debug_dishes,
       test_size=0.3,
       random_state=42,
   )
   val_df, test_df = train_test_split(
       temp_df,
       test_size=0.5,
       random_state=42,
   )

   debug_splits_dir = DATA_ROOT / "debug_splits"
   debug_splits_dir.mkdir(parents=True, exist_ok=True)

   train_df.to_csv(debug_splits_dir / "debug_train.csv", index=False)
   val_df.to_csv(debug_splits_dir / "debug_val.csv", index=False)
   test_df.to_csv(debug_splits_dir / "debug_test.csv", index=False)
   ```

For full experiments later, you can use the official `rgb_train_ids.txt` / `rgb_test_ids.txt` instead of random sampling.

---

## 6. Data Pipeline

All data loading is handled by `Nutrition5kOverheadDataset` in `src/data/nutrition5k_dataset.py`.

Key points:

* Expects a DataFrame/CSV with at least:

  * `dish_id`
  * `total_calories`

* Images are loaded from:

  ```text
  images_root / dish_id / "rgb.png"
  ```

* Uses shared transforms from `get_transforms(image_size=224)`:

  * **Train**: resize → random horizontal flip → light color jitter → normalize (ImageNet mean/std).
  * **Eval/Test**: resize → normalize.

This dataset is used for both ResNet and ViT, so swapping models is easy.

---

## 7. Models

### 7.1. Baseline – ResNet-50 Calorie Regressor

File: `src/models/baseline_cnn.py`

* Backbone: `torchvision.models.resnet50` (ImageNet pretrained).

* Final FC replaced with a **regression head**:

  ```python
  nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features, 1),
  )
  ```

* Option to **freeze the backbone** (useful for tiny debug sets):

  * Trains only the final layer on top of fixed visual features.

### 7.2. Main Model – ViT Calorie Regressor (timm)

File: `src/models/vit_regressor.py`

* Backbone: `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)`.
* Uses a **Vision Transformer (ViT-B/16)** pretrained on ImageNet.
* Output: scalar prediction for calories (shape `[B]` after `squeeze(-1)`).
* Can freeze backbone for the debug subset; later, unfreeze for full dataset training.

Both models plug into the same training and evaluation code.

---

## 8. Training (Debug Subset)

> For now, training is driven through the Colab notebook `notebooks/01_explore_nutrition5k.ipynb`.

### 8.1. Datasets & DataLoaders

```python
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.nutrition5k_dataset import Nutrition5kOverheadDataset, get_transforms

DATA_ROOT = Path("/content/drive/MyDrive/data/nutrition5k")
OVERHEAD_LOCAL_ROOT = DATA_ROOT / "imagery" / "realsense_overhead"
debug_splits_dir = DATA_ROOT / "debug_splits"

train_df = pd.read_csv(debug_splits_dir / "debug_train.csv")
val_df   = pd.read_csv(debug_splits_dir / "debug_val.csv")
test_df  = pd.read_csv(debug_splits_dir / "debug_test.csv")

train_transform, eval_transform = get_transforms(image_size=224)

train_dataset = Nutrition5kOverheadDataset(train_df, OVERHEAD_LOCAL_ROOT, "total_calories", transform=train_transform)
val_dataset   = Nutrition5kOverheadDataset(val_df,   OVERHEAD_LOCAL_ROOT, "total_calories", transform=eval_transform)
test_dataset  = Nutrition5kOverheadDataset(test_df,  OVERHEAD_LOCAL_ROOT, "total_calories", transform=eval_transform)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=torch.cuda.is_available())
```

### 8.2. Training loop

Encapsulated in `src/training/train_loop.py`:

* `train_one_epoch(...)`
* `evaluate(...)`
* `train_model(...)` (with checkpointing based on validation MAE)

Example (ResNet debug training):

```python
from src.models.baseline_cnn import ResNetCalorieRegressor
from src.training.train_loop import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetCalorieRegressor(
    backbone_name="resnet50",
    pretrained=True,
    dropout_p=0.3,
    freeze_backbone=True,   # good for tiny debug set
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

MODEL_DIR = Path("/content/drive/MyDrive/models/food-calorie-estimation")
checkpoint_path = MODEL_DIR / "baseline_resnet_debug.pt"

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    loss_fn=torch.nn.MSELoss(),
    num_epochs=5,
    use_amp=True,
    checkpoint_path=checkpoint_path,
)
```

ViT training is analogous, using `ViTCalorieRegressor` with e.g. `lr=3e-5`.

---

## 9. Evaluation & Error Analysis (Debug Subset)

Evaluation is implemented in:

* `src/evaluation/metrics.py`
* `src/evaluation/plots.py`
* `notebooks/01_explore_nutrition5k.ipynb` (Phase 6 section)

### 9.1. Metrics

`compute_regression_metrics(y_true, y_pred)` returns:

* `mae` – Mean Absolute Error (kCal)
* `mse` – Mean Squared Error
* `rmse` – Root Mean Squared Error (kCal)
* `mape` – Mean Absolute Percentage Error (%)
* `r2` – Coefficient of determination (R²)

Example:

```python
from src.evaluation.metrics import compute_regression_metrics, print_regression_metrics

metrics = compute_regression_metrics(y_true, y_pred)
print_regression_metrics(metrics)
```

### 9.2. Plots

`src/evaluation/plots.py` provides:

* `plot_true_vs_pred(y_true, y_pred, ...)`
* `plot_error_histogram(y_true, y_pred, ...)`

The debug-subset plots are saved to:

```text
reports/figures/
  debug_true_vs_pred_resnet.png
  debug_true_vs_pred_vit.png
  debug_error_hist_resnet.png
  debug_error_hist_vit.png
```

### 9.3. Debug Subset Results (92 train / 20 val / 20 test)

On the **tiny debug test set** (20 dishes), the models are not yet well-trained and often predict values close to 0 kCal. The numbers below mainly confirm that the pipeline works end-to-end.

| Model              | MAE (kCal) | RMSE (kCal) | MAPE (%) |   R²   |
| ------------------ | ---------: | ----------: | -------: | :----: |
| ResNet-50 baseline |     179.31 |      223.72 |    99.38 | -1.795 |
| ViT-B/16           |     178.64 |      223.29 |    97.67 | -1.784 |

> These are **debug-subset** metrics only (≈132 dishes total, 20 in test).
> They are noisy and not representative of what you can achieve by training on the full Nutrition5k train split.

---

## 10. Inference – Predict from a New Image

File: `scripts/predict.py`

The inference script loads a **ViT checkpoint**, applies the same eval transforms as training, and prints a calorie estimate.

### 10.1. CLI Usage

From the repo root:

```bash
python scripts/predict.py \
  "/path/to/food_image.jpg" \
  --checkpoint-path "/path/to/vit_calorie_regressor.pt" \
  --device auto
```

Example (Colab, using the debug ViT checkpoint):

```bash
python scripts/predict.py \
  "/content/drive/MyDrive/data/nutrition5k/imagery/realsense_overhead/dish_1559060106/rgb.png" \
  --checkpoint-path "/content/drive/MyDrive/models/food-calorie-estimation/vit_base_patch16_224_debug.pt" \
  --device auto
```

Sample output:

```text
Image: /content/drive/MyDrive/.../rgb.png
Checkpoint: /content/drive/MyDrive/.../vit_base_patch16_224_debug.pt
Device: cuda

Estimated calories: 0.3 kCal
Rounded estimate: 0 kCal
```

> With the current **debug** checkpoint (trained on ~100 dishes), predictions are often near 0 kCal and should **not** be considered accurate.
> After training on the full Nutrition5k train split, this same script will work with a stronger model.

### 10.2. Programmatic Usage (Python)

You can also call the helper directly:

```python
from pathlib import Path
import sys

REPO_ROOT = Path("/content/food-calorie-estimation")
sys.path.append(str(REPO_ROOT))

from scripts.predict import predict_calories

image_path = "/content/drive/MyDrive/data/nutrition5k/imagery/realsense_overhead/dish_1559060106/rgb.png"
checkpoint_path = "/content/drive/MyDrive/models/food-calorie-estimation/vit_base_patch16_224_debug.pt"

kcal = predict_calories(
    image_path=image_path,
    checkpoint_path=checkpoint_path,
    model_name="vit_base_patch16_224",
    device_str="auto",
    image_size=224,
)

print(f"Estimated calories: {kcal:.1f} kCal")
```

This is useful if you later wrap the model in an API (FastAPI/Flask) or GUI (Gradio/Streamlit).

---

## 11. Limitations & Future Work

Current limitations:

* Trained only on a **small debug subset** of Nutrition5k.
* Predictions are **not yet reliable** calorie estimates.
* Uses only **overhead RGB images** (no depth / multi-view).

Planned / possible extensions:

* **Full Nutrition5k training**

  * Use official train/test splits.
  * Train ViT (and optionally ResNet) with unfreezed backbones.
  * Report metrics on the full test set.

* **Multi-task regression**

  * Predict calories + macros (`total_fat`, `total_carb`, `total_protein`) and/or `total_mass` jointly.

* **Geometry & volume estimation**

  * Incorporate depth maps.
  * Use multiple viewpoints.

* **Model variants**

  * Try ConvNeXt, Swin, EfficientNet, etc.
  * Hyperparameter tuning (LR schedules, augmentations, etc.).

* **Deployment**

  * Expose a REST API with FastAPI/Flask.
  * Simple web UI (Gradio/Streamlit).
  * Dockerize for easier reproducibility.

---

## 12. License & Credits

* **Dataset:** Nutrition5k – please follow the official license and terms in the Nutrition5k repository.
* **Code:** Add your preferred license here (e.g., MIT).