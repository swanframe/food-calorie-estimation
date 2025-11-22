# Food Calorie Estimation from a Single Image

Estimate the total calories of a meal **from a single overhead image** using deep learning  
(CNN + Vision Transformer) on the **Nutrition5k** dataset.

> 🔧 Tech stack: PyTorch · torchvision · timm (Vision Transformers) · scikit-learn · Google Colab

---

## 1. Project Overview

This project predicts the **total calories (kCal)** of a plate of food from **one overhead RGB image**.

The focus is on:

- Building a **production-style deep learning pipeline**, not just a single monolithic notebook.
- Using **pretrained backbones** (ResNet-50, ViT-B/16) adapted for **regression**.
- Clean **module structure**, a reproducible **Colab experiment notebook**, and a simple **inference script** (`scripts/predict.py`).

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

At the moment, the models are trained on a **small debug subset** of Nutrition5k to validate the pipeline end to end.
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
│     └─ plots.py                    # Scatter & error histogram plots
├─ scripts/
│  ├─ __init__.py
│  └─ predict.py                     # CLI / Python inference: image → calories (ViT)
├─ reports/
│  └─ figures/                       # Evaluation plots (generated)
│     ├─ debug_true_vs_pred_resnet.png
│     ├─ debug_true_vs_pred_vit.png
│     ├─ debug_error_hist_resnet.png
│     └─ debug_error_hist_vit.png
├─ models/                           # Saved weights (.pt)   [gitignored]
└─ data/                             # Local data / notes    [gitignored]
```

> During development, all steps (exploration, training, evaluation) are run in a **single Colab notebook**:
> `notebooks/01_explore_nutrition5k.ipynb`.

---

## 3. Dataset – Nutrition5k

This project uses **Nutrition5k**, a dataset from Google Research with:

* ~5k real cafeteria dishes.
* For each dish:

  * Overhead RGB-D images.
  * Side-angle videos.
  * **Dish-level nutrition metadata**, including:

    * `dish_id`
    * `total_calories`
    * `total_mass`
    * `total_fat`, `total_carb`, `total_protein`
* Official **train/test splits** for RGB and depth.

Official repo:
`https://github.com/google-research-datasets/Nutrition5k`

In this project:

* We focus on **overhead RGB images** only:

  ```text
  imagery/realsense_overhead/<dish_id>/rgb.png
  ```

* We use **`total_calories`** as the main regression target.

For fast iteration, the current experiments use a **small debug subset**:

* ~200 dishes sampled from the combined metadata.
* After downloading images and dropping missing ones, ~132 dishes remain.
* Split into:

  * 92 train
  * 20 validation
  * 20 test

> ⚠️ The raw Nutrition5k data is **not included** in this repo.
> You must download it yourself and point the code to your local/Drive paths.

---

## 4. How to Run (Quickstart)

### 4.1. Colab (recommended)

1. Open the Colab notebook:
   `notebooks/01_explore_nutrition5k.ipynb`
2. Run the cells in order:

   * Clone repo + install deps.
   * Download Nutrition5k metadata + build debug subset.
   * Create datasets and dataloaders.
   * Train ResNet-50 and ViT-B/16 on the debug subset.
   * Evaluate and plot metrics.
   * Run an inference demo on a sample dish.

This will also:

* Save **trained debug checkpoints** to your Google Drive, under e.g.
  `/content/drive/MyDrive/models/food-calorie-estimation/`
* Save **evaluation plots** to `reports/figures/`.

### 4.2. Local setup (optional)

```bash
git clone https://github.com/swanframe/food-calorie-estimation.git
cd food-calorie-estimation

python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Note: Training on the full dataset is GPU-heavy.
> If you only want to explore the pipeline, Colab is usually easier.

---

## 5. Notebook: `01_explore_nutrition5k.ipynb`

The main Colab notebook is organized into:

1. **Environment setup**

   * Clone repo, install dependencies.
   * Configure paths (`REPO_ROOT`, `DATA_ROOT`, `MODEL_DIR`).
   * Set device and random seed.

2. **Download Nutrition5k metadata**

   * Copy `metadata/` and `dish_ids/` from the Nutrition5k GCS bucket to Google Drive.

3. **Build a debug subset**

   * Load `dish_metadata_cafe1.csv` and `dish_metadata_cafe2.csv`.
   * Keep only dish-level columns (drop long ingredient lists).
   * Sample `DEBUG_N_DISHES = 200` dishes with non-null `total_calories`.
   * Download overhead `rgb.png` images for sampled dishes.
   * Drop dishes with missing images (→ ~132 dishes).
   * Split into train/val/test and save:

     * `debug_train.csv`
     * `debug_val.csv`
     * `debug_test.csv`

4. **Datasets & DataLoaders**

   * Use `Nutrition5kOverheadDataset` with shared transforms from `get_transforms(image_size=224)`.
   * Wrap into `DataLoader`s for train/val/test and verify batch shapes.

5. **Training (debug subset)**

   * Train **ResNet-50** baseline (frozen backbone) for a few epochs.
   * Train **ViT-B/16** regressor (frozen backbone) for a few epochs.
   * Save best checkpoints (based on validation MAE) to Google Drive.

6. **Evaluation & plots**

   * Reload best checkpoints for ResNet and ViT.
   * Compute regression metrics on the debug test set.
   * Generate:

     * True vs predicted scatter plots.
     * Error histograms.
   * Save plots into `reports/figures/`.

7. **Inference demo**

   * Pick a random test dish.
   * Show the image.
   * Predict calories with the debug ViT checkpoint.
   * Display true vs predicted calories.

The notebook is meant to be **readable as an experiment report**, not just a scratchpad.

---

## 6. Data Pipeline

All image loading and preprocessing is handled by:

```text
src/data/nutrition5k_dataset.py
```

Key points:

* The dataset expects a DataFrame/CSV with at least:

  * `dish_id`
  * `total_calories`

* Images are loaded from:

  ```text
  images_root / dish_id / "rgb.png"
  ```

* Transforms are shared across models via `get_transforms(image_size=224)`:

  * **Train**:

    * Resize / center crop to 224×224.
    * Random horizontal flip.
    * Light augmentation.
    * Normalize with ImageNet mean/std.
  * **Eval/Test**:

    * Resize / center crop.
    * Normalize.

This makes it easy to switch backbones (ResNet, ViT, etc.) while keeping the same input pipeline.

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

* Optional **backbone freezing**:

  * For the tiny debug subset, only the head is trained.

### 7.2. ViT-B/16 Calorie Regressor

File: `src/models/vit_regressor.py`

* Backbone: `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)`.
* Uses a **Vision Transformer (ViT-B/16)** pretrained on ImageNet.
* Output: scalar calorie prediction per image.

Both models integrate with the same training loop and evaluation utilities.

---

## 8. Training (Debug Subset)

Training is encapsulated in:

```text
src/training/train_loop.py
```

Core features:

* Training loop with:

  * Mixed precision (`torch.amp`).
  * Progress logging per epoch.
* Validation loop with:

  * MAE monitoring.
  * **Checkpointing**: best model saved based on validation MAE.
* Utility `train_model(...)` used in the notebook for both ResNet and ViT.

In the current debug experiments:

* **ResNet-50**:

  * Pretrained backbone, frozen.
  * Trained head only, ~5 epochs.
* **ViT-B/16**:

  * Pretrained backbone, frozen.
  * Smaller learning rate, ~5 epochs.

The goal is to validate that the **end-to-end pipeline works**, not to maximize performance (yet).

---

## 9. Evaluation & Debug Results

Evaluation utilities live in:

```text
src/evaluation/metrics.py
src/evaluation/plots.py
```

### 9.1. Metrics

`compute_regression_metrics(y_true, y_pred)` returns:

* `mae` – Mean Absolute Error (kCal)
* `mse` – Mean Squared Error
* `rmse` – Root Mean Squared Error (kCal)
* `mape` – Mean Absolute Percentage Error (%)
* `r2` – Coefficient of determination (R²)

### 9.2. Plots

* `plot_true_vs_pred(...)` – scatter plot of true vs predicted calories.
* `plot_error_histogram(...)` – histogram of prediction errors.

The debug-subset plots are saved under:

```text
reports/figures/
  debug_true_vs_pred_resnet.png
  debug_true_vs_pred_vit.png
  debug_error_hist_resnet.png
  debug_error_hist_vit.png
```

### 9.3. Current Debug Metrics (92 train / 20 val / 20 test)

On the **tiny debug test set** (20 dishes), the results are mainly a sanity check:

| Model              | MAE (kCal) | RMSE (kCal) | MAPE (%) |   R²   |
| ------------------ | ---------: | ----------: | -------: | :----: |
| ResNet-50 baseline |     179.31 |      223.72 |    99.38 | -1.795 |
| ViT-B/16           |     178.64 |      223.29 |    97.67 | -1.784 |

> These are **debug-subset** metrics only (≈132 dishes total, 20 in test).
> They are noisy and not representative of what’s possible with full Nutrition5k training.

---

## 10. Inference – Predict from a New Image

File: `scripts/predict.py`

The inference script loads a **ViT checkpoint**, applies the same eval transforms, and prints a calorie estimate.

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

Sample output (with current debug model):

```text
Image: /content/drive/MyDrive/.../rgb.png
Checkpoint: /content/drive/MyDrive/.../vit_base_patch16_224_debug.pt
Device: cuda

Estimated calories: 0.3 kCal
Rounded estimate: 0 kCal
```

> With the current **debug** checkpoint (trained on ~100 dishes), predictions are often near 0 kCal
> and should **not** be considered accurate. After training on the full Nutrition5k train split,
> the same script can be reused with a stronger model.

### 10.2. Programmatic Usage (Python)

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

This can later be wrapped in an API (FastAPI/Flask) or a small UI (Gradio/Streamlit).

---

## 11. Limitations & Future Work

Current limitations:

* Trained only on a **small debug subset** of Nutrition5k.
* Predictions are **not yet reliable** calorie estimates.
* Uses only **overhead RGB images** (no depth / multi-view).
* No hyperparameter search or model selection has been done yet.

Planned / possible extensions:

* **Full Nutrition5k training**

  * Use official RGB train/test splits.
  * Train ViT (and optionally ResNet) with partially or fully unfrozen backbones.
  * Report robust metrics on the full test set.

* **Multi-task regression**

  * Predict calories + macros (`total_fat`, `total_carb`, `total_protein`) and/or `total_mass` jointly.

* **Geometry & volume estimation**

  * Incorporate depth maps and multi-view information.

* **Model variants**

  * Experiment with ConvNeXt, Swin, EfficientNet, etc.
  * Better augmentations and learning rate schedules.

* **Deployment**

  * REST API with FastAPI/Flask.
  * Simple web UI (Gradio/Streamlit).
  * Docker image for reproducible serving.

---

## 12. License & Credits

* **Dataset:** Nutrition5k – please follow the official license and terms from the Nutrition5k repository.
* **Code:** MIT.

---

## 13. Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)