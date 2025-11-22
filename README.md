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
- Clean **module structure**, **reproducible Colab notebooks**, and a simple **inference script** (`scripts/predict.py`).

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

Current experiments:

* A **small debug subset** (≈132 dishes) to validate the end-to-end pipeline.
* A **scaled subset using official RGB train/test splits**, with:

  * Up to 1000 train IDs requested from `rgb_train_ids.txt`
  * After filtering missing images: **566 train / 64 val / 450 test** actually used.

Metrics are still **far from production quality** and mainly show how the pipeline behaves as data scales.

---

## 2. Repository Structure

```text
food-calorie-estimation/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  ├─ 01_explore_nutrition5k.ipynb        # Exploration + debug subset training + evaluation (Colab)
│  └─ 02_full_training_nutrition5k.ipynb  # Scaled training on official RGB splits (configurable size)
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

> There are **two** main Colab notebooks:
>
> * `01_explore_nutrition5k.ipynb` – exploration + debug-scale experiment
> * `02_full_training_nutrition5k.ipynb` – scaled training on official splits (e.g. `N_TRAIN_LIMIT = 1000`)

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

Two levels of experiments:

1. **Debug subset (Notebook 01)**

   * ~200 dishes sampled from the combined metadata.
   * After downloading images and dropping missing ones, ~132 dishes remain.
   * Split into:

     * 92 train
     * 20 validation
     * 20 test

2. **Scaled subset using official splits (Notebook 02)**

   * Start from official `rgb_train_ids.txt` and `rgb_test_ids.txt`.
   * Configurable train size via `N_TRAIN_LIMIT` (e.g. 1000).
   * After merging with metadata and filtering missing RGB overhead images, one run used:

     * 566 train
     * 64 validation
     * 450 test

> ⚠️ The raw Nutrition5k data is **not included** in this repo.
> You must download it yourself and point the code to your local/Drive paths.

---

## 4. How to Run (Quickstart)

### 4.1. Colab (recommended)

#### Option A – Debug subset (fast, small)

1. Open: `notebooks/01_explore_nutrition5k.ipynb`
2. Run the cells in order:

   * Clone repo + install deps.
   * Download Nutrition5k metadata + build debug subset.
   * Create datasets and dataloaders.
   * Train ResNet-50 and ViT-B/16 on the debug subset.
   * Evaluate and plot metrics.
   * Run an inference demo on a sample dish.

This will:

* Save **debug checkpoints** to your Google Drive, for example:
  `/content/drive/MyDrive/models/food-calorie-estimation/`
* Save **evaluation plots** to `reports/figures/`.

#### Option B – Scaled training with official splits

1. Open: `notebooks/02_full_training_nutrition5k.ipynb`

2. At the top of the notebook, you’ll find a config cell:

   ```python
   # ---------- EXPERIMENT CONFIG ----------

   # Use 1000 train dish_ids from rgb_train_ids.txt
   N_TRAIN_LIMIT = 1000      # None => use full RGB train split

   USE_LOCAL_DISK = True     # store big data under /content/data/nutrition5k

   MODEL_ARCH = "vit"        # "vit", "resnet", or "both"

   IMAGE_SIZE = 224
   BATCH_SIZE = 32
   NUM_EPOCHS = 15
   LR_VIT = 3e-5
   LR_RESNET = 1e-4
   WEIGHT_DECAY = 1e-4
   ```

3. Run the notebook top to bottom:

   * Download metadata and dish_ids (if not already present).
   * Build train/val/test DataFrames from official RGB splits.
   * Download overhead RGB images for all used dish_ids.
   * Filter out missing images.
   * Train ViT (and optionally ResNet) on the scaled subset.
   * Evaluate on the test set and log metrics.

Checkpoints are saved to Google Drive, e.g.:

```text
/content/drive/MyDrive/models/food-calorie-estimation/
  vit_base_patch16_224_nutrition5k_1000.pt
```

### 4.2. Local setup (optional)

```bash
git clone https://github.com/swanframe/food-calorie-estimation.git
cd food-calorie-estimation

python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Note: Full-dataset training is GPU-heavy.
> If you only want to explore the pipeline, Colab is usually easier.

---

## 5. Notebook 01 – `01_explore_nutrition5k.ipynb` (Debug Subset)

The main **exploration + debug** notebook is organized into:

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

This notebook is meant to be **readable as an experiment report**, not just a scratchpad.

---

## 6. Notebook 02 – `02_full_training_nutrition5k.ipynb` (Scaled Training)

The **scaled training** notebook extends the pipeline to official RGB splits:

1. **Config cell**

   * Choose train size with `N_TRAIN_LIMIT` (e.g. 1000 or `None` for full train split).
   * Decide where to store data (`USE_LOCAL_DISK` vs Drive).
   * Select models to train (`MODEL_ARCH = "vit"`, `"resnet"`, or `"both"`).
   * Set hyperparameters (batch size, epochs, LR, weight decay).

2. **Environment & paths**

   * Clone repo, install dependencies.
   * Mount Google Drive (for saving checkpoints).
   * Use `/content/data/nutrition5k` for heavy downloads if `USE_LOCAL_DISK=True`.

3. **Metadata & splits**

   * Download `metadata/` and `dish_ids/` if needed.
   * Load:

     * `dish_metadata_cafe1.csv`, `dish_metadata_cafe2.csv`
     * `rgb_train_ids.txt`, `rgb_test_ids.txt`
   * Merge official RGB IDs with metadata.
   * Apply `N_TRAIN_LIMIT` to train IDs.
   * Create a validation split from the train set (e.g., 90% train / 10% val).

4. **Download images**

   * Download overhead `rgb.png` for all dish_ids in train/val/test.
   * Filter out dish_ids with missing RGB images.
   * One run with `N_TRAIN_LIMIT = 1000` produced:

     * 566 train
     * 64 val
     * 450 test

5. **Datasets, DataLoaders & training**

   * Build `Nutrition5kOverheadDataset` and `DataLoader`s for the scaled splits.
   * Train models (e.g. ViT-B/16 with unfrozen backbone) for `NUM_EPOCHS` epochs.
   * Save best checkpoint(s) based on validation MAE.

6. **Evaluation**

   * Reload best checkpoint(s).
   * Compute metrics on the scaled test set.
   * Optionally plot true vs predicted and error histograms for the scaled run.

---

## 7. Data Pipeline

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

## 8. Models

### 8.1. Baseline – ResNet-50 Calorie Regressor

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
  * For larger experiments, the backbone can be unfrozen in the scaled training notebook.

### 8.2. ViT-B/16 Calorie Regressor

File: `src/models/vit_regressor.py`

* Backbone: `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)`.
* Uses a **Vision Transformer (ViT-B/16)** pretrained on ImageNet.
* Output: scalar calorie prediction per image.
* Can be used with:

  * Frozen backbone (debug subset).
  * Unfrozen backbone (scaled training).

Both models integrate with the same training loop and evaluation utilities.

---

## 9. Training Logic

Training is encapsulated in:

```text
src/training/train_loop.py
```

Core features:

* Training loop:

  * Mixed precision (`torch.amp`).
  * Progress logging per epoch.
* Validation loop:

  * MAE monitoring.
  * **Checkpointing**: best model saved based on validation MAE.
* `train_model(...)` is used in both notebooks for ResNet and ViT.

In the current experiments:

* **Notebook 01 – Debug subset**

  * ResNet-50 and ViT-B/16 with **frozen backbones**, ~5 epochs.
  * Goal: verify that the pipeline runs from data → metrics.

* **Notebook 02 – Scaled subset (N_TRAIN_LIMIT = 1000)**

  * ViT-B/16 with **unfrozen backbone**, 15 epochs.
  * Effective dataset: 566 train / 64 val / 450 test (after filtering missing images).

---

## 10. Evaluation & Current Results

Evaluation utilities live in:

```text
src/evaluation/metrics.py
src/evaluation/plots.py
```

### 10.1. Metrics

`compute_regression_metrics(y_true, y_pred)` returns:

* `mae` – Mean Absolute Error (kCal)
* `mse` – Mean Squared Error
* `rmse` – Root Mean Squared Error (kCal)
* `mape` – Mean Absolute Percentage Error (%)
* `r2` – Coefficient of determination (R²)

> Note: **MAPE is not very meaningful** on this dataset because some dishes have `total_calories` near 0, which causes the percentage error to explode. For model comparison, MAE / RMSE / R² are more informative.

### 10.2. Plots

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

### 10.3. Current Results Summary

#### Debug subset (Notebook 01 – ~132 dishes total, 20 test)

| Model              | Train / Val / Test | MAE (kCal) | RMSE (kCal) | MAPE (%) |   R²   |
| ------------------ | ------------------ | ---------: | ----------: | -------: | :----: |
| ResNet-50 baseline | 92 / 20 / 20       |     179.31 |      223.72 |    99.38 | -1.795 |
| ViT-B/16           | 92 / 20 / 20       |     178.64 |      223.29 |    97.67 | -1.784 |

> These are **debug-subset** metrics: noisy and not representative of full-dataset performance.

#### Scaled subset (Notebook 02 – `N_TRAIN_LIMIT = 1000` → 566/64/450)

On the scaled subset using official RGB train/test splits with:

* 566 train
* 64 validation
* 450 test

ViT-B/16 (unfrozen backbone, 15 epochs) achieves:

| Model             | Train / Val / Test | MAE (kCal) | RMSE (kCal) |   R²   |
| ----------------- | ------------------ | ---------: | ----------: | :----: |
| ViT-B/16 (scaled) | 566 / 64 / 450     |     186.87 |      268.86 | -0.724 |

> These metrics are still poor in absolute terms, but they:

> * Show the pipeline working on larger, more realistic splits.
> * Provide a baseline for future improvements (more data, better loss, tuning, etc.).

---

## 11. Inference – Predict from a New Image

File: `scripts/predict.py`

The inference script loads a **ViT checkpoint**, applies the same eval transforms, and prints a calorie estimate.

### 11.1. CLI Usage

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

> With the current **debug** / early scaled checkpoints, predictions are often far from true values and should **not** be used as real calorie estimates.
> As the models are trained on more data with better tuning, this script can be reused with stronger checkpoints.

### 11.2. Programmatic Usage (Python)

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

## 12. Limitations & Future Work

Current limitations:

* Training so far is on:

  * A **small debug subset**, and
  * A **limited-scaled subset** (effective size 566/64/450), not the full dataset.
* Predictions are **not yet reliable** calorie estimates.
* Uses only **overhead RGB images** (no depth / multi-view).
* No extensive hyperparameter search, scheduling, or model selection yet.
* MAPE is not robust due to near-zero-calorie dishes.

Planned / possible extensions:

* **Full Nutrition5k training**

  * Use the full RGB train split (no `N_TRAIN_LIMIT`).
  * Train ViT (and optionally ResNet) with partially or fully unfrozen backbones.
  * Introduce LR schedulers and better regularization.
  * Report robust metrics on the full test set.

* **Better loss / evaluation**

  * Explore L1 loss or a combination of MAE + MSE.
  * Evaluate metrics separately for different calorie ranges.

* **Multi-task regression**

  * Predict calories + macros (`total_fat`, `total_carb`, `total_protein`) and/or `total_mass`.

* **Geometry & volume estimation**

  * Incorporate depth maps and multi-view information.

* **Model variants**

  * Experiment with ConvNeXt, Swin, EfficientNet, etc.

* **Deployment**

  * REST API with FastAPI/Flask.
  * Simple web UI (Gradio/Streamlit).
  * Docker image for reproducible serving.

---

## 13. License & Credits

* **Dataset:** Nutrition5k – please follow the official license and terms from the Nutrition5k repository.
* **Code:** MIT.

---

## 14. Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)