# Skin Cancer Classification Using Bayesian-Optimized Deep Neural Networks

An end-to-end deep learning pipeline built on Kaggle for multi-class skin lesion classification. The repository covers initial clinical image preprocessing and scales into a distributed, Bayesian-optimized **DenseNet201** training workflow on the **HAM10000** dermatoscopic dataset.

## Project Phases

1. **Phase 1 — Clinical Dataset Exploration & Preprocessing (`skin-cancer-fyrp_CNN&VGG.ipynb`)**
   - Audits nested dataset directories across 2,298 clinical `.png` lesion images (`imgs_part_1`–`imgs_part_3`) and `metadata.csv`.
   - Converts BGR images to RGB via OpenCV, resizes to `224 × 224`, normalizes pixel intensities to `[0.0, 1.0]`, and builds a `(2298, 224, 224, 3)` NumPy tensor with ID-based lookup and Matplotlib verification.

2. **Phase 2 — Bayesian-Optimized DenseNet201 Pipeline on HAM10000**
   - Scales classification to **10,015 dermatoscopic images** across **7 diagnostic categories** from the HAM10000 dataset.
   - Addresses severe class imbalance through **smart oversampling**, **balanced class weights**, and custom **Focal Loss** ($\gamma = 2.0$, $\alpha = 0.25$).
   - Runs **Bayesian Hyperparameter Optimization** (`KerasTuner`) to tune fine-tuning depth, dense head size, dropout rate, and learning rate.
   - Trains the final model using **mixed precision (`mixed_float16`)** distributed across **2× NVIDIA Tesla T4 GPUs** via `tf.distribute.MirroredStrategy`.

---

## Datasets

### Phase 1: Clinical Skin Lesion Dataset
- **Source:** `/kaggle/input/datasets/niharkumardash/skin-cancer-dataset`
- **Volume:** 2,298 `.png` images (`imgs_part_1`: 911, `imgs_part_2`: 659, `imgs_part_3`: 728) + `metadata.csv`
- **Tensor Output:** `(2298, 224, 224, 3)` normalized RGB array

### Phase 2: HAM10000 (`kmader/skin-cancer-mnist-ham10000`)
- **Volume:** 10,015 `.jpg` dermatoscopic images + `HAM10000_metadata.csv`
- **Balanced Volume:** 15,705 images after upsampling minority classes to a target of 1,500 samples each

| Class Code | Diagnosis | Original Count | After Oversampling | Training Class Weight |
| :--- | :--- | :---: | :---: | :---: |
| `nv` | Melanocytic nevi | 6,705 | 6,705 | `0.335` |
| `mel` | Melanoma | 1,113 | 1,500 | `1.496` |
| `bkl` | Benign keratosis-like lesions | 1,099 | 1,500 | `1.496` |
| `bcc` | Basal cell carcinoma | 514 | 1,500 | `1.495` |
| `akiec` | Actinic keratoses | 327 | 1,500 | `1.495` |
| `vasc` | Vascular lesions | 142 | 1,500 | `1.496` |
| `df` | Dermatofibroma | 115 | 1,500 | `1.496` |
| **Total** | **7 Classes** | **10,015** | **15,705** | — |

**Stratified Data Splits (Balanced Dataset):**
- **Train:** 12,013 samples (376 steps/epoch at batch size 32)
- **Validation:** 2,121 samples (67 steps/epoch)
- **Test:** 1,571 samples (10% holdout)

---

## Model Architecture & Training Strategy

### Backbone & Custom Head
- **Base Model:** ImageNet-pretrained `DenseNet201` (`include_top=False`, input `224 × 224 × 3`) with a tunable number of top layers unfrozen for fine-tuning.
- **Classification Head:**
  ```text
  DenseNet201 Base
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dropout (tunable)
    └── Dense (tunable units, ReLU)
    └── BatchNormalization
    └── Dropout (tunable)
    └── Dense (7 classes, Softmax, float32 output)
  ```

### Imbalance & Regularization Techniques
- **Data Augmentation (`ImageDataGenerator`):** DenseNet preprocessing, ±30° rotation, 20% zoom, 15% width/height shifts, 10% shear, `[0.8, 1.2]` brightness jitter, horizontal/vertical flips, and `reflect` fill mode.
- **Loss Function:** Custom multi-class **Focal Loss** (`gamma=2.0`, `alpha=0.25`) combined with balanced `class_weight` during training.
- **Compute Acceleration:** `mixed_float16` global policy (with `float32` softmax output for numerical stability) and dual-GPU `MirroredStrategy`.

---

## Bayesian Hyperparameter Optimization (Stage 1)

Hyperparameter tuning was performed using `keras_tuner.BayesianOptimization` (`max_trials=10`, `num_initial_points=3`, `8 epochs/trial`, optimizing `val_accuracy`):

| Hyperparameter | Search Space | Best Value Found |
| :--- | :--- | :---: |
| `learning_rate` | `[1e-3, 3e-4, 1e-4]` | `0.001` |
| `dropout` | `0.3 – 0.6` (step `0.1`) | `0.4` |
| `dense_units` | `[256, 512]` | `256` |
| `fine_tune_layers` | `50 – 400` (step `50`) | `150` |

- **Best Stage 1 Validation Accuracy:** `88.87%` (`0.8887`)
- **Total Tuning Runtime:** `06h 59m 38s` (Kaggle NVIDIA T4 GPU)

### Final Training (Stage 2)
- **Max Epochs:** `60`
- **Callbacks:**
  - `EarlyStopping` (`monitor='val_accuracy'`, `patience=10`, `restore_best_weights=True`)
  - `ReduceLROnPlateau` (`monitor='val_loss'`, `factor=0.3`, `patience=4`, `min_lr=1e-7`)
  - `ModelCheckpoint` (saves best weights to `output/best_model.keras`)
- **Final Test Results:** *(Add your Stage 2 test accuracy, macro F1-score, and ROC-AUC metrics here)*

---

## Prerequisites & Installation

Developed on Kaggle using **Python 3.12** and **TensorFlow 2.19.0** with **2× NVIDIA Tesla T4 GPUs**. To run locally:

```bash
pip install tensorflow keras-tuner numpy pandas scikit-learn opencv-python matplotlib seaborn tqdm jupyter
```

---

## Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. **Attach or download the datasets:**
   - Phase 1: `niharkumardash/skin-cancer-dataset`
   - Phase 2: `kmader/skin-cancer-mnist-ham10000`
3. **Update `DATASET_PATH`** in Cell 3 if running outside Kaggle:
   ```python
   DATASET_PATH = "/path/to/skin-cancer-mnist-ham10000"
   ```
4. **Run the notebooks sequentially** in Kaggle (GPU T4 ×2 recommended) or Jupyter Notebook.

---

## Generated Outputs

Running the Phase 2 pipeline saves the following artifacts under `/kaggle/working/`:

```text
working/
├── output/
│   ├── class_distribution_before.png
│   ├── class_distribution_after.png
│   ├── label_mapping.json
│   ├── best_hyperparameters.json
│   ├── tuning_history.csv
│   └── best_model.keras
└── tuner/
    ├── ham10000_bo/
    └── best_trial/
```
