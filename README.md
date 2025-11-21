# Ninjacart Fresh Produce Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-purple)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)  
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)  
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Graphics-teal)  
![CNN](https://img.shields.io/badge/CNN-Image%20Classification-orange)  
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-ResNet%2C%20VGG19%2C%20MobileNet-green)  
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Object%20Classification-crimson)

---

This project implements a computer vision solution to classify images of fresh produce and market scenes for **Ninjacart**, India’s largest fresh produce supply chain company. The goal is to build and compare deep learning models to accurately distinguish between **Onion**, **Potato**, **Tomato**, and general **Indian Market** (noise) images.

---

## 📁 Project Structure

```
ninjacart_image_classification/
│
├── ninjacart_data/               # Dataset (train/test splits)
│   ├── train/
│   │   ├── indian market/
│   │   ├── onion/
│   │   ├── potato/
│   │   └── tomato/
│   └── test/
│       ├── indian market/
│       ├── onion/
│       ├── potato/
│       └── tomato/
│
├── logs/                         # TensorBoard logs for each model
│
├── *.h5                          # Saved model weights (e.g., ResNet.h5)
│
└── README.md                     # This file
```

---

## 📦 Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- Matplotlib, Seaborn, Pandas, NumPy
- gdown (for downloading dataset)

Install dependencies:
```bash
pip install tensorflow pandas matplotlib seaborn numpy gdown
```

---

## 📥 Dataset

- **Source**: Curated by Ninjacart (available via Google Drive)
- **Classes**:
  - `indian market` (labeled as "noise")
  - `onion`
  - `potato`
  - `tomato`
- **Train Samples**:
  - Tomato: 789
  - Potato: 898
  - Onion: 849
  - Indian Market: 599
- **Test Samples**: Balanced (~80–106 per class)
- Images are **resized to 256×256** during preprocessing.

Download command used:
```bash
!gdown https://drive.google.com/uc?id=1clZX-lV_MLxKHSyeyTheX5OCQtNCUcqT
!unzip ninjacart_data.zip
```

---

## 🧠 Models Evaluated

| Model              | Test Accuracy | Train Accuracy | Key Features |
|--------------------|---------------|----------------|--------------|
| Custom CNN         | 76%           | 81%            | Baseline architecture |
| Custom CNN (Revamped) | 84%        | 89%            | +Data Augmentation, BatchNorm, Dropout, Callbacks |
| VGG19 (fine-tuned) | 91%           | 98%            | Pretrained on ImageNet |
| MobileNet (fine-tuned) | 90%       | 100%           | Lightweight, mobile-ready |
| **ResNet50 (fine-tuned)** | **94%**   | **99%**        | **Best performer** |

### Final Architecture (ResNet50)
- Input: (256, 256, 3)
- Base: `tf.keras.applications.ResNet50` (with `include_top=False`)
- Top Layers:
  - GlobalAveragePooling2D
  - Dropout (rate=0.1)
  - Dense(4, softmax)
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall

---

## 🛠️ Key Techniques Used

- **Data Augmentation**:
  - Random horizontal/vertical flips
  - Random rotation (±20%)
  - Random translation (±20%)
- **Regularization**:
  - Dropout
  - Batch Normalization
- **Callbacks**:
  - `ModelCheckpoint` (save best model)
  - `EarlyStopping` (patience=5)
  - `TensorBoard` (for monitoring)
- **Transfer Learning**: Leveraged ImageNet-pretrained models (VGG19, ResNet50, MobileNet)

---

## 📊 Evaluation Metrics

- Per-class accuracy on test set (ResNet50):
  - **Indian Market (noise)**: 91.36%
  - **Onion**: 98.8%
  - **Potato**: 86.42%
  - **Tomato**: 100%
- Overall test accuracy: **94%**
- Confusion matrices and class-wise metrics visualized using Seaborn.

---

## ▶️ How to Run

1. **Download and extract data** (as shown above).
2. **Load datasets** using `tf.keras.utils.image_dataset_from_directory`.
3. **Train a model** (e.g., ResNet50):
   ```python
   model.fit(aug_ds, validation_data=valid_ds, epochs=20, callbacks=[...])
   ```
4. **Evaluate** on test set:
   ```python
   model.evaluate(test_ds)
   ```
5. **Visualize predictions** and TensorBoard logs:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir logs/
   ```

---

## 🏆 Recommendation

**ResNet50** is recommended for production due to:
- Highest test accuracy (**94%**)
- Strong generalization (minimal overfitting)
- High confidence predictions on unseen data

For mobile deployment where model size matters, **MobileNet** (90% accuracy, only **16 MB**) is a close second.

---

## 📝 Notes

- Class names are alphabetically ordered by TensorFlow:  
  `['indian market', 'onion', 'potato', 'tomato']` → labels `[0, 1, 2, 3]`
- All models use **one-hot encoded labels** (`label_mode='categorical'`)
- Input images are **rescaled to [0, 1]** using `Rescaling(1./255)`

---

*Prepared by: Data Science Team @ Ninjacart*  
*Project Date: 2023*
