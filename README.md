# 🌿 LeafScan AI — Leaf Health Detection

> A college project that uses **transfer learning** to detect water stress, disease, and pest damage in leaf photos through a 4-stage deep learning pipeline.

---

## 📋 Problem Statement

Farmers and horticulturists often detect plant problems too late, leading to crop loss. Manual inspection is slow and expertise-dependent. LeafScan AI lets anyone photograph a leaf and instantly get a health diagnosis with actionable treatment advice.

## 👥 Target Users

- 🌾 Small-scale farmers
- 🌱 Home gardeners
- 🎓 Agriculture students and researchers
- 📱 Anyone with a smartphone camera

---

## ✨ Features

| Feature | Detail |
|---|---|
| Image upload | JPG / PNG drag-and-drop |
| Camera capture | Works on mobile + webcam |
| 4-stage pipeline | Quality → Dryness → Disease → Pest |
| Confidence scores | Per-stage probability |
| Low-confidence fallback | "Please recapture" prompt |
| Actionable advice | Specific treatment per diagnosis |

---

## 🏗️ System Architecture

```
User Photo
    │
    ▼
Stage 1: Quality Check       ← MobileNetV2 (binary: leaf / not_leaf)
    │ (rejected if not a leaf)
    ▼
Stage 2: Dryness Detection   ← EfficientNetB0 (3 classes)
    │
    ▼
Stage 3: Disease Detection   ← ResNet50 (7 classes, PlantVillage)
    │
    ▼
Stage 4: Pest Detection      ← EfficientNetB0 (3 classes)
    │
    ▼
Results Dashboard            ← Label + confidence + recommendation
```

---

## 🧠 ML Approach: Why a Pipeline?

Three approaches were considered:

| Approach | Pros | Cons | Our choice? |
|---|---|---|---|
| **Single multiclass model** | Simple, one training run | Can't handle overlapping problems (disease + pest at once) | ❌ |
| **Multi-label model** | Handles co-occurring issues | Needs multi-label datasets, harder to train | ❌ |
| **Pipeline (staged)** | Modular, each model specialises, easy to explain | Slower inference, more training runs | ✅ |

**We chose the pipeline** because it's:
- Easier to explain in a college presentation
- Each stage can be trained independently
- Models can be swapped or improved one at a time

---

## 📊 Datasets

### Stage 1 — Quality Check
- **Dataset**: ImageNet subset (plant images vs. non-plant)
- **Workaround**: Take 500 leaf photos + 500 random non-leaf images; label manually.

### Stage 2 — Dryness / Water Stress
- **Challenge**: No standard dryness dataset exists!
- **Practical workaround**: 
  1. Collect leaf photos (or use PlantVillage healthy images)
  2. Manually label as healthy / mild_stress / severe_stress based on visual browning, curling, yellowing
  3. A dataset of ~300 images per class is enough to start

### Stage 3 — Disease Detection
- **Dataset**: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — 54,000+ images, 38 disease classes
- **Download**: `kaggle datasets download -d emmarex/plantdisease`

### Stage 4 — Pest Damage
- **Dataset**: [IP102](https://github.com/xpwu95/IP102) — 102 insect pest classes
- **Simpler alternative**: Use 3 classes (healthy / mild_damage / severe_damage) and manually label ~200 images per class

---

## 🚀 Quick Start

### Run the app locally

```bash
git clone https://github.com/yourusername/leafscan-ai
cd leafscan-ai
pip install -r requirements.txt
streamlit run app.py
```

### Train a model (e.g. disease stage)

```bash
python train.py \
  --stage disease \
  --model resnet50 \
  --data_dir data/disease \
  --epochs 20
```

### Google Colab one-liner

```python
from train import train_in_colab
train_in_colab(stage="disease", data_dir="/content/plantvillage", epochs=20)
```

---

## 📁 Folder Structure

```
leafscan-ai/
├── app.py              ← Streamlit web app (main entry point)
├── predict.py          ← Pipeline inference logic
├── train.py            ← Training script with transfer learning
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── models/             ← Saved .pth weight files (after training)
│   ├── stage1_quality.pth
│   ├── stage2_dryness.pth
│   ├── stage3_disease.pth
│   └── stage4_pest.pth
└── data/               ← Dataset folders (not committed to git)
    ├── quality/
    │   ├── train/
    │   └── val/
    ├── dryness/
    ├── disease/
    └── pest/
```

---

## 📈 Evaluation Metrics

| Metric | Why we use it |
|---|---|
| **Accuracy** | Overall correctness |
| **Precision / Recall** | Especially recall — we don't want to miss a disease |
| **F1 Score** | Balances precision and recall on imbalanced classes |
| **Confusion Matrix** | See which classes are confused with each other |

---

## 🌐 Deployment

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → select `app.py`
4. Add model `.pth` files (or load from Google Drive in demo mode)
5. Deploy — free public URL!

---

## 🎓 Presentation Tips

- Explain **why transfer learning** works: ImageNet weights know edges, textures, shapes — perfect for leaf features
- Show the **4-stage architecture diagram** (in README or draw it on a whiteboard)
- Demo live using your phone camera on a real leaf
- Highlight the **uncertainty fallback** as a real-world safety feature

---

*Built by [Your Name] · Data Science Department · College Project 2024*
