"""
predict.py — LeafScan AI Inference Pipeline
=============================================
This file handles:
  - Loading pretrained / fine-tuned PyTorch models
  - Preprocessing images for each stage
  - Running the 4-stage pipeline
  - Low-confidence fallback logic

Usage (from app.py):
    from predict import run_pipeline
    results = run_pipeline(pil_image)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Confidence threshold — below this we show "uncertain, please recapture"
CONFIDENCE_THRESHOLD = 0.60

# Model file paths (relative to project root)
MODEL_DIR = "models"
MODEL_PATHS = {
    "quality":  os.path.join(MODEL_DIR, "stage1_quality.pth"),
    "dryness":  os.path.join(MODEL_DIR, "stage2_dryness.pth"),
    "disease":  os.path.join(MODEL_DIR, "stage3_disease.pth"),
    "pest":     os.path.join(MODEL_DIR, "stage4_pest.pth"),
}

# Class labels for each stage
LABELS = {
    "quality": ["not_leaf", "leaf"],              # Stage 1 binary
    "dryness": ["healthy", "mild_stress", "severe_stress"],  # Stage 2
    "disease": [                                  # Stage 3 (PlantVillage subset)
        "healthy",
        "bacterial_spot",
        "early_blight",
        "late_blight",
        "leaf_mold",
        "powdery_mildew",
        "septoria_leaf_spot",
    ],
    "pest": ["healthy", "mild_damage", "severe_damage"],     # Stage 4
}

# Recommended actions for each label
ACTIONS = {
    # Dryness actions
    "dryness": {
        "healthy":       "No action needed. Continue normal watering schedule.",
        "mild_stress":   "Increase watering frequency. Check soil moisture daily.",
        "severe_stress": "Urgent: water immediately, check for soil compaction or root issues.",
    },
    # Disease actions
    "disease": {
        "healthy":             "No disease detected. Keep monitoring regularly.",
        "bacterial_spot":      "Remove affected leaves. Apply copper-based bactericide.",
        "early_blight":        "Apply fungicide (chlorothalonil). Remove infected lower leaves.",
        "late_blight":         "Urgent: remove and dispose of infected plants. Apply mancozeb.",
        "leaf_mold":           "Improve air circulation. Apply potassium bicarbonate fungicide.",
        "powdery_mildew":      "Spray with diluted neem oil or sulfur-based fungicide.",
        "septoria_leaf_spot":  "Remove infected leaves. Apply fungicide every 7–10 days.",
    },
    # Pest actions
    "pest": {
        "healthy":       "No pest damage visible. Continue regular inspection.",
        "mild_damage":   "Inspect plant for insects. Consider neem oil spray.",
        "severe_damage": "Urgent: identify the pest type and apply targeted pesticide.",
    },
}


# ── Model Factory ──────────────────────────────────────────────────────────
def build_model(architecture: str, num_classes: int) -> nn.Module:
    """
    Builds a pretrained model with the final layer replaced for our task.
    We use transfer learning — pretrained on ImageNet, fine-tuned on leaf data.

    Supported architectures: 'efficientnet_b0', 'mobilenet_v2', 'resnet50'
    """
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        # Replace the final classifier layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


# ── Model Loader (cached per session) ──────────────────────────────────────
_model_cache: dict = {}  # Avoids reloading on every Streamlit interaction

def load_model(stage: str, architecture: str, num_classes: int) -> nn.Module:
    """
    Loads a trained model from disk (or falls back to random weights for demo).
    In production: replace the fallback with an assertion.
    """
    if stage in _model_cache:
        return _model_cache[stage]

    model = build_model(architecture, num_classes)
    path = MODEL_PATHS[stage]

    if os.path.exists(path):
        # Load your fine-tuned weights
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded {stage} model from {path}")
    else:
        # DEMO MODE: random weights — replace with trained weights for real use
        print(f"⚠️  No weights found at {path} — using untrained model (DEMO only)")

    model.eval()
    model.to(DEVICE)
    _model_cache[stage] = model
    return model


# ── Image Preprocessing ────────────────────────────────────────────────────
# ImageNet normalization — required for pretrained EfficientNet/ResNet/MobileNet
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),        # All models expect 224×224
    transforms.ToTensor(),                # HWC → CHW, [0,255] → [0.0,1.0]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet channel means
        std =[0.229, 0.224, 0.225],       # ImageNet channel stds
    ),
])

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised tensor batch of shape [1, 3, 224, 224]."""
    tensor = PREPROCESS(pil_image)        # [3, 224, 224]
    return tensor.unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]


# ── Single-Stage Inference ─────────────────────────────────────────────────
def infer(model: nn.Module, image_tensor: torch.Tensor, labels: list) -> dict:
    """
    Run a forward pass through a model and return:
      - label       : predicted class name
      - confidence  : max softmax probability (0.0 – 1.0)
      - all_probs   : dict of {class: probability}
      - is_uncertain: True if confidence < CONFIDENCE_THRESHOLD
    """
    with torch.no_grad():
        logits = model(image_tensor)              # [1, num_classes]
        probs  = torch.softmax(logits, dim=1)[0]  # [num_classes]

    probs_np   = probs.cpu().numpy()
    top_idx    = int(np.argmax(probs_np))
    confidence = float(probs_np[top_idx])

    return {
        "label":        labels[top_idx],
        "confidence":   confidence,
        "all_probs":    {lbl: float(p) for lbl, p in zip(labels, probs_np)},
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
    }


# ── Stage 1: Quality / Validity Check ─────────────────────────────────────
def stage1_quality_check(image_tensor: torch.Tensor) -> bool:
    """
    Returns True if the image is a valid leaf, False otherwise.
    Uses MobileNetV2 (lightweight — fast first-pass filter).
    """
    model = load_model("quality", "mobilenet_v2", num_classes=2)
    result = infer(model, image_tensor, LABELS["quality"])

    # If uncertain even about leaf/non-leaf, fail safe and reject
    if result["is_uncertain"]:
        return False

    return result["label"] == "leaf"


# ── Stage 2: Dryness Detection ─────────────────────────────────────────────
def stage2_dryness(image_tensor: torch.Tensor) -> dict:
    """
    Detects water stress / dryness using EfficientNetB0.
    Classes: healthy | mild_stress | severe_stress
    """
    model  = load_model("dryness", "efficientnet_b0", num_classes=3)
    result = infer(model, image_tensor, LABELS["dryness"])
    result["action"] = ACTIONS["dryness"].get(result["label"], "Monitor closely.")
    return result


# ── Stage 3: Disease Detection ─────────────────────────────────────────────
def stage3_disease(image_tensor: torch.Tensor) -> dict:
    """
    Detects common leaf diseases using ResNet50.
    Trained on PlantVillage dataset (38 classes reduced to 7 here).
    """
    model  = load_model("disease", "resnet50", num_classes=len(LABELS["disease"]))
    result = infer(model, image_tensor, LABELS["disease"])
    result["action"] = ACTIONS["disease"].get(result["label"], "Consult an agronomist.")
    return result


# ── Stage 4: Pest Damage Detection ─────────────────────────────────────────
def stage4_pest(image_tensor: torch.Tensor) -> dict:
    """
    Detects pest damage using EfficientNetB0.
    Can be trained on IP102 dataset (102 insect pest classes) or custom labels.
    Classes simplified to: healthy | mild_damage | severe_damage
    """
    model  = load_model("pest", "efficientnet_b0", num_classes=3)
    result = infer(model, image_tensor, LABELS["pest"])
    result["action"] = ACTIONS["pest"].get(result["label"], "Inspect the plant carefully.")
    return result


# ── Main Pipeline ──────────────────────────────────────────────────────────
def run_pipeline(pil_image: Image.Image) -> dict:
    """
    Master 4-stage inference pipeline.
    Returns a dict with results for each stage.

    Example output:
    {
      "is_valid_leaf": True,
      "dryness":  {"label": "mild_stress", "confidence": 0.82, "action": "...", "is_uncertain": False},
      "disease":  {"label": "healthy",     "confidence": 0.91, "action": "...", "is_uncertain": False},
      "pest":     {"label": "mild_damage", "confidence": 0.55, "action": "...", "is_uncertain": True},
    }
    """
    # Preprocess once — reuse for all stages
    image_tensor = preprocess_image(pil_image)

    # ── Stage 1 ──────────────────────────────────────────────────────────
    is_valid = stage1_quality_check(image_tensor)

    if not is_valid:
        return {"is_valid_leaf": False}

    # ── Stages 2–4 (run sequentially; can be parallelised with ThreadPoolExecutor) ──
    dryness_result = stage2_dryness(image_tensor)
    disease_result = stage3_disease(image_tensor)
    pest_result    = stage4_pest(image_tensor)

    return {
        "is_valid_leaf": True,
        "dryness":       dryness_result,
        "disease":       disease_result,
        "pest":          pest_result,
    }
