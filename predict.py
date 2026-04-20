"""
predict.py — LeafScan AI Inference Pipeline
=============================================
Drop your four .pth files into models/ and this file handles everything:
  - Loading each model once and caching it (fast repeat calls)
  - Preprocessing images identically to how they were trained
  - Running the 4-stage pipeline in sequence
  - Returning structured results with confidence and recommended action

Your confirmed class orders from training (do not change these):
  Stage 1: ['leaf', 'not_leaf']        — MobileNetV2
  Stage 2: ['healthy', 'mild_stress', 'severe_stress']  — EfficientNetB0
  Stage 3: ['bacterial_spot', 'early_blight', 'healthy', 'late_blight',
             'leaf_mold', 'septoria_leaf_spot', 'tomato_mosaic_virus']  — ResNet50
  Stage 4: ['healthy', 'mild_damage', 'severe_damage']  — EfficientNetB0
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Confidence threshold ───────────────────────────────────────────────────
# If the top prediction is below this, we show "uncertain — please recapture"
CONFIDENCE_THRESHOLD = 0.60

# ── Model paths ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATHS = {
    "quality": os.path.join(MODEL_DIR, "stage1_quality.pth"),
    "dryness": os.path.join(MODEL_DIR, "stage2_dryness.pth"),
    "disease": os.path.join(MODEL_DIR, "stage3_disease.pth"),
    "pest":    os.path.join(MODEL_DIR, "stage4_pest.pth"),
}

# ── Class labels — confirmed from your training outputs ───────────────────
LABELS = {
    # Stage 1: ImageFolder sorted alphabetically → leaf=0, not_leaf=1
    "quality": ["leaf", "not_leaf"],

    # Stage 2: confirmed from week2 Cell 9 output
    "dryness": ["healthy", "mild_stress", "severe_stress"],

    # Stage 3: confirmed from week1 Cell 4 output
    "disease": [
        "bacterial_spot",
        "early_blight",
        "healthy",
        "late_blight",
        "leaf_mold",
        "septoria_leaf_spot",
        "tomato_mosaic_virus",
    ],

    # Stage 4: confirmed from week2 Cell 9 output
    "pest": ["healthy", "mild_damage", "severe_damage"],
}

# ── Recommended actions ────────────────────────────────────────────────────
ACTIONS = {
    "dryness": {
        "healthy":       "No action needed. Continue your normal watering schedule.",
        "mild_stress":   "Increase watering frequency. Check soil moisture daily — it should be damp 2cm below the surface.",
        "severe_stress": "Water immediately. Check for soil compaction, root rot, or drainage issues. Move plant to shade if outdoors.",
    },
    "disease": {
        "healthy":            "No disease detected. Continue monitoring every 3–5 days.",
        "bacterial_spot":     "Remove affected leaves. Apply copper-based bactericide spray. Avoid overhead watering.",
        "early_blight":       "Remove infected lower leaves. Apply chlorothalonil or mancozeb fungicide every 7 days.",
        "late_blight":        "Urgent: remove and destroy infected material. Apply mancozeb fungicide immediately. Isolate plant.",
        "leaf_mold":          "Improve airflow around plant. Reduce humidity. Apply potassium bicarbonate fungicide.",
        "septoria_leaf_spot": "Remove spotted leaves. Apply fungicide (chlorothalonil) every 7–10 days. Avoid wetting foliage.",
        "tomato_mosaic_virus":"No cure — remove and destroy infected plants. Disinfect hands and tools. Control aphid vectors.",
    },
    "pest": {
        "healthy":       "No pest damage visible. Continue regular inspection every few days.",
        "mild_damage":   "Inspect plant closely for insects (undersides of leaves). Apply neem oil spray as a first treatment.",
        "severe_damage": "Urgent: identify the pest type — look for frass, webbing, or insects. Apply targeted pesticide or call an agronomist.",
    },
}


# ── Model builder ──────────────────────────────────────────────────────────
def _build_model(stage: str) -> nn.Module:
    """
    Constructs the correct architecture for each stage.
    Must match exactly what was used during training.
    """
    num_classes = len(LABELS[stage])

    if stage == "quality":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif stage in ("dryness", "pest"):
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif stage == "disease":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return m


# ── Model cache — load once, reuse forever ─────────────────────────────────
_model_cache: dict = {}

def _load_model(stage: str) -> nn.Module:
    """
    Loads model weights from disk on first call, then returns cached version.
    Falls back to random weights with a warning if .pth file is missing.
    """
    if stage in _model_cache:
        return _model_cache[stage]

    model = _build_model(stage)
    path  = MODEL_PATHS[stage]

    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded {stage} model from {path}")
    else:
        print(f"WARNING: No weights at {path} — running in DEMO mode (random predictions)")

    model.eval()
    model.to(DEVICE)
    _model_cache[stage] = model
    return model


# ── Image preprocessing ────────────────────────────────────────────────────
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])

def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    return _PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


# ── Single-stage inference ─────────────────────────────────────────────────
def _infer(stage: str, tensor: torch.Tensor) -> dict:
    model    = _load_model(stage)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    top_idx    = int(np.argmax(probs))
    label      = LABELS[stage][top_idx]
    confidence = float(probs[top_idx])

    return {
        "label":        label,
        "confidence":   confidence,
        "all_probs":    {lbl: float(p) for lbl, p in zip(LABELS[stage], probs)},
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
    }


# ── Individual stage functions ─────────────────────────────────────────────
def _stage1_is_valid_leaf(tensor):
    result    = _infer("quality", tensor)
    leaf_conf = result["all_probs"].get("leaf", 0.0)
    if result["is_uncertain"] or result["label"] != "leaf":
        return False, leaf_conf
    return True, leaf_conf

def _stage2_dryness(tensor):
    r = _infer("dryness", tensor)
    r["action"] = ACTIONS["dryness"].get(r["label"], "Monitor the plant closely.")
    return r

def _stage3_disease(tensor):
    r = _infer("disease", tensor)
    r["action"] = ACTIONS["disease"].get(r["label"], "Consult an agronomist.")
    return r

def _stage4_pest(tensor):
    r = _infer("pest", tensor)
    r["action"] = ACTIONS["pest"].get(r["label"], "Inspect the plant carefully.")
    return r


# ── Public API ─────────────────────────────────────────────────────────────
def run_pipeline(pil_image: Image.Image) -> dict:
    """
    Run the full 4-stage LeafScan pipeline on a PIL image.
    app.py calls only this function — nothing else.

    Returns dict with 'is_valid_leaf' key always present.
    If False, only 'leaf_confidence' is also present.
    If True, 'dryness', 'disease', and 'pest' dicts are also present.
    """
    tensor = _preprocess(pil_image)

    is_valid, leaf_conf = _stage1_is_valid_leaf(tensor)
    if not is_valid:
        return {"is_valid_leaf": False, "leaf_confidence": leaf_conf}

    return {
        "is_valid_leaf":   True,
        "leaf_confidence": leaf_conf,
        "dryness":         _stage2_dryness(tensor),
        "disease":         _stage3_disease(tensor),
        "pest":            _stage4_pest(tensor),
    }
