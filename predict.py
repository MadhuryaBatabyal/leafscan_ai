"""
predict.py — LeafScan AI Inference Pipeline
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import requests

# ── Absolute paths ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Google Drive file IDs ──────────────────────────────────────────────────
_GDRIVE_IDS = {
    "stage2_dryness.pth": "1qaziTwtPpD7F2d-kqcDlddzTXo_qvCAu",
    "stage3_disease.pth": "1VB1bOLAtI295NQ8N9cGWsdbhCw7E-ujE",
    "stage4_pest.pth":    "1_6LxyzCFY_W_GBb8iXy_Xq_vlT1VBBox",
}


def _download_file_from_gdrive(file_id: str, dest_path: str):
    session  = requests.Session()
    url      = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)

    # Handle large-file confirmation cookie
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        response = session.get(
            url, params={"id": file_id, "confirm": token}, stream=True
        )

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _download_models_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, file_id in _GDRIVE_IDS.items():
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            _download_file_from_gdrive(file_id, local_path)
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"  Saved {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  Found {filename} — skipping download")

_download_models_if_missing()


# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.60

MODEL_PATHS = {
    "dryness": os.path.join(MODEL_DIR, "stage2_dryness.pth"),
    "disease": os.path.join(MODEL_DIR, "stage3_disease.pth"),
    "pest":    os.path.join(MODEL_DIR, "stage4_pest.pth"),
}

# ── Confirmed class labels ─────────────────────────────────────────────────
LABELS = {
    "dryness": ["healthy", "mild_stress", "severe_stress"],
    "disease": [
        "bacterial_spot",
        "early_blight",
        "healthy",
        "late_blight",
        "leaf_mold",
        "septoria_leaf_spot",
        "tomato_mosaic_virus",
    ],
    "pest": ["healthy", "mild_damage", "severe_damage"],
}

ACTIONS = {
    "dryness": {
        "healthy":       "No action needed. Continue your normal watering schedule.",
        "mild_stress":   "Increase watering frequency. Check soil moisture daily.",
        "severe_stress": "Water immediately. Check for soil compaction or root issues.",
    },
    "disease": {
        "healthy":            "No disease detected. Continue monitoring every 3-5 days.",
        "bacterial_spot":     "Remove affected leaves. Apply copper-based bactericide. Avoid overhead watering.",
        "early_blight":       "Remove infected lower leaves. Apply chlorothalonil fungicide every 7 days.",
        "late_blight":        "Urgent: remove and destroy infected material. Apply mancozeb immediately.",
        "leaf_mold":          "Improve airflow. Reduce humidity. Apply potassium bicarbonate fungicide.",
        "septoria_leaf_spot": "Remove spotted leaves. Apply fungicide every 7-10 days.",
        "tomato_mosaic_virus":"No cure — remove infected plants. Disinfect tools. Control aphid vectors.",
    },
    "pest": {
        "healthy":       "No pest damage visible. Continue regular inspection.",
        "mild_damage":   "Check undersides of leaves for insects. Apply neem oil spray.",
        "severe_damage": "Urgent: identify pest type and apply targeted pesticide.",
    },
}


# ── Model builder — reads num_classes from saved weights ──────────────────
def _build_model(stage: str, num_classes: int) -> nn.Module:
    """
    num_classes is detected from the saved .pth file directly,
    so this always matches regardless of how many classes were trained.
    """
    if stage in ("dryness", "pest"):
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif stage == "disease":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return m


def _unwrap_checkpoint(raw) -> dict:
    """
    Given whatever torch.load() returns, extract a flat state_dict.
    Handles: plain state_dict, nested checkpoint dicts,
             DataParallel 'module.' prefix, full nn.Module objects.
    """
    # Full model object — return its state_dict
    if isinstance(raw, nn.Module):
        return raw.state_dict()

    # Plain dict — check for common nested checkpoint keys
    if isinstance(raw, dict):
        for nested_key in ("state_dict", "model_state_dict", "model"):
            val = raw.get(nested_key)
            if isinstance(val, dict):
                raw = val
                break
            if isinstance(val, nn.Module):
                return val.state_dict()

    # Strip DataParallel 'module.' prefix if present
    if isinstance(raw, dict) and any(k.startswith("module.") for k in raw):
        raw = {k[len("module."):]: v for k, v in raw.items()}

    return raw


def _detect_num_classes(path: str, stage: str) -> int:
    """
    Reads the saved file and extracts the output size of the final layer.
    Uses a key-agnostic strategy: find the last weight tensor in the
    state_dict — that is always the classification head regardless of
    how the model was saved or named.
    """
    raw = torch.load(path, map_location="cpu")

    # Full model object shortcut
    if isinstance(raw, nn.Module):
        if stage == "disease":
            return raw.fc.out_features
        else:
            return raw.classifier[1].out_features

    state = _unwrap_checkpoint(raw)

    # ── Try known key names first (fast path) ─────────────────────────────
    known_keys = {
        "disease": ("fc.weight", "module.fc.weight"),
        "dryness": ("classifier.1.weight", "module.classifier.1.weight"),
        "pest":    ("classifier.1.weight", "module.classifier.1.weight"),
    }
    for key in known_keys.get(stage, ()):
        if key in state:
            print(f"  [{stage}] found weight key: {key!r}")
            return state[key].shape[0]

    # ── Fallback: scan all keys and use the last weight tensor ────────────
    # The final classification layer is always the last "*.weight" in
    # insertion order, which matches PyTorch's layer registration order.
    weight_keys = [k for k, v in state.items()
                   if k.endswith(".weight") and v.ndim == 2]
    if weight_keys:
        last_key = weight_keys[-1]
        num_classes = state[last_key].shape[0]
        print(f"  [{stage}] key-scan fallback — using {last_key!r} → {num_classes} classes")
        return num_classes

    # ── Last resort: print all keys so the developer can diagnose ─────────
    all_keys = list(state.keys()) if isinstance(state, dict) else []
    raise KeyError(
        f"[{stage}] Cannot detect num_classes from '{path}'.\n"
        f"  Top-level keys ({len(all_keys)}): {all_keys[:20]}"
    )


# ── Model cache ────────────────────────────────────────────────────────────
_model_cache: dict = {}

def _load_model(stage: str) -> nn.Module:
    if stage in _model_cache:
        return _model_cache[stage]

    path = MODEL_PATHS[stage]

    if not os.path.exists(path):
        print(f"WARNING: {path} not found — demo mode (random predictions)")
        num_classes = len(LABELS[stage])
        model = _build_model(stage, num_classes)
        model.eval()
        model.to(DEVICE)
        _model_cache[stage] = model
        return model

    # Detect actual number of classes from the saved weights
    num_classes = _detect_num_classes(path, stage)
    print(f"Loading {stage} model — detected {num_classes} output classes")

    # Warn if it doesn't match our LABELS list
    expected = len(LABELS[stage])
    if num_classes != expected:
        print(
            f"  WARNING: model has {num_classes} classes but LABELS['{stage}'] "
            f"has {expected}. Predictions may be misaligned."
        )

    model = _build_model(stage, num_classes)
    raw = torch.load(path, map_location=DEVICE)

    # Full model object — use directly
    if isinstance(raw, nn.Module):
        model = raw.to(DEVICE)
        model.eval()
        _model_cache[stage] = model
        return model

    state = _unwrap_checkpoint(raw)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    _model_cache[stage] = model
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────
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


# ── Inference ──────────────────────────────────────────────────────────────
def _infer(stage: str, tensor: torch.Tensor) -> dict:
    model = _load_model(stage)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    top_idx    = int(np.argmax(probs))
    num_cls    = len(probs)

    # Guard: if model output size differs from LABELS, truncate/pad safely
    stage_labels = LABELS[stage]
    if num_cls != len(stage_labels):
        # Extend labels with placeholders if model has more classes
        stage_labels = stage_labels + [f"class_{i}" for i in range(len(stage_labels), num_cls)]

    label      = stage_labels[top_idx]
    confidence = float(probs[top_idx])

    return {
        "label":        label,
        "confidence":   confidence,
        "all_probs":    {lbl: float(p) for lbl, p in zip(stage_labels, probs)},
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
    }


# ── Stage functions ────────────────────────────────────────────────────────
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
    """Run the LeafScan pipeline. Stage 1 bypassed."""
    tensor = _preprocess(pil_image)
    return {
        "is_valid_leaf":   True,
        "leaf_confidence": 1.0,
        "dryness":         _stage2_dryness(tensor),
        "disease":         _stage3_disease(tensor),
        "pest":            _stage4_pest(tensor),
    }
