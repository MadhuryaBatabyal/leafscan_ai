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

# ── Absolute paths — works on local machine and Streamlit Cloud ────────────
# __file__ is always the location of this script regardless of where
# the process was launched from. All paths are built relative to it.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Auto-download models from Google Drive if missing ─────────────────────
try:
    import gdown
except ImportError:
    os.system("pip install -q gdown")
    import gdown

_GDRIVE_IDS = {
    "stage2_dryness.pth": "1qaziTwtPpD7F2d-kqcDlddzTXo_qvCAu",
    "stage3_disease.pth": "1VB1bOLAtI295NQ8N9cGWsdbhCw7E-ujE",
    "stage4_pest.pth":    "1_6LxyzCFY_W_GBb8iXy_Xq_vlT1VBBox",
}

def _download_models_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, file_id in _GDRIVE_IDS.items():
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False, fuzzy=True)
            print(f"  Saved to {local_path}")
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
        "mild_stress":   "Increase watering frequency. Check soil moisture daily — it should be damp 2cm below the surface.",
        "severe_stress": "Water immediately. Check for soil compaction, root rot, or drainage issues. Move plant to shade if outdoors.",
    },
    "disease": {
        "healthy":            "No disease detected. Continue monitoring every 3-5 days.",
        "bacterial_spot":     "Remove affected leaves. Apply copper-based bactericide spray. Avoid overhead watering.",
        "early_blight":       "Remove infected lower leaves. Apply chlorothalonil or mancozeb fungicide every 7 days.",
        "late_blight":        "Urgent: remove and destroy infected material. Apply mancozeb fungicide immediately. Isolate plant.",
        "leaf_mold":          "Improve airflow around plant. Reduce humidity. Apply potassium bicarbonate fungicide.",
        "septoria_leaf_spot": "Remove spotted leaves. Apply fungicide (chlorothalonil) every 7-10 days. Avoid wetting foliage.",
        "tomato_mosaic_virus":"No cure — remove and destroy infected plants. Disinfect hands and tools. Control aphid vectors.",
    },
    "pest": {
        "healthy":       "No pest damage visible. Continue regular inspection every few days.",
        "mild_damage":   "Inspect plant closely for insects (undersides of leaves). Apply neem oil spray as a first treatment.",
        "severe_damage": "Urgent: identify the pest type — look for frass, webbing, or insects. Apply targeted pesticide or call an agronomist.",
    },
}


def _build_model(stage: str) -> nn.Module:
    num_classes = len(LABELS[stage])
    if stage in ("dryness", "pest"):
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif stage == "disease":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return m


_model_cache: dict = {}

def _load_model(stage: str) -> nn.Module:
    if stage in _model_cache:
        return _model_cache[stage]
    model = _build_model(stage)
    path  = MODEL_PATHS[stage]
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded {stage} model")
    else:
        print(f"WARNING: {path} not found — demo mode")
    model.eval()
    model.to(DEVICE)
    _model_cache[stage] = model
    return model


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


def _infer(stage: str, tensor: torch.Tensor) -> dict:
    model = _load_model(stage)
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


def run_pipeline(pil_image: Image.Image) -> dict:
    """
    Run the LeafScan pipeline on a PIL image.
    Stage 1 bypassed — always passes.
    """
    tensor = _preprocess(pil_image)
    return {
        "is_valid_leaf":   True,
        "leaf_confidence": 1.0,
        "dryness":         _stage2_dryness(tensor),
        "disease":         _stage3_disease(tensor),
        "pest":            _stage4_pest(tensor),
    }
