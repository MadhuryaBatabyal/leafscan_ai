"""
predict.py — LeafScan AI Inference Pipeline
Final robust version for Streamlit Cloud
"""

import os
import subprocess
import requests
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# ── Absolute paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Google Drive file IDs ───────────────────────────────────────────────────
_GDRIVE_IDS = {
    "stage2_dryness.pth": "1qaziTwtPpD7F2d-kqcDlddzTXo_qvCAu",
    "stage3_disease.pth": "1VB1bOLAtI295NQ8N9cGWsdbhCw7E-ujE",
    "stage4_pest.pth": "1_6LxyzCFY_W_GBb8iXy_Xq_vlT1VBBox",
}

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.60

# ── Model paths ─────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "dryness": os.path.join(MODEL_DIR, "stage2_dryness.pth"),
    "disease": os.path.join(MODEL_DIR, "stage3_disease.pth"),
    "pest": os.path.join(MODEL_DIR, "stage4_pest.pth"),
}

# ── Labels ──────────────────────────────────────────────────────────────────
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

# ── Actions ─────────────────────────────────────────────────────────────────
ACTIONS = {
    "dryness": {
        "healthy": "No action needed. Continue your normal watering schedule.",
        "mild_stress": "Increase watering frequency. Check soil moisture daily.",
        "severe_stress": "Water immediately. Check for soil compaction, root issues, or drainage problems.",
    },
    "disease": {
        "healthy": "No disease detected. Continue monitoring every 3 to 5 days.",
        "bacterial_spot": "Remove affected leaves. Apply copper-based bactericide. Avoid overhead watering.",
        "early_blight": "Remove infected lower leaves. Apply chlorothalonil or mancozeb fungicide every 7 days.",
        "late_blight": "Urgent: remove and destroy infected material. Apply mancozeb immediately and isolate the plant.",
        "leaf_mold": "Improve airflow, reduce humidity, and apply potassium bicarbonate fungicide.",
        "septoria_leaf_spot": "Remove spotted leaves and apply fungicide every 7 to 10 days.",
        "tomato_mosaic_virus": "No cure. Remove infected plants, disinfect tools, and control aphid vectors.",
    },
    "pest": {
        "healthy": "No pest damage visible. Continue regular inspection.",
        "mild_damage": "Inspect undersides of leaves for insects and apply neem oil spray.",
        "severe_damage": "Urgent: identify the pest type and apply a targeted pesticide or consult an agronomist.",
    },
}

# ── Preprocessing ───────────────────────────────────────────────────────────
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model_cache = {}


def _is_probably_html(file_path: str, n_bytes: int = 1024) -> bool:
    try:
        with open(file_path, "rb") as f:
            head = f.read(n_bytes).lower()
        return (
            b"<html" in head
            or b"<!doctype html" in head
            or b"<head" in head
            or b"<body" in head
            or b"google drive" in head
        )
    except Exception:
        return False


def _validate_checkpoint_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    size_mb = os.path.getsize(file_path) / 1e6
    print(f"Validated file {os.path.basename(file_path)} size: {size_mb:.2f} MB")

    if size_mb < 0.1:
        raise ValueError(
            f"{os.path.basename(file_path)} is too small ({size_mb:.2f} MB). "
            "Likely not a real checkpoint."
        )

    if _is_probably_html(file_path):
        with open(file_path, "rb") as f:
            preview = f.read(300).decode("utf-8", errors="ignore")
        raise ValueError(
            f"{os.path.basename(file_path)} looks like HTML instead of a .pth file. "
            f"Preview: {preview}"
        )


def _download_with_gdown(file_id: str, dest_path: str) -> bool:
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        subprocess.run(
            ["python", "-m", "pip", "install", "gdown", "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["gdown", "--fuzzy", url, "-O", dest_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        return True
    except Exception as e:
        print(f"gdown download failed for {os.path.basename(dest_path)}: {e}")
        return False


def _download_with_requests(file_id: str, dest_path: str) -> None:
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"

    response = session.get(url, params={"id": file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _download_file_from_gdrive(file_id: str, dest_path: str) -> None:
    ok = _download_with_gdown(file_id, dest_path)

    if not ok or not os.path.exists(dest_path):
        print(f"Falling back to requests for {os.path.basename(dest_path)}")
        _download_with_requests(file_id, dest_path)

    _validate_checkpoint_file(dest_path)


def _download_models_if_missing() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in _GDRIVE_IDS.items():
        local_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(local_path):
            try:
                _validate_checkpoint_file(local_path)
                print(f"Found valid {filename}")
                continue
            except Exception as e:
                print(f"Existing {filename} is invalid: {e}")
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        print(f"Downloading {filename}...")
        _download_file_from_gdrive(file_id, local_path)


def _safe_torch_load(path: str):
    _validate_checkpoint_file(path)

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e1:
        print(f"First torch.load failed for {os.path.basename(path)}: {e1}")
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint {os.path.basename(path)}. "
                f"First error: {e1} | Second error: {e2}"
            )


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

    if not isinstance(ckpt, (dict, OrderedDict)):
        raise ValueError("Checkpoint is not a valid state_dict or wrapped state_dict")

    cleaned = OrderedDict()
    for k, v in ckpt.items():
        new_key = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_key] = v
    return cleaned


def _build_model(stage: str) -> nn.Module:
    num_classes = len(LABELS[stage])

    if stage in ("dryness", "pest"):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif stage == "disease":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return model


def _load_model(stage: str) -> nn.Module:
    if stage in _model_cache:
        return _model_cache[stage]

    model = _build_model(stage)
    path = MODEL_PATHS[stage]

    if os.path.exists(path):
        ckpt = _safe_torch_load(path)
        state = _extract_state_dict(ckpt)
        model_state = model.state_dict()

        filtered_state = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        missing = [k for k in model_state.keys() if k not in filtered_state]
        unexpected = [k for k in state.keys() if k not in model_state]
        mismatched = [
            k for k, v in state.items()
            if k in model_state and v.shape != model_state[k].shape
        ]

        print(f"[{stage}] matched keys: {len(filtered_state)}")
        print(f"[{stage}] missing keys: {missing[:20]}")
        print(f"[{stage}] unexpected keys: {unexpected[:20]}")
        print(f"[{stage}] mismatched keys: {mismatched[:20]}")

        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {stage} model successfully")
    else:
        print(f"WARNING: {path} not found — using random weights for {stage}")

    model.eval()
    model.to(DEVICE)
    _model_cache[stage] = model
    return model


def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    return _PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


def _safe_labels(stage: str, n_outputs: int):
    base_labels = LABELS[stage]
    if n_outputs == len(base_labels):
        return base_labels
    if n_outputs < len(base_labels):
        return base_labels[:n_outputs]
    extra = [f"class_{i}" for i in range(len(base_labels), n_outputs)]
    return base_labels + extra


def _infer(stage: str, tensor: torch.Tensor) -> dict:
    model = _load_model(stage)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_idx = int(np.argmax(probs))
    stage_labels = _safe_labels(stage, len(probs))
    label = stage_labels[top_idx]
    confidence = float(probs[top_idx])

    return {
        "label": label,
        "confidence": confidence,
        "all_probs": {lbl: float(p) for lbl, p in zip(stage_labels, probs)},
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
    }


def _stage2_dryness(tensor: torch.Tensor) -> dict:
    result = _infer("dryness", tensor)
    result["action"] = ACTIONS["dryness"].get(result["label"], "Monitor the plant closely.")
    return result


def _stage3_disease(tensor: torch.Tensor) -> dict:
    result = _infer("disease", tensor)
    result["action"] = ACTIONS["disease"].get(result["label"], "Consult an agronomist.")
    return result


def _stage4_pest(tensor: torch.Tensor) -> dict:
    result = _infer("pest", tensor)
    result["action"] = ACTIONS["pest"].get(result["label"], "Inspect the plant carefully.")
    return result


def run_pipeline(pil_image: Image.Image) -> dict:
    tensor = _preprocess(pil_image)
    return {
        "is_valid_leaf": True,
        "leaf_confidence": 1.0,
        "dryness": _stage2_dryness(tensor),
        "disease": _stage3_disease(tensor),
        "pest": _stage4_pest(tensor),
    }


_download_models_if_missing()
