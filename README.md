# Skincaria

Skincaria is a local skincare recommendation prototype. It analyzes a user image or manual concern, normalizes visible skin-condition labels, retrieves matching skincare product context from a local ChromaDB knowledge base, and streams an Indonesian recommendation through an Ollama-hosted vision/language model.

## Current Pipeline

1. The user uploads or captures a still face image.
2. `pipeline.py` asks the configured Ollama model to return normalized skin labels:
   - `skin_type`
   - `acne_severity`
   - `redness`
   - `pore_condition`
   - `dehydration`
3. The normalized labels and user concern are used as a retrieval query against the skincare knowledge base.
4. The recommendation step combines the labels, user concern, and retrieved product context.

The default model is configured with `SKINCARIA_MODEL`, falling back to `gemma4:e4b`.

## Dataset Sources

The skin-problem perception dataset is based on the Roboflow Universe project:

- Original dataset: https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean3/browse
- Annotated Skincaria fork/version: https://app.roboflow.com/riset-lqcum/skincaria-dataset/train

The Skincaria Roboflow version currently includes 447 additional annotated images. The target perception labels are:

- Acne
- Blackheads
- Dark-Spots
- Dry-Skin
- Enlarged-Pores
- Eyebags
- Oily-Skin
- Skin-Redness
- Whiteheads
- Wrinkles

## Running Locally

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the app:

```powershell
.\.venv\Scripts\python.exe main.py
```

Open:

- Image tester: http://localhost:8000
- Full app: http://localhost:8000/app

Optional startup checks:

```powershell
$env:SKINCARIA_STARTUP_CHECKS = "1"
.\.venv\Scripts\python.exe main.py
```

## Dataset Notes

The Roboflow segmentation work should be treated as the perception layer, not the final reasoning engine. The reasoning engine should consume structured perception evidence such as class, confidence, mask area, affected facial zones, and uncertainty flags, then produce skincare advice with safety limits and no medical diagnosis claims.
