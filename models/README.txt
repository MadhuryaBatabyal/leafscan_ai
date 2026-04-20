This folder must contain the trained model weights (.pth files) to run LeafScan AI.

Required files:
  stage1_quality.pth   (optional — skip if Stage 1 not trained)
  stage2_dryness.pth
  stage3_disease.pth
  stage4_pest.pth

These files are not committed to git because they are too large (20–90 MB each).

To get the model files:
  - Download from Google Drive: [paste your Drive folder link here]
  - OR train them yourself using the Colab notebooks in the repo

Place all .pth files directly in this models/ folder before running:
  streamlit run app.py
