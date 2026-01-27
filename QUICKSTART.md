# Quick Reference Guide

## Initial Setup (One-time)

```bash
# Option 1: Automatic setup (Windows)
setup.bat

# Option 2: Manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Demo Workflow

### 1. Collect Data
```bash
python src/collect_data.py
```
- Select mode (1=Alphabet, 2=Words)
- Press keys to save samples
- Collect 20-30 samples per sign
- Press 'q' to quit

### 2. Train Model
```bash
python src/train_demo.py
```
- Trains on collected data
- Saves model to `models/`

### 3. Run App
```bash
streamlit run app/streamlit_app.py
```
- Opens browser with UI
- Click "Start Camera"
- Adjust settings as needed

## WLASL Workflow

### 1. Extract Landmarks
```bash
python src/extract_wlasl.py ^
  --videos_dir C:\path\to\videos ^
  --labels_json data\wlasl_labels.json ^
  --output_dir data\wlasl_landmarks ^
  --max_videos_per_class 50
```

### 2. Check Data
```bash
python src/train_wlasl.py --data_dir data/wlasl_landmarks
```

## File Locations

- Training data: `data/demo_landmarks.csv`
- Trained model: `models/demo_model.pkl`
- Label mapping: `models/label_map.json`
- WLASL sequences: `data/wlasl_landmarks/<label>/<video_id>.npy`

## Troubleshooting

### Camera not opening
```python
# Try different camera index in code
cap = cv2.VideoCapture(1)  # Change 0 to 1, 2, etc.
```

### No hands detected
- Improve lighting
- Plain background
- Hand closer to camera
- Ensure full hand visible

### Low accuracy
- Collect more samples (30+)
- Balance dataset
- Use consistent lighting
- Try more distinct signs

## Key Settings

**Confidence Threshold**
- 0.5-0.7: More sensitive, faster response
- 0.7-0.9: More precise, fewer false positives

**Stability Frames**
- 5-10: Faster, less stable
- 15-30: Slower, more stable
