#  Sign Language Captioner

Real-time sign language recognition using MediaPipe Hands and machine learning. This project provides both a demo system for quick alphabet/word recognition and a research pipeline for processing the WLASL dataset.

## Features

- **Real-time captioning** with webcam using Streamlit UI
- **Two modes**: Alphabet (A-Z) and Words (20 common signs)
- **Text-to-speech** output of recognized signs
- **Stability filtering** to avoid false positives
- **WLASL preprocessing pipeline** for research applications
- **Windows-friendly** with no GPU requirement

## Project Structure

```
sign-captioner/
├── app/
│   └── streamlit_app.py          # Real-time captioning UI
├── src/
│   ├── collect_data.py            # Data collection via webcam
│   ├── features.py                # Landmark normalization & feature extraction
│   ├── train_demo.py              # Train demo classifier
│   ├── extract_wlasl.py           # Extract landmarks from WLASL videos
│   └── train_wlasl.py             # LSTM training stub (future)
├── data/
│   ├── demo_landmarks.csv         # Collected training samples
│   └── wlasl_landmarks/           # Processed WLASL sequences
├── models/
│   ├── demo_model.pkl             # Trained classifier
│   └── label_map.json             # Label mappings
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python

Ensure you have Python 3.8+ installed on Windows.

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PyTorch on Windows, install the CPU version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start: Demo System

### Step 1: Collect Training Data

Run the data collection script to capture hand landmarks for signs:

```bash
python src/collect_data.py
```

**Instructions:**
1. Select mode:
   - **Mode 1 (Alphabet)**: Press A-Z to save samples for each letter
   - **Mode 2 (Words)**: Press keys 1-9, 0, a-j for the 20 word labels
2. Position your hand in view of the webcam
3. Press the corresponding key when making the sign
4. Collect at least 20-30 samples per sign for good performance
5. Press 's' to show statistics
6. Press 'q' to quit

**Word labels and keys:**
```
1: HELLO       6: SORRY       a: BAD         f: COME
2: YES         7: HELP        b: WATER       g: WANT
3: NO          8: STOP        c: EAT         h: NAME
4: THANK_YOU   9: LOVE        d: MORE        i: FRIEND
5: PLEASE      0: GOOD        e: DONE        j: GO
```

### Step 2: Train the Model

Once you have collected sufficient data:

```bash
python src/train_demo.py
```

This will:
- Load your collected samples from `data/demo_landmarks.csv`
- Train a Random Forest classifier
- Display accuracy and confusion matrix
- Save the model to `models/demo_model.pkl`
- Save label mappings to `models/label_map.json`

### Step 3: Run Real-Time Captioner

Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

**UI Features:**
- **Left panel**: Controls for mode, confidence threshold, stability frames
- **Middle panel**: Live webcam feed with landmark visualization
- **Right panel**: Transcript of recognized signs

**Controls:**
- **Mode**: Switch between Alphabet and Words mode
- **Confidence Threshold**: Minimum confidence required (0.0-1.0)
- **Stability Frames**: Number of consecutive frames needed to commit a prediction
- **Clear Transcript**: Reset the transcript
- **Speak Transcript**: Use text-to-speech to read the transcript aloud

## WLASL Research Pipeline

For working with the WLASL video dataset:

### Step 1: Extract Landmarks from Videos

```bash
python src/extract_wlasl.py \
  --videos_dir path/to/wlasl/videos \
  --labels_json path/to/labels.json \
  --output_dir data/wlasl_landmarks \
  --max_videos_per_class 50
```

**Arguments:**
- `--videos_dir`: Directory containing WLASL video files
- `--labels_json`: JSON file mapping video IDs to labels (format: `{"video_id": "label"}`)
- `--output_dir`: Where to save extracted landmark sequences (default: `data/wlasl_landmarks`)
- `--max_videos_per_class`: Maximum videos to process per class (default: 50)
- `--video_ext`: Video file extension (default: `.mp4`)

This will:
- Process each video with MediaPipe Hands
- Extract hand landmarks for each frame
- Save sequences as `.npy` files organized by label
- Print statistics on processed videos

### Step 2: Train LSTM Model (Stub)

```bash
python src/train_wlasl.py --data_dir data/wlasl_landmarks
```

Currently, this is a stub that:
- Loads processed landmark sequences
- Shows dataset statistics
- Demonstrates LSTM model architecture
- Provides next steps for full implementation

## Tips for Best Performance

### Data Collection
- **Consistency**: Use the same lighting and background
- **Variety**: Collect samples with slight variations in hand position and angle
- **Quantity**: Aim for 30+ samples per sign
- **Balance**: Collect similar amounts for each sign

### Model Training
- More data = better accuracy
- Ensure balanced classes (similar number of samples for each sign)
- If accuracy is low, collect more diverse training data

### Real-Time Captioning
- **Lighting**: Ensure good lighting for hand detection
- **Background**: Plain backgrounds work best
- **Distance**: Keep hand at medium distance from camera
- **Confidence Threshold**: Lower for easier detection (0.5-0.7), higher for more precision (0.7-0.9)
- **Stability Frames**: Lower for faster response (5-10), higher for more stability (15-30)

## Troubleshooting

### Webcam Issues
- Ensure no other application is using the webcam
- Try changing camera index in code if default camera (0) doesn't work
- On Windows, grant camera permissions to Python

### MediaPipe Not Detecting Hands
- Improve lighting
- Move hand closer to camera
- Ensure hand is fully visible
- Try plain background

### Low Model Accuracy
- Collect more training samples (30+ per sign)
- Ensure consistent hand positioning during collection
- Balance your dataset (equal samples per sign)
- Try different signs that are more visually distinct

### Streamlit Issues
- If app doesn't start, ensure Streamlit is installed: `pip install streamlit`
- If video doesn't show, check webcam permissions
- If TTS doesn't work, ensure `pyttsx3` is installed correctly

## Future Enhancements

Potential improvements for this project:

1. **Two-hand support**: Modify MediaPipe to track both hands
2. **Temporal models**: Implement full LSTM training for dynamic gestures
3. **Data augmentation**: Add rotation, scaling, noise to training data
4. **Model optimization**: Try different classifiers (SVM, Neural Networks)
5. **Mobile deployment**: Create mobile app version
6. **Sentence construction**: Add grammar rules for natural language output
7. **Custom vocabulary**: Allow users to add their own signs

## Technical Details

### Feature Extraction
- **Landmarks**: 21 hand landmarks per frame (MediaPipe Hands)
- **Normalization**: Center and scale invariant
- **Feature vector**: 63 dimensions (21 landmarks × 3 coordinates)

### Models
- **Demo**: Random Forest classifier (100 estimators)
- **WLASL**: LSTM architecture (future implementation)

### Dependencies
- **Computer Vision**: OpenCV, MediaPipe
- **ML/DL**: scikit-learn, PyTorch
- **UI**: Streamlit
- **Audio**: pyttsx3

## Credits

- **MediaPipe Hands**: Google's hand tracking solution
- **WLASL Dataset**: Word-Level American Sign Language dataset
- Built with Python, OpenCV, MediaPipe, scikit-learn, PyTorch, and Streamlit

## License

This project is for educational purposes. Please respect the licenses of the underlying libraries and datasets.

---
