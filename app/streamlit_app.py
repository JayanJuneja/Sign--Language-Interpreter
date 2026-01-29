"""
Real-time sign language captioning with Streamlit UI.
Compatible with MediaPipe 0.10.32+
"""
import streamlit as st
import cv2
import numpy as np
import pickle
import json
import os
from collections import deque
import pyttsx3
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_features_simple(hand_landmarks_list):
    """Extract features from hand landmarks."""
    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        return np.zeros(63)
    
    # Get first hand
    hand_landmarks = hand_landmarks_list[0]
    
    # Extract landmark coordinates
    landmarks = []
    for lm in hand_landmarks:
        landmarks.append([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    # Normalize
    centroid = np.mean(landmarks, axis=0)
    centered = landmarks - centroid
    distances = np.linalg.norm(centered, axis=1)
    scale = np.max(distances)
    if scale < 1e-6:
        scale = 1.0
    normalized = centered / scale
    
    # Flatten to feature vector
    features = normalized.flatten()
    return features


def draw_landmarks_on_image(bgr_image, detection_result):
    """Draw hand landmarks on image."""
    if not detection_result.hand_landmarks:
        return bgr_image
    
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(bgr_image)
    
    # Loop through detected hands
    for hand_landmarks in hand_landmarks_list:
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * bgr_image.shape[1])
            y = int(landmark.y * bgr_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * bgr_image.shape[1]), int(start.y * bgr_image.shape[0]))
            end_point = (int(end.x * bgr_image.shape[1]), int(end.y * bgr_image.shape[0]))
            
            cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)
    
    return annotated_image


class SignLanguageCaptioner:
    def __init__(self, model_path, label_map_path):
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load label mapping
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            self.idx_to_label = {int(k): v for k, v in label_data['idx_to_label'].items()}
        
        # Download hand landmarker model if needed
        import urllib.request
        landmarker_path = 'hand_landmarker.task'
        
        if not os.path.exists(landmarker_path):
            st.info("Downloading hand detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, landmarker_path)
        
        # Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Stability tracking
        self.prediction_buffer = deque(maxlen=30)
        self.last_committed_label = None
        
    def predict(self, frame):
        """
        Predict sign from frame.
        
        Returns:
            (label, confidence, annotated_frame)
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)
        
        # Draw landmarks
        annotated_frame = draw_landmarks_on_image(frame, detection_result)
        
        if detection_result.hand_landmarks:
            # Extract features
            features = extract_features_simple(detection_result.hand_landmarks)
            features = features.reshape(1, -1)
            
            # Predict
            pred_idx = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            confidence = pred_proba[pred_idx]
            label = self.idx_to_label[pred_idx]
            
            return label, confidence, annotated_frame
        else:
            return None, 0.0, annotated_frame
    
    def check_stability(self, label, confidence, threshold, stability_frames):
        """
        Check if prediction is stable enough to commit.
        
        Returns:
            stable_label or None
        """
        if label is None or confidence < threshold:
            self.prediction_buffer.clear()
            return None
        
        # Add to buffer
        self.prediction_buffer.append(label)
        
        # Check if we have enough frames
        if len(self.prediction_buffer) < stability_frames:
            return None
        
        # Check if all recent predictions are the same
        recent_predictions = list(self.prediction_buffer)[-stability_frames:]
        if len(set(recent_predictions)) == 1:
            stable_label = recent_predictions[0]
            
            # Only commit if different from last
            if stable_label != self.last_committed_label:
                self.last_committed_label = stable_label
                self.prediction_buffer.clear()
                return stable_label
        
        return None


def text_to_speech_async(text):
    """Run text-to-speech in a separate thread to avoid blocking."""
    def speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()


def main():
    st.set_page_config(
        page_title="Sign Language Captioner",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("welocme to JJ's First Real-Time Sign Language Captioner")
    
    # Check if model exists
    model_path = 'models/demo_model.pkl'
    label_map_path = 'models/label_map.json'
    
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        st.error("❌ Model not found!")
        st.info("Please train a model first by running: `python src/train_demo.py`")
        st.stop()
    
    # Initialize captioner
    if 'captioner' not in st.session_state:
        st.session_state.captioner = SignLanguageCaptioner(model_path, label_map_path)
    
    # Initialize transcript
    if 'transcript' not in st.session_state:
        st.session_state.transcript = []
    
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Column 1: Controls
    with col1:
        st.subheader("⚙️ Controls")
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["Alphabet", "Words"],
            index=0
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        # Stability frames
        stability_frames = st.slider(
            "Stability Frames",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Number of consecutive frames with same prediction to commit"
        )
        
        st.divider()
        
        # Action buttons
        if st.button("🗑️ Clear Transcript", use_container_width=True):
            st.session_state.transcript = []
            st.session_state.captioner.last_committed_label = None
            st.rerun()
        
        if st.button("🔊 Speak Transcript", use_container_width=True):
            if st.session_state.transcript:
                text = " ".join(st.session_state.transcript)
                text_to_speech_async(text)
                st.success("Speaking...")
            else:
                st.warning("Transcript is empty")
    
    # Column 2: Video feed
    with col2:
        st.subheader("📹 Live Feed")
        
        video_placeholder = st.empty()
        caption_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        # Start/Stop button
        if 'running' not in st.session_state:
            st.session_state.running = False
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Start Camera", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        with col_b:
            if st.button("⏸️ Stop Camera", use_container_width=True):
                st.session_state.running = False
                st.rerun()
    
    # Column 3: Transcript
    with col3:
        st.subheader("📝 Transcript")
        
        transcript_text = st.text_area(
            "Recognized Signs",
            value=" ".join(st.session_state.transcript),
            height=200,
            disabled=True
        )
        
        if st.session_state.transcript:
            st.info(f"Last: **{st.session_state.transcript[-1]}**")
        else:
            st.info("No signs detected yet")
    
    # Video capture loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not open webcam")
            st.session_state.running = False
        else:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Could not read frame")
                    break
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                
                # Predict
                label, confidence, annotated_frame = st.session_state.captioner.predict(frame)
                
                # Check stability
                stable_label = st.session_state.captioner.check_stability(
                    label, confidence, confidence_threshold, stability_frames
                )
                
                # Add to transcript if stable
                if stable_label is not None:
                    st.session_state.transcript.append(stable_label)
                    st.rerun()
                
                # Display
                video_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )
                
                if label is not None:
                    caption_placeholder.markdown(f"### Predicted: **{label}**")
                    
                    if confidence >= confidence_threshold:
                        confidence_placeholder.success(f"Confidence: {confidence:.2%}")
                    else:
                        confidence_placeholder.warning(f"Confidence: {confidence:.2%} (too low)")
                else:
                    caption_placeholder.markdown("### No hand detected")
                    confidence_placeholder.info("Show your hand to the camera")
            
            cap.release()


if __name__ == "__main__":
    main()