"""
Real-time sign language captioning with Streamlit UI.
"""
import streamlit as st
import cv2
import numpy as np
import pickle
import json
import os
import sys
from collections import deque
import pyttsx3
import threading

# Import MediaPipe
import mediapipe as mp
from mediapipe import solutions


def extract_features_simple(hand_landmarks):
    """Extract features from hand landmarks."""
    if hand_landmarks is None:
        return np.zeros(63)
    
    # Extract landmark coordinates
    landmarks = []
    for lm in hand_landmarks.landmark:
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


class SignLanguageCaptioner:
    def __init__(self, model_path, label_map_path):
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load label mapping
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            self.idx_to_label = {int(k): v for k, v in label_data['idx_to_label'].items()}
        
        # Setup MediaPipe
        self.mp_hands = solutions.hands
        self.mp_drawing = solutions.drawing_utils
        self.mp_drawing_styles = solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract features
            hand_landmarks = results.multi_hand_landmarks[0]
            features = extract_features_simple(hand_landmarks)
            features = features.reshape(1, -1)
            
            # Predict
            pred_idx = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            confidence = pred_proba[pred_idx]
            label = self.idx_to_label[pred_idx]
            
            return label, confidence, frame
        else:
            return None, 0.0, frame
    
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
    
    st.title("🤟 Real-Time Sign Language Captioner")
    
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
        with col_b:
            if st.button("⏸️ Stop Camera", use_container_width=True):
                st.session_state.running = False
    
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