"""
Data collection script for sign language gestures.
Opens webcam and saves hand landmark features when user presses a key.
Compatible with MediaPipe 0.10.32+
"""
import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Label modes
ALPHABET_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
WORD_LABELS = [
    'HELLO', 'YES', 'NO', 'THANK_YOU', 'PLEASE', 
    'SORRY', 'HELP', 'STOP', 'LOVE', 'GOOD', 
    'BAD', 'WATER', 'EAT', 'MORE', 'DONE', 
    'GO', 'COME', 'WANT', 'NAME', 'FRIEND', 'FUCK YOU Too!'
]

# Key mappings for words
WORD_KEY_MAP = {
    '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
    '6': 5, '7': 6, '8': 7, '9': 8, '0': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14,
    'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19
}


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


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on image."""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through detected hands
    for hand_landmarks in hand_landmarks_list:
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
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
            
            start_point = (int(start.x * rgb_image.shape[1]), int(start.y * rgb_image.shape[0]))
            end_point = (int(end.x * rgb_image.shape[1]), int(end.y * rgb_image.shape[0]))
            
            cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)
    
    return annotated_image


def main():
    # Download hand landmarker model
    import urllib.request
    model_path = 'hand_landmarker.task'
    
    if not os.path.exists(model_path):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please check your internet connection and try again.")
            return
    
    # Setup MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/demo_landmarks.csv'
    
    # Load existing data
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        samples_collected = len(df)
        print(f"Loaded existing data: {samples_collected} samples")
    else:
        df = pd.DataFrame()
        samples_collected = 0
    
    # Mode selection
    print("\n=== SIGN LANGUAGE DATA COLLECTION ===")
    print("Select mode:")
    print("1. Alphabet mode (A-Z)")
    print("2. Word mode (20 common words)")
    mode_choice = input("Enter 1 or 2: ").strip()
    
    if mode_choice == '1':
        mode = 'alphabet'
        labels = ALPHABET_LABELS
        print("\nAlphabet mode selected")
        print("Press A-Z to save samples for each letter")
    else:
        mode = 'words'
        labels = WORD_LABELS
        print("\nWord mode selected")
        print("Available words:")
        for i, word in enumerate(WORD_LABELS):
            key = list(WORD_KEY_MAP.keys())[i]
            print(f"  {key}: {word}")
        print("\nPress the corresponding key to save a sample")
    
    print("\nPress 'q' to quit")
    print("Press 's' to show statistics")
    print("\n⚠️  IMPORTANT: Click on the camera window to activate it for keyboard input!")
    print("\n")
    
    # Create window
    window_name = 'Sign Language Data Collection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            detection_result = detector.detect(mp_image)
            
            # Draw landmarks
            if detection_result.hand_landmarks:
                annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
                frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Display instructions
            cv2.putText(frame, f"Mode: {mode.upper()}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detection_result.hand_landmarks:
                cv2.putText(frame, "Hand detected - Press key to save!", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add reminder to click window
            cv2.putText(frame, "Click this window first, then press keys!", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            # Handle key press - increase wait time for better key detection
            key = cv2.waitKey(10) & 0xFF
            
            if key == 255:  # No key pressed
                continue
            
            if  key == 27:  # removed 'q' or ESC
                print("\nQuitting...")
                break
            elif key == ord('s'):
                if len(df) > 0:
                    print("\n=== Statistics ===")
                    print(df['label'].value_counts().sort_index())
                    print(f"Total samples: {len(df)}")
                else:
                    print("No samples collected yet")
            else:
                # Check if valid label key pressed
                label = None
                
                if mode == 'alphabet':
                    char = chr(key).upper()
                    if char in ALPHABET_LABELS:
                        label = char
                else:
                    char = chr(key).lower()
                    if char in WORD_KEY_MAP:
                        idx = WORD_KEY_MAP[char]
                        label = WORD_LABELS[idx]
                
                # Save sample
                if label is not None:
                    if detection_result.hand_landmarks:
                        features = extract_features_simple(detection_result.hand_landmarks)
                        
                        feature_cols = [f'feature_{i}' for i in range(len(features))]
                        new_row = pd.DataFrame([list(features) + [label]], 
                                              columns=feature_cols + ['label'])
                        
                        df = pd.concat([df, new_row], ignore_index=True)
                        samples_collected += 1
                        
                        df.to_csv(csv_path, index=False)
                        print(f"✓ Saved sample for '{label}' (Total: {samples_collected})")
                    else:
                        print(f"✗ No hand detected - cannot save sample for '{label}'")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nData collection complete!")
        print(f"Total samples saved: {samples_collected}")
        print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
    main()