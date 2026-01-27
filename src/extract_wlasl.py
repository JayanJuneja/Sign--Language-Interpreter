"""
Extract hand landmarks from WLASL videos and save as sequences.
Compatible with MediaPipe 0.10.30+
"""
import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from mediapipe import solutions
    mp_hands = solutions.hands
except (ImportError, AttributeError):
    import mediapipe as mp
    mp_hands = mp.solutions.hands


def extract_landmarks_from_video(video_path, hands):
    """
    Extract hand landmarks from all frames of a video.
    
    Args:
        video_path: Path to video file
        hands: MediaPipe Hands object
        
    Returns:
        numpy array of shape (num_frames, 21, 3) or None if no hands detected
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    landmarks_sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Take first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            landmarks_sequence.append(landmarks)
        else:
            # No hand detected in this frame - skip or use zeros
            # For now, we skip frames without hands
            pass
    
    cap.release()
    
    if len(landmarks_sequence) == 0:
        return None
    
    return np.array(landmarks_sequence)


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks from WLASL videos')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Path to directory containing WLASL videos')
    parser.add_argument('--labels_json', type=str, required=True,
                        help='Path to JSON file mapping video_id to label')
    parser.add_argument('--output_dir', type=str, default='data/wlasl_landmarks',
                        help='Output directory for landmark sequences')
    parser.add_argument('--max_videos_per_class', type=int, default=50,
                        help='Maximum number of videos to process per class')
    parser.add_argument('--video_ext', type=str, default='.mp4',
                        help='Video file extension')
    
    args = parser.parse_args()
    
    # Load labels
    print(f"Loading labels from {args.labels_json}...")
    with open(args.labels_json, 'r') as f:
        labels_data = json.load(f)
    
    print(f"Found {len(labels_data)} video-label mappings")
    
    # Setup MediaPipe
    print("Initializing MediaPipe Hands...")
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Group videos by label
    label_to_videos = {}
    for video_id, label in labels_data.items():
        if label not in label_to_videos:
            label_to_videos[label] = []
        label_to_videos[label].append(video_id)
    
    print(f"\nFound {len(label_to_videos)} unique labels")
    
    # Process videos
    total_processed = 0
    total_failed = 0
    stats_per_class = {}
    
    for label, video_ids in label_to_videos.items():
        print(f"\n{'='*60}")
        print(f"Processing label: {label}")
        print(f"Total videos available: {len(video_ids)}")
        
        # Limit videos per class
        video_ids = video_ids[:args.max_videos_per_class]
        print(f"Processing: {len(video_ids)} videos")
        
        # Create label directory
        label_dir = os.path.join(args.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        processed = 0
        failed = 0
        
        for video_id in tqdm(video_ids, desc=f"  {label}"):
            # Find video file
            video_path = os.path.join(args.videos_dir, f"{video_id}{args.video_ext}")
            
            if not os.path.exists(video_path):
                failed += 1
                continue
            
            # Extract landmarks
            landmarks_seq = extract_landmarks_from_video(video_path, hands)
            
            if landmarks_seq is None:
                failed += 1
                continue
            
            # Save sequence
            output_path = os.path.join(label_dir, f"{video_id}.npy")
            np.save(output_path, landmarks_seq)
            
            processed += 1
        
        stats_per_class[label] = processed
        total_processed += processed
        total_failed += failed
        
        print(f"  ✓ Saved: {processed}")
        print(f"  ✗ Failed: {failed}")
    
    # Cleanup
    hands.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos processed: {total_processed}")
    print(f"Total videos failed: {total_failed}")
    print(f"\nSequences per class:")
    for label, count in sorted(stats_per_class.items()):
        print(f"  {label}: {count}")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
