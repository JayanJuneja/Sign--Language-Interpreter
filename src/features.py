"""
Feature extraction and normalization utilities for hand landmarks.
"""
import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks to be translation and scale invariant.
    
    Args:
        landmarks: Array of shape (21, 3) containing x, y, z coordinates
        
    Returns:
        Normalized landmarks array of shape (21, 3)
    """
    if landmarks is None or len(landmarks) == 0:
        return None
    
    landmarks = np.array(landmarks)
    
    # Calculate centroid (wrist is landmark 0, but we use all points)
    centroid = np.mean(landmarks, axis=0)
    
    # Center the landmarks
    centered = landmarks - centroid
    
    # Calculate scale (max distance from centroid)
    distances = np.linalg.norm(centered, axis=1)
    scale = np.max(distances)
    
    # Avoid division by zero
    if scale < 1e-6:
        scale = 1.0
    
    # Scale normalize
    normalized = centered / scale
    
    return normalized


def landmarks_to_feature_vector(landmarks):
    """
    Convert normalized landmarks to a fixed-length feature vector.
    
    Args:
        landmarks: Array of shape (21, 3) containing normalized x, y, z coordinates
        
    Returns:
        Feature vector of shape (63,) - flattened x, y, z for all 21 points
    """
    if landmarks is None:
        # Return zeros if no hand detected
        return np.zeros(63)
    
    # Flatten to 1D vector
    features = landmarks.flatten()
    
    return features


def extract_features_from_mediapipe(hand_landmarks):
    """
    Extract features from MediaPipe hand landmarks object.
    
    Args:
        hand_landmarks: MediaPipe hand_landmarks object
        
    Returns:
        Feature vector of shape (63,) or None if no hand detected
    """
    if hand_landmarks is None:
        return np.zeros(63)
    
    # Extract landmark coordinates
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    # Normalize
    normalized = normalize_landmarks(landmarks)
    
    # Convert to feature vector
    features = landmarks_to_feature_vector(normalized)
    
    return features


def get_feature_dimension():
    """
    Get the dimension of the feature vector.
    
    Returns:
        int: Feature dimension (63 for 21 landmarks × 3 coordinates)
    """
    return 63
