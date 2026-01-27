"""
Train a simple classifier on collected demo landmarks.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load data
    csv_path = 'data/demo_landmarks.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("Please run collect_data.py first to collect training samples")
        return
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Classes: {sorted(df['label'].unique())}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Check minimum samples per class
    class_counts = df['label'].value_counts()
    min_samples = class_counts.min()
    
    if min_samples < 2:
        print(f"\nWarning: Some classes have fewer than 2 samples")
        print("You may want to collect more data for better performance")
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"\nFeature shape: {X.shape}")
    
    # Create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert labels to indices
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Split data
    test_size = 0.2
    if len(df) < 10:
        test_size = 0.1  # Use smaller test set for very small datasets
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=42,
        stratify=y_encoded if min_samples >= 2 else None
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n=== Training Results ===")
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    
    print("\n=== Validation Results ===")
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(classification_report(y_val, y_val_pred, target_names=target_names, zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/demo_model.pkl'
    label_map_path = 'models/label_map.json'
    
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Saving label map to {label_map_path}...")
    with open(label_map_path, 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Label map saved to: {label_map_path}")
    print(f"\nYou can now run the Streamlit app with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
