import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Define consistent model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# Load saved embeddings and labels
embeddings_path = os.path.join(MODEL_DIR, 'embeddings.npy')
labels_path = os.path.join(MODEL_DIR, 'labels.npy')

if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("❌ Embeddings or labels not found! Run extract_embeddings.py first.")

embeddings = np.load(embeddings_path)
labels = np.load(labels_path)

# Encode string labels to integers
le = LabelEncoder()
integer_labels = le.fit_transform(labels)

# Train SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, integer_labels)

# Save classifier and label encoder
joblib.dump(classifier, os.path.join(MODEL_DIR, 'svm_classifier.joblib'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

print("✅ Classifier training complete and models saved.")
