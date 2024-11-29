from hmm import load_data_from_directories, extract_mfcc, train_hmms, recognize_speech as hmm_recognize
from sklearn_crfsuite import CRF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Paths to your training and test data folders
TRAIN_DATASET_PATH = 'train_data'
TEST_DATASET_PATH = 'test_data'

# ========== CRF Utility Functions ==========

def prepare_crf_features(data):
    """Prepare features for CRF from MFCC data."""
    feature_list = []
    for mfcc in data:
        features = [{"mfcc_" + str(i): frame[i] for i in range(len(frame))} for frame in mfcc]
        feature_list.append(features)
    return feature_list

def sequence_to_feature_vector(data):
    """Convert each sequence into a single feature vector (e.g., mean of all frames)."""
    feature_vectors = []
    for mfcc in data:
        feature_vectors.append(np.mean(mfcc, axis=0))  # Take the mean across frames
    return np.array(feature_vectors)

def train_crf_and_classifier(data, labels):
    """Train CRF for feature extraction and logistic regression for sequence classification."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Prepare features for CRF
    crf_features = prepare_crf_features(data)
    crf_model = CRF(algorithm='lbfgs', max_iterations=100)
    crf_model.fit(crf_features, [[str(y[i])] * len(data[i]) for i in range(len(data))])

    # Extract refined features (mean of CRF scores)
    refined_features = sequence_to_feature_vector(data)
    
    # Train logistic regression on refined features
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(refined_features, y)

    return crf_model, classifier, le

def evaluate_crf(classifier, le, test_data, true_labels):
    """Evaluate CRF model accuracy, precision, recall, and F1 score."""
    feature_vectors = sequence_to_feature_vector(test_data)
    predictions = classifier.predict(feature_vectors)

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=le.classes_)
    
    print("CRF Model Evaluation:")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

# ========== Main Execution ==========

if __name__ == "__main__":
    print("Loading training data...")
    train_data, train_labels = load_data_from_directories(TRAIN_DATASET_PATH)
    print(f"Loaded {len(train_data)} training samples.")

    print("Training HMM models...")
    hmm_models, hmm_le = train_hmms(train_data, train_labels)
    print("HMM models trained successfully!")

    print("Training CRF and logistic regression...")
    crf_model, crf_classifier, crf_le = train_crf_and_classifier(train_data, train_labels)
    print("CRF models trained successfully!")

    print("Loading and evaluating test data...")
test_data, true_labels = load_data_from_directories(TEST_DATASET_PATH)
print(f"Loaded {len(test_data)} test samples.")

# Encode true labels for HMM evaluation
true_labels = hmm_le.transform(true_labels)

# Evaluate HMM models
print("Evaluating HMM model...")
hmm_predictions = []
for mfcc in test_data:
    scores = {word: model.score(mfcc) for word, model in hmm_models.items()}
    predicted_word = max(scores, key=scores.get)
    predicted_label = hmm_le.transform([predicted_word])[0]
    hmm_predictions.append(predicted_label)

hmm_accuracy = accuracy_score(true_labels, hmm_predictions)
hmm_report = classification_report(true_labels, hmm_predictions, target_names=hmm_le.classes_)

print("HMM Model Evaluation:")
print(f"Model Accuracy: {hmm_accuracy * 100:.2f}%")
print("Classification Report:\n", hmm_report)

# Evaluate CRF model
evaluate_crf(crf_classifier, crf_le, test_data, true_labels)
