import os
import numpy as np
import librosa
from sklearn_crfsuite import CRF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== Audio Processing ==========

def extract_mfcc(audio_file):
    """Extract MFCC features from a .wav file."""
    audio, sr = librosa.load(audio_file, sr=None)  # Load audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    return mfcc.T  # Transpose to have time frames as rows

def load_data_from_directories(dataset_path):
    """Load audio data from directories and extract MFCC features."""
    data = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):  # Process only directories
            for filename in os.listdir(label_path):
                if filename.endswith('.wav'):  # Process only .wav files
                    file_path = os.path.join(label_path, filename)
                    mfcc = extract_mfcc(file_path)
                    data.append(mfcc)
                    labels.append(label)  # Folder name is the label

    return data, labels

# ========== Feature Preparation for CRF ==========

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

# ========== Model Training ==========

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

# ========== Speech Recognition ==========

def recognize_speech(crf_model, classifier, le, audio_file):
    """Recognize speech using CRF and logistic regression."""
    mfcc = extract_mfcc(audio_file)
    feature_vector = np.mean(mfcc, axis=0).reshape(1, -1)  # Convert sequence to single feature vector
    predicted_label = classifier.predict(feature_vector)[0]
    return le.inverse_transform([predicted_label])[0]

# ========== Test Data and Evaluation ==========

def load_test_data(test_path, le):
    """Load test data and return MFCCs and encoded labels."""
    test_data = []
    true_labels = []

    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        if os.path.isdir(folder_path):  # Process only directories
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):  # Process only .wav files
                    file_path = os.path.join(folder_path, filename)
                    label = folder  # Folder name is the label
                    mfcc = extract_mfcc(file_path)
                    test_data.append(mfcc)
                    true_labels.append(label)

    return test_data, le.transform(true_labels)

def evaluate(classifier, le, test_data, true_labels):
    """Evaluate model accuracy on the test set."""
    feature_vectors = sequence_to_feature_vector(test_data)

    predictions = classifier.predict(feature_vectors)
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# ========== Main Execution ==========

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "train_data"
    TEST_DATASET_PATH = "test_data"

    print("Loading training data...")
    train_data, train_labels = load_data_from_directories(TRAIN_DATASET_PATH)
    print(f"Loaded {len(train_data)} training samples.")

    print("Training CRF and logistic regression...")
    crf_model, classifier, le = train_crf_and_classifier(train_data, train_labels)
    print("Models trained successfully!")

    print("Loading test data...")
    test_data, true_labels = load_test_data(TEST_DATASET_PATH, le)
    print(f"Loaded {len(test_data)} test samples.")

    print("Evaluating model...")
    accuracy = evaluate(classifier, le, test_data, true_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Test on a single audio file
    test_audio_file = "test_data/cat/43.wav"  # Replace with a valid file path
    print("Recognizing single test audio...")
    recognized_word = recognize_speech(crf_model, classifier, le, test_audio_file)
    print(f"Recognized word: {recognized_word}")
