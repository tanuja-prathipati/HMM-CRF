import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

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

# ========== Model Training ==========

def train_hmms(data, labels, n_states=2, n_iter=100, tol=1e-2):
    """Train separate HMM models for each word."""
    le = LabelEncoder()
    y = le.fit_transform(labels)

    models = {}
    for class_label in np.unique(y):
        word = le.inverse_transform([class_label])[0]
        word_data = [data[i] for i in range(len(data)) if y[i] == class_label]
        
        # Combine sequences for training
        X = np.concatenate(word_data, axis=0)
        lengths = [len(seq) for seq in word_data]

        # Train an HMM for the current word
        model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag", 
            n_iter=n_iter, 
            tol=tol
        )
        model.fit(X, lengths)
        models[word] = model

    return models, le

# ========== Speech Recognition ==========

def recognize_speech(models, le, audio_file):
    """Recognize speech using trained HMM models."""
    mfcc = extract_mfcc(audio_file)
    scores = {word: model.score(mfcc) for word, model in models.items()}
    return max(scores, key=scores.get)

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

def evaluate(models, le, test_data, true_labels):
    """Evaluate model accuracy on the test set."""
    predictions = []
    for mfcc in test_data:
        scores = {word: model.score(mfcc) for word, model in models.items()}
        predicted_word = max(scores, key=scores.get)
        predicted_label = le.transform([predicted_word])[0]
        predictions.append(predicted_label)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

def compute_auc_metrics(models, le, test_data, true_labels):
    """Compute AUC-ROC and AUC-PRC for the recognition system."""
    true_labels_one_hot = np.eye(len(le.classes_))[true_labels]  # One-hot encoding for multi-class AUC
    scores_matrix = []

    for mfcc in test_data:
        scores = np.array([model.score(mfcc) for model in models.values()])
        normalized_scores = scores - np.max(scores)  # Numerical stability
        scores_exp = np.exp(normalized_scores)
        probabilities = scores_exp / np.sum(scores_exp)  # Softmax
        scores_matrix.append(probabilities)

    scores_matrix = np.array(scores_matrix)

    # AUC-ROC and AUC-PRC for each class
    roc_auc = []
    prc_auc = []

    for i, class_name in enumerate(le.classes_):
        roc_auc.append(roc_auc_score(true_labels_one_hot[:, i], scores_matrix[:, i]))
        prc_auc.append(average_precision_score(true_labels_one_hot[:, i], scores_matrix[:, i]))

        # Plot ROC curve for this class
        fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], scores_matrix[:, i])
        plt.plot(fpr, tpr, label=f"Class {class_name} (AUC={roc_auc[-1]:.2f})")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Plot Precision-Recall curve
    for i, class_name in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(true_labels_one_hot[:, i], scores_matrix[:, i])
        plt.plot(recall, precision, label=f"Class {class_name} (AUC={prc_auc[-1]:.2f})")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # Average metrics
    avg_roc_auc = np.mean(roc_auc)
    avg_prc_auc = np.mean(prc_auc)

    return avg_roc_auc, avg_prc_auc

# ========== Main Execution ==========

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "train_data"
    TEST_DATASET_PATH = "test_data"

    print("Loading training data...")
    train_data, train_labels = load_data_from_directories(TRAIN_DATASET_PATH)
    print(f"Loaded {len(train_data)} training samples.")

    print("Training HMM models...")
    models, le = train_hmms(train_data, train_labels)
    print("Models trained successfully!")

    print("Loading test data...")
    test_data, true_labels = load_test_data(TEST_DATASET_PATH, le)
    print(f"Loaded {len(test_data)} test samples.")

    print("Evaluating model...")
    accuracy = evaluate(models, le, test_data, true_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    print("Computing AUC metrics...")
    avg_roc_auc, avg_prc_auc = compute_auc_metrics(models, le, test_data, true_labels)
    print(f"Average AUC-ROC: {avg_roc_auc:.2f}")
    print(f"Average AUC-PRC: {avg_prc_auc:.2f}")

    # Test on a single audio file
    test_audio_file = "test_data/cat/43.wav"  # Replace with a valid file path
    print("Recognizing single test audio...")
    recognized_word = recognize_speech(models, le, test_audio_file)
    print(f"Recognized word: {recognized_word}")
