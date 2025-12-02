import numpy as np
from sklearn.ensemble import RandomForestClassifier

from transform import stft


def extract_features(stft_mag_db: np.ndarray) -> np.ndarray:
    """Extract features from STFT for classification."""
    features = []

    # Statistical features
    features.append(np.mean(stft_mag_db))  # Mean energy
    features.append(np.var(stft_mag_db))  # Variance
    features.append(np.std(stft_mag_db))  # Standard deviation
    features.append(np.max(stft_mag_db))  # Max value
    features.append(np.min(stft_mag_db))  # Min value
    features.append(np.median(stft_mag_db))  # Median

    # Spectral features (across frequency bins)
    freq_means = np.mean(stft_mag_db, axis=1)
    features.append(np.mean(freq_means))
    features.append(np.var(freq_means))
    features.append(np.max(freq_means))

    # Temporal features (across time frames)
    time_means = np.mean(stft_mag_db, axis=0)
    features.append(np.mean(time_means))
    features.append(np.var(time_means))
    features.append(np.max(time_means))

    # Energy concentration
    threshold = np.mean(stft_mag_db) + np.std(stft_mag_db)
    features.append(np.sum(stft_mag_db > threshold))

    return np.array(features)


class RandomForest:
    def __init__(self, n_estimators=100, random_state=42, max_depth=10):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, max_depth=max_depth
        )

    def train(self, train_positive, train_negative):
        """Train the random forest classifier."""
        # Extract features from training samples
        X_train = []
        y_train = []

        for sample in train_positive:
            stft_rep = stft(sample)
            features = extract_features(stft_rep)
            X_train.append(features)
            y_train.append(1)

        for sample in train_negative:
            stft_rep = stft(sample)
            features = extract_features(stft_rep)
            X_train.append(features)
            y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.clf.fit(X_train, y_train)
        self.trained = True

    def detect(self, sample):
        """Detect if the STFT represents a bubble."""
        stft_rep = stft(sample)
        features = extract_features(stft_rep)
        prediction = self.clf.predict([features])[0]
        return bool(prediction)
