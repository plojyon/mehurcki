import numpy as np
from sklearn.linear_model import LogisticRegression

from sample import get_sample


class LogisticRegressionDetector:
    """A bubble detector using logistic regression."""

    def __init__(self, threshold: float = 0.5, max_iter: int = 1000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.threshold = threshold

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pos = []
        neg = []

        for interval in positive_intervals:
            pos.append(get_sample(data, interval))
        for interval in negative_intervals:
            neg.append(get_sample(data, interval))

        X_train = np.array(pos + neg)
        y_train = np.array([1] * len(pos) + [0] * len(neg))
        self.model.fit(X_train, y_train)

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        predictions = []
        for interval in intervals:
            prediction = self.model.predict([get_sample(data, interval)])[0]
            predictions.append(prediction >= self.threshold)
        return predictions
