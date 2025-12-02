from sklearn.metrics import precision_recall_fscore_support

from detectors.random_forest import RandomForest


class Constant:
    def train(self, train_positive, train_negative):
        pass

    def detect(self, sample):
        return True


class BubbleDetector:
    """A bubble detector superclass."""

    detectors = {
        "constant_true": Constant,
        "random_forest": RandomForest,
    }

    def __init__(self, model_name: str, parameters: dict = {}):
        self.model = self.detectors[model_name](**parameters)
        self.name = model_name

    def train(self, train_positive, train_negative):
        """Train the bubble detector using positive and negative samples."""
        return self.model.train(train_positive, train_negative)

    def detect(self, sample):
        """Detect bubbles in the STFT representation."""
        return self.model.detect(sample)

    def evaluate(self, positive, negative, to_stdout=True):
        """Evaluate the bubble detector on the test set."""
        samples = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)

        predictions = []
        for sample in samples:
            prediction = 1 if self.detect(sample) else 0
            predictions.append(prediction)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )

        if to_stdout:
            print(f"{'='*50}")
            print(self.name)
            print(f"{'='*50}")
            print(
                f"Precision: {precision:.3f}"
                f" ({precision*100:.0f}% of detections were real bubbles)"
            )
            print(
                f"Recall:    {recall:.3f}"
                f" ({recall*100:.0f}% of actual bubbles were detected)"
            )
            print(f"F1-Score:  {f1:.3f}")
            print(f"{'='*50}")
        return precision, recall, f1
