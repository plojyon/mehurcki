import os

import matplotlib.pyplot as plt
from fire import Fire

from detect import BubbleDetector
from load import load_annotations, load_wav
from plot import plot_annotations, plot_wav
from transform import sample_training_data

SAMPLE_RATE = 44100
WINDOW_SIZE = 0.3  # seconds
TRAIN_RATIO = 0.7

annotations = load_annotations("data/annotations.json")
wav_files = [f for f in os.listdir("data/") if f.endswith(".wav")]


def prepare_data():
    print(f"Found {len(wav_files)} wav files")

    positive_samples = []
    negative_samples = []

    for file_name in wav_files:
        _, audio = load_wav(f"data/{file_name}")
        audio = audio.astype(float)

        if file_name in annotations:
            pos, neg = sample_training_data(
                audio,
                annotations[file_name],
                window_size=int(WINDOW_SIZE * SAMPLE_RATE),
                count_positive=30,
                count_negative=30,
            )
            positive_samples.extend(pos)
            negative_samples.extend(neg)

    # Split into train and test sets
    n_pos_train = int(TRAIN_RATIO * len(positive_samples))
    n_neg_train = int(TRAIN_RATIO * len(negative_samples))

    train_positive = positive_samples[:n_pos_train]
    test_positive = positive_samples[n_pos_train:]

    train_negative = negative_samples[:n_neg_train]
    test_negative = negative_samples[n_neg_train:]

    print("Total samples: ", end="")
    print(f"{len(positive_samples)} positive, ", end="")
    print(f"{len(negative_samples)} negative")

    print("Training samples: ", end="")
    print(f"{len(train_positive)} positive, ", end="")
    print(f"{len(train_negative)} negative")

    print("Testing samples: ", end="")
    print(f"{len(test_positive)} positive, ", end="")
    print(f"{len(test_negative)} negative")

    return train_positive, train_negative, test_positive, test_negative


def train_detector(
    model: str,
    train_positive,
    train_negative,
    test_positive,
    test_negative,
):
    detector = BubbleDetector(model)
    detector.train(
        train_positive=train_positive,
        train_negative=train_negative,
    )
    detector.evaluate(positive=test_positive, negative=test_negative)


def visualize_waveform(file_name):
    t, audio = load_wav(file_name)
    audio = audio.astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_wav(ax, t, audio, title=file_name)

    base_name = os.path.basename(file_name)
    if base_name in annotations:
        plot_annotations(ax, annotations[base_name], audio)

    plt.show()


class Main:
    def visualize_waveform(self, file_name: str):
        visualize_waveform(file_name)

    def train_detector(self, name="all"):
        data = prepare_data()

        available_models = BubbleDetector.detectors.keys()
        if name == "all":
            for model_name in available_models:
                train_detector(model_name, *data)
        else:
            if name not in available_models:
                raise ValueError(f"Available models: {list(available_models)}")
            train_detector(name, *data)


if __name__ == "__main__":
    Fire(Main)
