import bisect
import random

import numpy as np

from load import BubbleAnnotation


def get_sample(data: np.ndarray, interval: BubbleAnnotation):
    """Extract sample from data given an interval."""
    s, e = interval.start, interval.end

    # handle edge cases
    if interval.end - interval.start == 0:
        if interval.end >= data.shape[1]:
            s -= 1
        else:
            e += 1

    sample = data[:, s:e]
    if sample.shape[0] <= 100:
        sample = sample.flatten()
    else:
        # sample = sample.mean(axis=1)
        sample = sample[:, 0]
    return sample


def invert_intervals(intervals: BubbleAnnotation, total_length: int):
    """Invert a list of intervals."""
    inverted = []
    prev_end = 0
    for start, end in intervals:
        if start > prev_end:  # handle zero-width gaps
            inverted.append((prev_end, start))

        prev_end = end

    if prev_end < total_length:  # final interval
        inverted.append((prev_end, total_length))

    return inverted


def sample_within_annotations(
    audio: np.ndarray,
    intervals: list[BubbleAnnotation],
    window_size: int,
    count: int,
) -> list[BubbleAnnotation]:
    """Sample fixed-size windows within given intervals."""
    legal_sampling_ranges = []
    if not intervals:
        print("No annotations to sample from! Wtf?")
        return []
    for start, end in intervals:
        if end - start >= window_size:
            legal_sampling_ranges.append((start, end - window_size))
    if not legal_sampling_ranges:
        print(f"Cannot sample anything with window size {window_size}!")
        print(f"Largest annotation size: {max((end - start) for start, end in intervals)}")
        return []

    # Build cumulative lengths
    lengths = []
    total = 0
    for start, end in legal_sampling_ranges:
        total += end - start
        lengths.append(total)

    samples = []
    for _ in range(count):
        r = random.randint(0, total)

        # find the sampled range
        idx = bisect.bisect_left(lengths, r)
        start, end = legal_sampling_ranges[idx]

        # calculate offset
        prev = lengths[idx - 1] if idx > 0 else 0
        samples.append(start + (r - prev))

    return [BubbleAnnotation(start=sample, end=sample + window_size) for sample in samples]


def sample_training_data(
    audio: np.ndarray,
    annotations: list[BubbleAnnotation],
    window_size: int,  # in samples
    count_positive: int,
    count_negative: int,
) -> tuple[list[BubbleAnnotation], list[BubbleAnnotation]]:
    """Extract fixed-size windows from audio.

    Windows have either full or no overlap with bubbles.
    """
    positive_windows = sample_within_annotations(
        audio, annotations, window_size, count_positive
    )

    inverted_annotations = invert_intervals(annotations, len(audio))
    negative_windows = sample_within_annotations(
        audio, inverted_annotations, window_size, count_negative
    )

    return positive_windows, negative_windows
