import numpy as np

# from src.sample import hann_window


def plot_wav(ax, t, audio, title):
    """Plot the audio signal."""
    ax.plot(t, audio, linewidth=0.5, color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1])
    ax.set_title(title)


def plot_annotations(ax, annotations, audio):
    """Plot bubble annotations on the audio signal."""
    for i, (start_idx, end_idx) in enumerate(annotations, start=1):
        start_s = start_idx / 44100.0
        end_s = end_idx / 44100.0

        ax.axvspan(start_s, end_s, color="red", alpha=0.25)

        y_loc = 0.9 * (np.max(audio) if np.max(np.abs(audio)) > 0 else 1.0)
        ax.text(
            (start_s + end_s) / 2, y_loc, str(i), ha="center", va="top", color="black"
        )


def plot_window(ax, start_s, end_s):
    """Draw a square window on the audio signal."""
    return None
    window = hann_window(int((end_s - start_s) * 44100))
    t_window = np.arange(len(window)) / 44100.0 + start_s
    ax.plot(
        t_window,
        window * np.max(ax.get_ylim()) * 0.8 + np.min(ax.get_ylim()),
        color="green",
        linewidth=2,
        label="Hann Window",
    )
    ax.legend()
