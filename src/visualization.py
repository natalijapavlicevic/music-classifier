import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def plot_waveform(path, genre, save_dir='figures'):
    
    y, sr= librosa.load(path, sr=22050)
    genre_dir = os.path.join(save_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)

    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y,sr=sr)
    plt.title(f'Waveform ({genre.capitalize()})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(genre_dir, "waveform.png"),
        dpi=300
    )
    plt.close()

def plot_mel_spectrogram(path, genre, save_dir='figures'):
    
    y, sr = librosa.load(path, sr=22050)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    genre_dir = os.path.join(save_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-spectrogram ({genre.capitalize()})')
    plt.tight_layout()
    plt.savefig(
        os.path.join(genre_dir, "mel_spectrogram.png"),
        dpi=300
    )
    plt.close()

def compare_genres(genre_paths, save_dir='figures'):
    
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))

    for i, (genre, path) in enumerate(genre_paths.items()):
        y, sr = librosa.load(path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.subplot(1, 2, i + 1)
        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-spectrogram ({genre})')

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "genre_comparison.png"),
        dpi=300
    )
    plt.close()


# Visualization flags
PLOT_WAVEFORM = True
PLOT_MEL_SPECTROGRAM = True
COMPARE_GENRES = False


if __name__ == "__main__":

    jazz_path = "data/raw/genres_original/jazz/jazz.00000.wav"

    if PLOT_WAVEFORM:
        plot_waveform(jazz_path, "jazz")

    if PLOT_MEL_SPECTROGRAM:
        plot_mel_spectrogram(jazz_path, "jazz")

    if COMPARE_GENRES:
        compare_genres({
            "Classical": "data/raw/genres_original/classical/classical.00000.wav",
            "Metal": "data/raw/genres_original/metal/metal.00000.wav"
        })


