import os
import numpy as np
import librosa

BASE_DATA_DIR = "data/raw/genres_original"
TARGET_DATA_DIR = "data/processed"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128

X = []
y = []

genres = sorted([
    g for g in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, g))
])

for label, genre in enumerate(genres):
    genre_path = os.path.join(BASE_DATA_DIR, genre)

    for file in os.listdir(genre_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(genre_path, file)

        try:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Warning: Skipping file {file_path} ({e})")
            continue

        if len(signal) >= SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            continue

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        X.append(mel_db)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

os.makedirs(TARGET_DATA_DIR, exist_ok=True)
np.save(os.path.join(TARGET_DATA_DIR, "X_spectrograms.npy"), X)
np.save(os.path.join(TARGET_DATA_DIR, "y_spectrograms.npy"), y)