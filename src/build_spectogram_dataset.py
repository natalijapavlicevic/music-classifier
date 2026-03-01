import os
import numpy as np
import librosa

BASE_DATA_DIR = "data/raw/genres_original"
TARGET_DATA_DIR = "data/processed"

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # GTZAN pesme su 30s
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Parametri za segmentaciju (OVO JE KLJUČNO)
NUM_SEGMENTS = 10 
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
N_MELS = 128

os.makedirs(TARGET_DATA_DIR, exist_ok=True)

X, y = [], []

genres = sorted([g for g in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, g))])

for label, genre in enumerate(genres):
    genre_path = os.path.join(BASE_DATA_DIR, genre)
    print(f"Obradjujem: {genre}...")
    
    for file in os.listdir(genre_path):
        if not file.endswith(".wav") or file == "jazz.00054.wav": # Oštećen fajl u GTZAN
            continue

        file_path = os.path.join(genre_path, file)
        try:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        except Exception as e:
            continue

        # Deli pesmu na 10 delova (svaki deo je ~3 sekunde)
        for s in range(NUM_SEGMENTS):
            start_sample = SAMPLES_PER_SEGMENT * s
            finish_sample = start_sample + SAMPLES_PER_SEGMENT
            
            chunk = signal[start_sample:finish_sample]
            
            # Mel-spektrogram za taj chunk
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalizacija na fiksni opseg -80 do 0 dB (skalirano na 0 do 1)
            mel_db = (mel_db + 80) / 80 

            # Osiguravamo fiksnu širinu (treba da bude ~130 piksela za 3s)
            # Ako je chunk malo kraći zbog zaokruživanja, dodajemo padding
            if mel_db.shape[1] < 128:
                mel_db = np.pad(mel_db, ((0,0), (0, 128 - mel_db.shape[1])), mode='constant')
            else:
                mel_db = mel_db[:, :128]

            X.append(mel_db[..., np.newaxis])
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

np.save(os.path.join(TARGET_DATA_DIR, "X_spectrograms.npy"), X)
np.save(os.path.join(TARGET_DATA_DIR, "y_spectrograms.npy"), y)

print(f"Završeno! Novi oblik podataka: {X.shape}") # Trebalo bi da bude (~10000, 128, 128, 1)