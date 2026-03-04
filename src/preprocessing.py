import librosa
import os
import numpy as np

data_dir = 'data/raw/genres_original'

genres = ['blues', 'classical', 'country', 'disco',
          'hiphop', 'jazz', 'metal', 'pop', 'reggae',
          'rock']

failed_files = []
X = []
y_labels = []

for genre in genres:
    genre_path = os.path.join(data_dir, genre)
    for file in os.listdir(genre_path):
        if not file.endswith('.wav'):
            continue
        
        path = os.path.join(genre_path, file)
        
        try: 
            y, sr = librosa.load(path, sr=22050)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            mel = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_mean = np.mean(mel_db, axis=1)
            mel_std = np.std(mel_db, axis=1)
            
            features = np.concatenate((mfcc_mean, mfcc_std,mel_mean,mel_std))
            
            X.append(features)
            y_labels.append(genre)
            
        except Exception as e:
            print(f'Error loading path {path}: {e}')
            failed_files.append(path)
            continue

        
print(f'Number of files failed to load: {failed_files}')
X = np.array(X)
y_labels = np.array(y_labels)

os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X.npy', X)
np.save('data/processed/y.npy', y_labels)