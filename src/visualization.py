import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

path = 'data/raw/genres_original/rock/rock.00022.wav'

y, sr= librosa.load(path, sr=22050)

plt.figure(figsize=(12,4))
librosa.display.waveshow(y,sr=sr)
plt.title('Waveform (Rock)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
#plt.savefig('figures/rock/waveform_rock.png', dpi=300)
#plt.close()


mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(12, 5))
librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis='time',
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram (Rock)')
plt.tight_layout()
plt.savefig('figures/rock/mel_spectrogram_rock5.png', dpi=300)
plt.close()



genres = {
    'Rock': 'data/raw/genres_original/rock/rock.00022.wav',
    'Blues': 'data/raw/genres_original/blues/blues.00000.wav'
}

plt.figure(figsize=(14, 6))

for i, (genre, path) in enumerate(genres.items()):
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
plt.savefig('figures/compare/rockvsblues.png', dpi=300)
plt.close()




