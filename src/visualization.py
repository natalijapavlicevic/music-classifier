import argparse
import os
import random

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt



BASE_DATA_DIR = "data/raw/genres_original"
FIGURES_DIR = "data/figures"
LOG_FILE = os.path.join(FIGURES_DIR, "used_files.txt")
SAMPLE_RATE = 22050
N_MELS = 128
FILES_PER_GENRE = 5
RANDOM_SEED = 42

def plot_waveform(path, genre, index):
    
    y, sr= librosa.load(path, sr=SAMPLE_RATE)
    genre_dir = os.path.join(FIGURES_DIR, genre)
    os.makedirs(genre_dir, exist_ok=True)

    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y,sr=sr)
    plt.title(f'Waveform ({genre.capitalize()})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(genre_dir, f"waveform_{index}.png"),
        dpi=300
    )
    plt.close()

def plot_mel_spectrogram(path, genre,index):
    
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    genre_dir = os.path.join(FIGURES_DIR, genre)
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
        os.path.join(genre_dir, f"mel_spectrogram_{index}.png"),
        dpi=300
    )
    plt.close()

def compare_two_files(
    ref_path,
    ref_label,
    other_path,
    other_label,
    save_path
):
    y1, sr = librosa.load(ref_path, sr=SAMPLE_RATE)
    y2, _ = librosa.load(other_path, sr=SAMPLE_RATE)

    mel1 = librosa.feature.melspectrogram(
        y=y1, sr=sr, n_mels=N_MELS
    )
    mel2 = librosa.feature.melspectrogram(
        y=y2, sr=sr, n_mels=N_MELS
    )

    mel1_db = librosa.power_to_db(mel1, ref=np.max)
    mel2_db = librosa.power_to_db(mel2, ref=np.max)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        mel1_db, sr=sr, x_axis="time", y_axis="mel"
    )
    plt.title(ref_label)
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        mel2_db, sr=sr, x_axis="time", y_axis="mel"
    )
    plt.title(other_label)
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_random_files(directory, n):
    files = [
        f for f in os.listdir(directory)
        if f.endswith(".wav")
    ]
    return random.sample(files, n)

def get_random_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    return random.choice(files)


def main(args):
    random.seed(RANDOM_SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    genres = [
        g for g in os.listdir(BASE_DATA_DIR)
        if os.path.isdir(os.path.join(BASE_DATA_DIR,g))
    ]
    
    with open(LOG_FILE, "w") as log_file:
     log_file.write("Used audio files\n")
     log_file.write("=================\n\n")

     for genre in genres:
         genre_dir = os.path.join(BASE_DATA_DIR, genre)
         selected_files = get_random_files(genre_dir, FILES_PER_GENRE)

         log_file.write(f"{genre.upper()}:\n")

         for i, filename in enumerate(selected_files, start=1):
             path = os.path.join(genre_dir, filename)
             log_file.write(f"  {filename}\n")

             if args.plot_waveform:
                 plot_waveform(path, genre, i)

             if args.plot_mel:
                 plot_mel_spectrogram(path, genre, i)

             if args.compare_genres:
                 other_genres = [g for g in genres if g != genre]
                 other_genre = random.choice(other_genres)

                 other_dir = os.path.join(BASE_DATA_DIR, other_genre)
                 other_file = get_random_file(other_dir)
                 other_path = os.path.join(other_dir, other_file)

                 log_file.write(
                     f"    COMPARE WITH: {other_genre}/{other_file}\n"
                 )

                 compare_two_files(
                     ref_path=path,
                     ref_label=genre.capitalize(),
                     other_path=other_path,
                     other_label=other_genre.capitalize(),
                     save_path=os.path.join(
                         FIGURES_DIR,
                         genre,
                         f"compare_{i}.png"
                     )
                 )

         log_file.write("\n")


if __name__ == "__main__":

   parser = argparse.ArgumentParser(
        description="Audio data visualization"
    )

   parser.add_argument(
        "--plot-waveform",
        action="store_true",
        help="Generate waveform plots"
    )

   parser.add_argument(
        "--plot-mel",
        action="store_true",
        help="Generate mel-spectrograms"
    )

   parser.add_argument(
        "--compare-genres",
        action="store_true",
        help="Compare each file with a random file from another genre"
    )

   args = parser.parse_args()
   main(args)
