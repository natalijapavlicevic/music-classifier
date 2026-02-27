# Music Genre Classifier

This work is developed as part of the _Computational Intelligence_ course at _University of Belgrade, Faculty od Mathematics_, by following students:

- [Natalija Pavlićević 68/2022](https://github.com/natalijapavlicevic/)
- [Tamara Krstić 96/2022](https://github.com/TamaraKrstic)

## Project Description
The project addresses the problem of automatic music genre classification using techniques from computational intelligence and machine learning.

## Project Structure
- `data/` - dataset and extracted features
- `src/` - preprocessing, models, training, evaluation
- `results/` - metrics and visualizations

## Datasets

This project uses the [GTZAN Music Genre Classification dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which is publicly available on Kaggle.  
The dataset contains audio tracks labeled by music genre and is used for supervised music genre classification.
The dataset also includes pre-extracted audio features in CSV format, which were not used in the main experiments.

Due to size and licensing constraints, the dataset is not included in this repository.

### How to use
1. Download the dataset from Kaggle.
2. Extract it into `data/raw/`.
3. Run the preprocessing script to extract features.

## Setup
Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

- Raw audio files are stored in `data/raw/genres_original`
- Files are loaded (using `librosa` python library), the ones that fail to load are skipped and logged in `failed_files`
- Features are extracted using **MFCC** (Mel-Frequency Cepstral Coefficients):
    - 20 coefficients per frame
    - Calculated mean and standard deviation for each coefficient
    - Concatenated to form a 40-dimensional feature vector per audio file
- Labels correspond to music genres
- Processed data saved as `X.npy` and `y.npy`

To execute data preprocessing, position yourself in root directory of this repository and run:

```bash
python3 src/preprocessing.py
```

## Data Visualization

- To better understand the extracted features, several  visualizations are generated  
- **Audio waveforms** illustrate  signal amplitude over time (one for each genre)
- **Mel-spectrograms** are visualized to show the time–frequency representation of audio signals
- A comparison of mel-spectrograms from different music genres highlights clear spectral differences between genres
- Generated figures are saved in the `data/figures/` directory.
- The list of audio files used in the visualizations is recorded in `data/figures/used_files.txt`.

To execute data visualization, position yourself in root directory of this repository and run:

```bash
python3 src/visualization.py [FLAGS]
```  

| Flag               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `--plot-waveform`  | Generate waveform plots                                          |
| `--plot-mel`       | Generate mel-spectrograms                                        |
| `--compare-genres` | Compare each selected file with a random file from another genre |
