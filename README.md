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
- Files are loaded (using `librosa` python library), the ones that fail to load are skipped and logged in `failed_files` varable
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

## Baseline Model

The baseline model is a fully connected neural network (MLP) trained on precomputed MFCC statistical features (mean and standard deviation).

### Architecture

- Dense (128 units, ReLU)
- Dense (64 units, ReLU)
- Dense (10 units, Softmax)

### Training Setup

- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 25
- Batch size: 32

### Results

- Training accuracy: ~98%
- Validation accuracy: ~70%
- Test accuracy: 67%

The model shows moderate overfitting, as training accuracy approaches 100% while validation and test accuracy stabilize around 70%. This is expected for a simple MLP without regularization or convolutional layers.

The baseline serves as a performance reference for future improvements (e.g., regularization, CNN-based models, or data augmentation).

### Running the Baseline Model

Make sure the precomputed feature files (`X.npy` and `y.npy`) are available in the `data/processed/` directory.

To train and evaluate the baseline model, run:

```bash
python3 src/baseline_model.py
```