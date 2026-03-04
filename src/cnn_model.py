import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import seaborn as sns

DATA_PATH = 'data/processed/X_spectrograms.npy'
LABEL_PATH = 'data/processed/y_spectrograms.npy'
MODEL_PATH = 'data/models/gtzan_cnn.keras' 

X = np.load(DATA_PATH)
y = np.load(LABEL_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock']


def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4), 
        layers.Dense(10, activation='softmax')
    ])
    return model

    
def train_model():
    print('Training new model...')
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_cnn(input_shape)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100, 
        batch_size=32,
        callbacks=[early_stop]
    )
    
    os.makedirs('data/models', exist_ok=True)
    model.save(MODEL_PATH)
    print('Model saved successfully.')
    return model, history


def results(model, history=None):
    os.makedirs('results/cnn', exist_ok=True)

    if history is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig('results/cnn/results.png')


    y_pred_probs = model.predict(X_test) 
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = y_test

    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap='Blues')
    plt.xlabel('Predicted genre')
    plt.ylabel('Actual genre')
    plt.title('Confusion matrix - CNN Model')
    plt.savefig('results/cnn/confusion_matrix.png')

    print('\nDetailed classification report:\n')
    print(classification_report(y_test_labels, y_pred_labels, target_names=genres))
    
    
if os.path.exists(MODEL_PATH):
    print('Loading existing model...')
    model = tf.keras.models.load_model(MODEL_PATH)
    results(model, history=None)
    
else:
    model, history = train_model()
    results(model, history)
