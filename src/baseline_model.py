import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')


le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded) 


X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1
)

loss, acc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest accuracy: {acc:.4f}")

y_pred_probs = model.predict(X_test_scaled) 

y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap='Blues')
plt.xlabel('Predicted genre')
plt.ylabel('Actual genre')
plt.title('Confusion matrix - Baseline MLP Model')
os.makedirs('results/mlp', exist_ok=True)
plt.savefig('results/mlp/confusion_matrix.png')

print('\nDetailed classification report:\n')
print(classification_report(y_test_labels, y_pred_labels, target_names=genres))



