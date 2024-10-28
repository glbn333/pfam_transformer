import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from model_architecture import TransformerBlock, TokenAndPositionEmbedding


model = keras.models.load_model(
    "pfam_transformer_trained_5epochs.keras",
)

df = pd.read_csv('preprocessed_sequences_encoded.csv')

X = df['sequence']
y = df['encoded_family']

# Tokenize the sequences
X_tokenized = []
for seq in X:
    tokens = sp.EncodeAsIds(seq)
    X_tokenized.append(tokens)

# Pad the sequences
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_tokenized, maxlen=128, padding='post')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_padded, y)

print(f"Test accuracy: {accuracy:.2f}")

# Predict the class labels for the test data
y_pred = model.predict(X_padded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
confusion_matrix = tf.math.confusion_matrix(y, y_pred_classes)

print("Confusion Matrix:")
print(confusion_matrix)

# Calculate the classification report
from sklearn.metrics import classification_report

print("Classification Report:")

print(classification_report(y, y_pred_classes, target_names=family_dict.values()))

# Save the confusion matrix as an image
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=family_dict.values(), yticklabels=family_dict.values())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')


