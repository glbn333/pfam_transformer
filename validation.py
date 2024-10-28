import tensorflow as tf
from tensorflow import keras
from sentencepiece import SentencePieceProcessor
from keras import layers
import numpy as np
import pandas as pd
from model_architecture import create_model

# Load the trained SentencePiece tokenizer
sp = SentencePieceProcessor()
sp.Load('amino_acids.model')

# Get the vocabulary size
vocab_size = sp.GetPieceSize()

# Maximum sequence length
maxlen = 128

# Embedding dimension

embed_dim = 64

# Number of attention heads
num_heads = 8

# Feed-forward dimension
ff_dim = 64*4

# Number of Transformer blocks
num_layers = 4

# Number of output classes
num_classes = 32

model = create_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, num_classes) 

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

model.load_weights('training/pfam_transformer_trained_5epochs.weights.h5')

learning_rate = 1e-4  # Learning rate for the optimizer
# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy"])

# Evaluate the model on the test data
# loss, accuracy = model.evaluate(X_padded, y)

# print(f"Test accuracy: {accuracy:.2f}")

# Predict the class labels for the test data
y_pred = model.predict(X_padded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
confusion_matrix = tf.math.confusion_matrix(y, y_pred_classes)

print("Confusion Matrix:")
print(confusion_matrix)

#Family dict : df['encoded'] : df['family']
family_dict = dict(zip(df['encoded_family'], df['family']))

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


