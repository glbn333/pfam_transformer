# Define the model parameters
from sentencepiece import SentencePieceProcessor
from model_architecture import create_model
import os

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

# Create the Transformer model
model = create_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, num_classes)

checkpoint_path = "./training/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path.format(epoch=0))


# Display the model summary
model.summary()
#Output :
'''
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ token_and_position_embedding         │ (None, 128, 64)             │         328,192 │
│ (TokenAndPositionEmbedding)          │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block (TransformerBlock) │ (None, 128, 64)             │         166,016 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_1                  │ (None, 128, 64)             │         166,016 │
│ (TransformerBlock)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_2                  │ (None, 128, 64)             │         166,016 │
│ (TransformerBlock)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_3                  │ (None, 128, 64)             │         166,016 │
│ (TransformerBlock)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling1d             │ (None, 64)                  │               0 │
│ (GlobalAveragePooling1D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_8 (Dense)                      │ (None, 32)                  │           2,080 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 994,336 (3.79 MB)
 Trainable params: 994,336 (3.79 MB)
 Non-trainable params: 0 (0.00 B)
'''

# Train the model using preprocessed_data.csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Set the hyperparameters
batch_size = 32  # Batch size for training
num_epochs = 5  # Number of epochs for training
learning_rate = 1e-4  # Learning rate for the optimizer

# Load the data from the CSV file
data = pd.read_csv("preprocessed_sequences.csv")

# Encode family names using LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['encoded_family'] = le.fit_transform(data['family'])

# Save new CSV file with encoded family names
# data.to_csv('preprocessed_sequences_encoded.csv', index=False)

# Tokenize the sequences using the SentencePiece tokenizer
X = []
for seq in data["sequence"]:
    tokens = sp.EncodeAsIds(seq)
    X.append(tokens)

# Pad the sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen, padding='post')

# Convert the Pfam family labels to numerical representations
y = data["encoded_family"].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy"])

# Create a callback that saves the model's weights every 1 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
    # filepath=checkpoint_path, 
   # verbose=1, 
   # save_weights_only=True,
   # save_freq=*batch_size)

# Train the model
model.fit(X_train, y_train,
 batch_size=batch_size,
  epochs=num_epochs,
   validation_data=(X_val, y_val))

# Save the trained model
model.save_weights('./training/pfam_transformer_trained_5epochs.weights.h5')

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
