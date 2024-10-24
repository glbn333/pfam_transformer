# Define the model parameters
from sentencepiece import SentencePieceProcessor
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

# Create the Transformer model
model = create_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, num_classes)




# Display the model summary
model.summary()
#Output :
#Model: "model"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#input_1 (InputLayer)         [(None, 128)]             0
#_________________________________________________________________
#token_and_position_embedding (None, 128, 64)           332800
#_________________________________________________________________
#transformer_block (Transforme (None, 128, 64)           66048
#_________________________________________________________________
#transformer_block_1 (Transfor (None, 128, 64)           66048
#_________________________________________________________________
#transformer_block_2 (Transfor (None, 128, 64)           66048
#_________________________________________________________________
#transformer_block_3 (Transfor (None, 128, 64)           66048
#_________________________________________________________________
#global_average_pooling1d (Glo (None, 64)                0
#_________________________________________________________________
#dense (Dense)                (None, 32)                2080
#=================================================================
#Total params: 597,072
#Trainable params: 597,072
#Non-trainable params: 0
#_________________________________________________________________

# Save the model architecture to a file
model.save("transformer_model.keras")


'''

import pandas as pd
import numpy as np
import tensorflow as tf
from model_architecture import create_model
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Set the hyperparameters
vocab_size = 20  # Size of the vocabulary (number of unique amino acids)
maxlen = 1000  # Maximum length of the input sequences
embed_dim = 64  # Dimensionality of the embedding layer
num_heads = 8  # Number of attention heads in the multi-head attention layer
ff_dim = 256  # Dimensionality of the feed-forward network
num_layers = 4  # Number of Transformer blocks in the model
num_classes = 10  # Number of output classes (Pfam families)
batch_size = 32  # Batch size for training
num_epochs = 10  # Number of epochs for training
learning_rate = 1e-4  # Learning rate for the optimizer

# Load the data from the CSV file
data = pd.read_csv("sequences.csv")

# Remove dots from the sequences
data["sequence"] = data["sequence"].str.replace(".", "")

# Convert the sequences to numerical representations
# Here, we use a simple one-hot encoding scheme
# You can replace this with your own encoding scheme if needed
X = np.zeros((len(data), maxlen, vocab_size))
for i, seq in enumerate(data["sequence"]):
    for j, aa in enumerate(seq):
        X[i, j, ord(aa) - ord("A")] = 1

# Convert the Pfam family labels to numerical representations
# Here, we use a simple label encoding scheme
# You can replace this with your own encoding scheme if needed
y = data["family"].astype("category").cat.codes.values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Transformer model
model = create_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, num_classes)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))

# Save the trained model
model.save("pfam_transformer.h5")'''
