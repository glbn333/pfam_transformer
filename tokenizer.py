import pandas as pd
import sentencepiece as spm

# Load the CSV file
#df = pd.read_csv('preprocessed_sequences.csv')

#with open('sequences.txt', 'w') as f:
#    for seq in df['sequence']:
#        f.write(seq + '\n')

# Train the SentencePiece tokenizer
#spm.SentencePieceTrainer.train('--input=sequences.txt --model_prefix=amino_acids --vocab_size=5000')

# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.Load('amino_acids.model')    

# Tokenize the sequences
sequence_test = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS'
tokens = sp.EncodeAsIds(sequence_test)
print(tokens)
#Output : [713, 6, 200, 48, 46, 532, 33, 37, 12, 379, 85, 140, 2721, 304, 
# 484, 194, 275, 1069, 8, 1357, 1235, 39, 1642, 173, 574, 2002, 253, 881, 68
# , 2190, 384, 231, 31, 1596, 333, 1479, 96, 289, 1133, 463, 361, 240, 838,
#  1390, 150, 154, 1015, 294, 730, 257, 235, 349, 114, 1795, 105, 110, 520, 1709,
#  1039, 897, 120, 157, 89, 52, 691, 3, 308, 1448, 3442, 10, 1747, 148, 1411,
#  1902, 8, 657, 941, 427, 406, 1145]

import tensorflow as tf

# Pad the sequences
max_length = 128
sequences = [tokens]
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
print(padded_sequences)
#Output : [[1747  148 1411 1902    8  657  941  427  406 1145]]

# Define the embedding layer
from tensorflow.keras.layers import Embedding

vocab_size = sp.GetPieceSize()
print('vocab size =', vocab_size)
#Output : vocab size = 5000

embedding_dim = 64

embedding_layer = Embedding(vocab_size, embedding_dim)
embedded_sequences = embedding_layer(padded_sequences)
print(embedded_sequences)
print(embedded_sequences.shape)
