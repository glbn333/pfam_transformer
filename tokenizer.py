import pandas as pd
import sentencepiece as spm

# Load the CSV file
df = pd.read_csv('preprocessed_sequences.csv')

with open('sequences.txt', 'w') as f:
    for seq in df['sequence']:
        f.write(seq + '\n')

# Train the SentencePiece tokenizer
spm.SentencePieceTrainer.train('--input=sequences.txt --model_prefix=amino_acids --vocab_size=5000')

# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.Load('amino_acids.model')    

# Tokenize the sequences
sequence_test = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS'
tokens = sp.EncodeAsPieces(sequence_test)
print(tokens)

