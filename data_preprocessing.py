import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set the path to the input file containing the sequences and families
input_file = "filtered_families.csv"

# Set the path to the output file for the preprocessed sequences
output_file = "preprocessed_sequences.csv"

# Load the data from the input file
data = pd.read_csv(input_file)

# Remove the '.' characters from each sequence
data["sequence"] = data["sequence"].str.replace(".", "")

# Save the preprocessed sequences to a new file
data.to_csv(output_file, index=False)