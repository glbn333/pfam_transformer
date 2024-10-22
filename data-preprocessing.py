import re
import csv
from collections import defaultdict

# Path to the SEED file
seed_file = r"C:/Users/gagou/Desktop/Data science and info/lebrun/pfam_transformer_project/Pfam-A.seed/Pfam-A.seed"

# Dictionary to store the sequences for each family
family_sequences = defaultdict(list)

# Regular expressions to match the family description and sequence count
family_re = re.compile(r"#=GF DE\s+([^\n]+)")
sequence_count_re = re.compile(r"#=GF SQ\s+(\d+)")
pattern = re.compile(r'^([A-Z0-9_]+/\d+-\d+)\s+([A-Z.]+)$')

# Variables to store the current family and sequence count
current_family = None
nb_sequences = 0
list_of_fams = []

    # Read the SEED file line by line, specifying the encoding as UTF-8
with open(seed_file, "r", encoding="utf-8") as file:
    for line in file:
        line.strip()

        # If this line contains a family description, update the current family
        family_match = family_re.match(line)
        if family_match:
            current_family = family_match.group(1)

            #print(current_family)

        # If this line contains the sequence count, update the sequence count
        sequence_count_match = sequence_count_re.match(line)
        if sequence_count_match:
            nb_sequences = int(sequence_count_match.group(1))
            
            if nb_sequences>900:
                list_of_fams.append(current_family)
                print(current_family, nb_sequences)

        if current_family in list_of_fams:
            sequence_match = pattern.match(line)
            if sequence_match:
                current_sequence = sequence_match.group(2)
                family_sequences[current_family].append(current_sequence)
                #print(current_sequence)
            #print(nb_sequences)

print(f"Kept {sum(len(sequences) for sequences in family_sequences.values())} sequences in {len(family_sequences)} families")


# Save the filtered families to a CSV file
output_file = r"C:/Users/gagou/Desktop/Data science and info/lebrun/pfam_transformer_project/filtered_families.csv"

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sequence", "family"])
    for family, sequences in family_sequences.items():
        for sequence in sequences:
            writer.writerow([sequence, family])


'''
            if sequence_count_match:
                sequence_count = int(sequence_count_match.group(1))

        # If this line contains a sequence, add it to the current family
        if line.startswith(">"):
            sequence = line[1:].strip()
            family_sequences[current_family].append(sequence)

# Filter the families to keep only those with more than 900 sequences
filtered_families = {family: sequences for family, sequences in family_sequences.items() if len(sequences) > 900}

# Print the number of sequences and families kept
print(f"Kept {sum(len(sequences) for sequences in filtered_families.values())} sequences in {len(filtered_families)} families")

# Save the filtered families to a CSV file
with open("filtered_families.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sequence", "family"])
    for family, sequences in filtered_families.items():
        for sequence in sequences:
            writer.writerow([sequence, family])'''