import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('preprocessed_sequences.csv')

df['family'] = df['family'].str.replace('domain', '').astype('category')  # Extract the main family name

family_dict = {i: family for i, family in enumerate(df['family'].unique())}  # Create a dictionary of family names'

df['encoded_family'] = df['family'].cat.codes  # Encode the family names

fig = plt.subplots(figsize=(10, 8))  # Set the figure size

sns.countplot(x='encoded_family', data=df)  # Countplot of the 'family' column


plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.xlabel('Pfam Family')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label
plt.title('Distribution of Pfam Families')  # Set the title of the plot
plt.show()
fig[0].savefig('family_distribution_preprocessed.png')  # Save the plot as an image file