# Pfam Classification using a Transformer and Protein Sequences

This project aims to classify protein sequences into their respective Pfam families using a Transformer model. The project consists of several steps, including data preprocessing, model architecture, training, and evaluation. I wanted to perform this project after reading an article of Nambiar et al. of 2020 called "Transforming the Language of Life: Transformer Neural Networks for Protein Prediction Tasks" and a M2 report by Aliouane and Bendahmane named "Nouvelle approche de prédiction des classes protéiques issues d’un séquençage NGS par Deep Learning".

I wanted to perform this project using my own database hence I downloaded a dataset from InterPro. This dataset contained multiple sequence alignments of a significant amount of protein domains.

Link to download original dataset:

https://www.ebi.ac.uk/interpro/download/pfam/

The file is Pfam-A Seed alignment : "Pfam-A.seed.gz"

**I am currently working on the model architecture and the training script, they are not finished!**

---

## Data Extraction and Preprocessing

The data preprocessing step involves loading the data from the `Pfam-A.seed.gz` file. Using REGEX, I identified the families containing more than 900 sequences. I extracted them to a CSV file. All of this was done in the `data-extraction.py` file.

I plotted a countplot of the different families in the following figure. Each x-label corresponds to one category.
![Number of sequences per category](family_distribution.png)

We can observe that at some point the condition to extract the wanted families failed as we can see 4 families with less than 900 sequences. We will remove them from the dataset afterwards.

## Tokenization of our sequences

## Model Architecture

The model architecture consists of a Transformer model with the following components:

* Token and position embedding layers
* Multiple Transformer blocks, each consisting of a multi-head attention layer, a feed-forward network, and layer normalization
* A global average pooling layer
* An output layer with a softmax activation function

The model is implemented using TensorFlow and the Keras API.


## Training

The training step involves creating the Transformer model using the `create_model` function from the `model_architecture` script, compiling the model, training the model using the training data, and saving the trained model to a file. The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function.

## Evaluation

The evaluation step involves evaluating the performance of the trained model on the validation set and reporting the accuracy and loss metrics.

## Usage

To use this project, you will need to install the following dependencies:

* TensorFlow
* NumPy
* Pandas
* Seaborn
* Scikit-learn

The filtered CSV file is already uploaded to the repository. You can either directly use it or download the original dataset and perform the extraction using `data-extraction.py`. You will have to manually modify the path of your file.

You can then run the `data_preprocessing.py` script to preprocess the data,

the `model_architecture.py` script to define the model architecture, and the `training.py` script to train the model. The trained model will be saved to a file, which can be used to make predictions on new data.

## Results

## Future Work



