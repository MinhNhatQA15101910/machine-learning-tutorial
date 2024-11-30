# Natural Language Processing

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

## Cleaning the texts
## Creating the Bag of Words model
## Splitting the dataset into the Training set and Test set
## Training the Naive Bayes model on the Training set
## Predicting the test set results
## Making the Confusion Matrix
