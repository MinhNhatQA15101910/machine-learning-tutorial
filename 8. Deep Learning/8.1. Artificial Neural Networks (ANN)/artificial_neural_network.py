# Artificial Neural Network

## Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

## Part 1 - Data Preprocessing
### Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

### Encoding categorical data
#### Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#### One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))

print(X)

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Part 2 - Building the ANN
### Initializing the ANN
### Adding the input layer and the first hidden layer
### Adding the second hidden layer
### Adding the output layer
## Part 3 - Training the ANN
### Compiling the ANN
### Training the ANN on the Training set
## Part 4 - Making the predictions and evaluating the model
### Predicting the result of a single observation
### Predicting the Test set results
### Making the Confusion Matrix
