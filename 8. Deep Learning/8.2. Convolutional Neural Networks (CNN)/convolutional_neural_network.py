# Convolutional Neural Network

## Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

## Part 1 - Data Preprocessing
### Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    "dataset/training_set", target_size=(64, 64), batch_size=32, class_mode="binary"
)

### Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_set = test_datagen.flow_from_directory(
    "dataset/test_set", target_size=(64, 64), batch_size=32, class_mode="binary"
)

## Part 2 - Building the CNN
### Initializing the CNN
### Step 1 - Convolution
### Step 2 - Pooling
### Adding a second convolutional layer
### Step 3 - Flattening
### Step 4 - Full Connection
### Step 5 - Output layer
## Part 3 - Training the CNN
### Compiling the CNN
### Training the CNN on the Training set and evaluating it on the Test set
## Part 4 - Making a single prediction
