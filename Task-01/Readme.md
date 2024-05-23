# Deep Neural Network (DNN) for Image Classification using MNIST Dataset
The following code is a detailed explanation of the first task from ACM-SIG-AI recruitment tasks.<br>
The code includes loading and preprocessing the data and training the network and evaluating its accuracy.<br>

#### Libraries used
1. numpy
2. Tensorflow
3. Keras
4. matplotlib

-------------------

# Code overview
``` py
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
```py
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```
The images are reshaped to include a single channel - gray (grayscale).
## 2. Model Architecture
```py
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
```
 What is the purpose of all these layers - a surface level overview of how they work :
 
