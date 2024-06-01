# Deep Neural Network (DNN) for Image Classification using MNIST Dataset
The following code is a detailed explanation of the first task from ACM-SIG-AI recruitment tasks.<br>
The code includes loading and preprocessing the data and training the network and evaluating its accuracy.<br>

#### Libraries used
1. numpy
2. Tensorflow and Keras
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
## 2. Network Architecture
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

<details>
  <summary><i><b>Surface level working of the architecture</b></i></summary>

* **Conv2D**: Introduces a 2D convolution kernel (a matrix) used for detecting spaces and edges. It works like a set of moving filters over the input image which is then activated using an activation function (ReLU in this case).
* **MaxPool2D**: Reduces the spatial dimensions of the input , performing downsampling by taking the maximum value in each region (part) of the input.
* **Dropout**: Regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting.
* **Flatten**: Converts the 3D output of convolutional layers into a 1D matrix, making it suitable for input to fully connected layers.
* **Dense**: Fully connected layers which perform high-level classification. Each neuron in a Dense layer is connected to every neuron in the previous layer.
* **Softmax Activation**: Applied in the final layer for multi-class classification, converting raw scores into probabilities .
  
</details>

## 3. Network Compilation
```py
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
## 4. Training and Evaluation
```py
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    verbose=2,
                    validation_data=(x_test, y_test))
```
```py
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy: {accuracy * 100}')
```
## 5. Prediction
```py
for i in range(5):
    image = x_train[i]
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.show()
    image = image.reshape(1, 28, 28, 1)
    p = model.predict(image)
    print('Predicted:', argmax(p))
```
------------------------------------------------------------
### Output
<b><i>Accuracy : 99.36000108718872</i></b>

![image](https://github.com/0x-d15c0/ACM-SIGAI-TASKS/assets/117750351/82de7f30-e97e-46e3-a106-90a9bf161342)
![image](https://github.com/0x-d15c0/ACM-SIGAI-TASKS/assets/117750351/f894c69b-0060-41d3-a3b4-2cd2bea61392)

## Reference
1. https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/
2. https://medium.com/@jwbtmf/reshaping-the-dataset-for-neural-network-15ee7bcea25e
3. https://www.geeksforgeeks.org/introduction-convolution-neural-network/
