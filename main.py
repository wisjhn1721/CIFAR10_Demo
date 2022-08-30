# John Wise
# 8/29/22
# This project is intended to serve as a sample of deep learning
# We will be training a deep learning model to classify images based on the CIFAR-10 dataset
# Tutorial: https://www.youtube.com/watch?v=7HPwo4wnJeA

import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# load the cifar10 dataset
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()

# Here we see there are 50,000 training images and 1,000 test images
# We can see that each image is 32 X 32 and has 3 values, one for Red, Green, and Blue respectively
print("Shape of training dataset: ", X_train.shape)
print("Shape of test dataset: ", X_test.shape)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()


# The shape of y is (50000,1) it looks like [[6], [9], [9], ... [5]]
# We want to leave array as length of 50,000 but flatten each index tto a value rather than a list, thus (-1,)
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Here's a picture from the dataset:
plot_sample(X_train, y_train, 0)


# Normalize the images to a number from 0 to 1. Image has 3 channels (R,G,B)
# and each value in the channel can range from 0 to 255.
# Hence to normalize in 0-->1 range, we need to divide it by 255
X_train = X_train / 255.0
X_test = X_test / 255.0

# Now lets start building some models! We will start with an Artificial Neural Net and then create a ConvNet
# Build simple artificial neural network for image classification
# This network will have 2 hidden layers. The input layer will have 32 X 32 X 3 = 3,072 nodes.
# This first hidden layer will have 3,000 nodes with a ReLU activation function. i.e. f(x) = max(0,x).
# The second hidden layer will have 1,000 nodes again with a ReLU activation function.
# Lastly, the output layer will have 10 nodes which will provide scores for our 10 categories.
# A Softmax activation function is used in the last layer. Learn more here: https://www.pinecone.io/learn/softmax-activation/
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# In this model we will use Stochastic Gradient Descent(SGD) during backpropagation. More here: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
# The loss function will be a categorical cross entropy loss function. More here: https://vitalflux.com/keras-categorical-cross-entropy-loss-function/
ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Here we will perform forward passes and back props to train the network
ann.fit(X_train, y_train, epochs=5)

# We can see that at the end of 5 epochs, accuracy is at around 49%... Not great.
# Here's a report of how well our model is performing
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# Lets try a Convolutional Neural Net!
# in this ConvNet we will have the input layer (32 x 32 x 3), a convolutional layer with 32 3 x 3 filters,
# a pooling layer which will downsample by half, a convolutional layer with 64 3 x 3 filters, another pooling layer
# which will down sample by half, a flatten layer (Flatten() method converts multi-dimensional matrix to single
# dimensional matrix), a dense layer (Dense Layer is simple layer of neurons in which each neuron receives input
# from all the neurons of previous layer and is used to classify image based on output from convolutional layers),
# And lastly a dense layer of 10 which will produce our 10 category scores.
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Lets print out some information about the model
print("\n\nConvolutional Neural Network Model:\n")
print(cnn.summary())

# In this model we will use Adam (adaptive moment estimation) during backpropagation. Adam is an optimization algorithm
# that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative
# based in training data. More here: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# The loss function will be a categorical cross entropy loss function.
# More here: https://vitalflux.com/keras-categorical-cross-entropy-loss-function/
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Here we will perform forward passes and back props to train the network
cnn.fit(X_train, y_train, epochs=40)

# With CNN, at the end 5 epochs, accuracy was at around 70% which is a significant improvement over ANN!
print("Score:", cnn.evaluate(X_test,y_test))

y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]
# Now lets show a sample and see how well we did
plot_sample(X_test, y_test, 13)
print(' What we predicted: ', classes[y_classes[13]])


# Here's a report of how well our model is performing
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("CNN Classification Report: \n", classification_report(y_test, y_pred_classes))
