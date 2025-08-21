#first MLP code

import numpy as np
import matplotlib.pyplot as plt
import random 
import math 
import struct
import tensorflow as tf



#every instance of the dataset is in the format 28x28 normalized after the function
def load_MNIST():

    # Carregar o dataset MNIST do TensorFlow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizar os valores dos pixels para o intervalo [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # verify format
    print("train format:", x_train.shape)
    print("test format:", x_test.shape)
    print("y_train:", y_train.shape)

    # Mostrar a primeira imagem
    # plt.imshow(x_train[0], cmap='gray')
    # plt.title(f"Rótulo: {y_train[0]}")
    # plt.show()

    return x_train, x_test, y_train, y_test



# function to reshape the data from 28x28 to an array of 784 values, in batches, and return the data reshaped
def input_layer(X_batch):
    """
    function that receive a batch of the dataset and reshape to an 1D array
    """
    batch_size = X_batch.shape[0]
    X_flatten = X_batch.reshape(batch_size, -1)
    
    return X_flatten



#function to generate the weight matrix between the layers 1 to 2 
def weight_matrix(num_layer1, num_layer2):
    
    #np.random.radn is a normal distribution
    W = np.random.randn(num_layer2, num_layer1) * np.sqrt(2/num_layer1)
    return W



#function to generate the bias column vector
def bias(num_layer2):
    return np.zeros((num_layer2, 1))



#activation function
def ReLU(x):
    return np.maximum(0, x)



#defining the probabilitie distribution for the output layer
# Z.shape() = (32 instances, num_neurons_previous_layer)
def softmax(z):

    #z = each element in each raw of the matrix, np.max(z) = the highst value in the raw. (raw = instance)
    #exp_z is a matrix like z, but with the exponential computed for each value
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) #subtracting the maximun to avoid overflow

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)



#defining the one hot encoder transformation: it takes each instance and transform in a binary array classes like
# yi = 3 = [0, 0, 0, 1, 0, 0 ,0, 0, 0, 0, 0]
# y_true.shape() = (batch_size)
def onehot_encoder(y_true, num_class): 

    #create a matrix with zeros that each raw is an intance with 10 classes
    y_onehot = np.zeros((y_true.shape[0], num_class))

    #switch the class number (integer in the y_true) in the instance to 1
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    return y_onehot



#defining the error function
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-9  # small value to avoid log(0)

    loss = -np.sum(y_true * np.log(y_pred + epsilon), axis=1)

    return np.mean(loss)


#not used.
#Z.shape() = (32 instances, 9 neurons = 9 classes)
def output_layer(W, X, b, y_true):
    Z = np.dot(X, W.T) + b.T
    y_pred = softmax(Z)
    loss = cross_entropy_loss(y_pred, y_true)

    return loss


#define the forward pass to compute the next value of the neuron after the activation function
# X_flatten.shape() = (32 instances, 728 values)
# Weighs.shape() = (num_neurons_layer_receiving, num_neurons_layer_sending)
# Bias.shape() = (num_neurons_layer_receiving, 1)
#Z.shape = (32 instances, num_neurons_layer_receiving)

def forward_pass(X, Weight_list, biases_list, y_true):

    X_flatten = input_layer(X)

    Z_list = []
    Activation_list = []

    #adding the first activation as the raw data
    Activation_list.append(X_flatten)

    #hidden layers
    A = X_flatten
    for i in range(len(Weight_list) - 1):
        Z = np.dot(A, Weight_list[i].T) + biases_list[i].T
        A = ReLU(Z)

        Z_list.append(Z)
        Activation_list.append(A)

    #output layer
    Z_out = np.dot(A, Weight_list[-1].T) + biases_list[-1].T
    y_pred = softmax(Z_out)

    Z_list.append(Z_out)
    Activation_list.append(y_pred)

    #loss
    loss = cross_entropy_loss(y_pred, y_true)

    return loss, Z_list, Activation_list



#defining backpropagation with ReLU
def backpropagation(y_true, w_list, b_list, Z_list, A_list):
    """
    A_list = activation list (A_list[-1] = y_pred)
    """

    #number of layer
    L = len(w_list)
    batch_size = y_true.shape[0]
    m = batch_size

    #lists to store the gradients 

    dW_list = [None] * L     #[None, None, None, ..., None]
    db_list = [None] * L

    #Loss in the output layer
    y_pred = A_list[-1]     #y_pred.shape() = (batch_size, num_class)
    dZ = y_pred - y_true 


    #gradients in the output
    
    A = A_list[-2] #activation in the penultimate layer 
    dW_list[L-1] = np.dot(dZ.T, A) / m   #shape: (num classes, num neurons last hidden layer)
    db_list[L-1] = np.sum(dZ, axis=0, keepdims=True).T / m   #shape: (num_neurons, 1)


    #propagation to the hidden layer
    for i in range(L-2, -1, -1):
        W_next = w_list[i + 1]  #weight matrix of the next layer
        Z_i = Z_list[i]         #values of the layer i

        #new delta Z without ReLU derivate 
        dA = np.dot(dZ, W_next) 
        
        #d(ReLU)
        dReLU = (Z_i > 0).astype(float)     #shape = (batch size, num neurons i layer)
        dZ = dA * dReLU     #hadamard product to * each element with each element, thenn, for each value in dZ < 0 will be 0 and for dZ > 0 will be 1
        
        A_prev = A_list[i]

        dW = np.dot(dZ.T, A_prev) / m          # shape: (n_neurons_i, n_neurons_i-1)
        db = np.sum(dZ, axis=0, keepdims=True).T / batch_size   # shape: (n_neurons_i, 1)

        dW_list[i] = dW
        db_list[i] = db

    return dW_list, db_list


#function to predict and evaluate
def predict(X, w_list, b_list):
    #flatten the X with shape (batch_size, 28, 28)

    X_flatten = input_layer(X)
    A = X_flatten

    #hidden layers
    for i in range(len(w_list) - 1):
        Z_hidden = np.dot(A, w_list[i].T) + b_list[i].T
        A = ReLU(Z_hidden)

    #output layer
    Z_out = np.dot(A, w_list[-1].T) + b_list[-1].T
    prediction = softmax(Z_out)

    return prediction


#defining the acuracy function

def accuracy(X, y_true, w_list, b_list):

    #since we have y_true from the original data like an array of integers, then we compare directly with the prediction result
    y_pred = predict(X, w_list, b_list)             #shape = (batch_size, num_classes)
    pred_labels = np.argmax(y_pred, axis = 1)       #return the index (the class predction) of the highest value in a row (an instance). Shape = (batch size,)
    acc = np.mean(pred_labels == y_true)            #pred_labels == y_true return a boolean array with the same size of pred_labels

    return acc


#training function

def train_mlp(X_train, y_train, n_epochs, batch_size, learning_rate, num_layers):
    
    #labels in binary class:
    num_class = np.unique(y_train).shape[0]
    y_true = onehot_encoder(y_train, num_class)
    
    #getting the number of batches to separate X_train and y_train:
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    #initalizing the weights matrix and biases
    weights = []
    biases = []

    for i in range(len(num_layers) - 1):
        W = weight_matrix(num_layers[i], num_layers[i+1])
        b = bias(num_layers[i+1])

        weights.append(W)
        biases.append(b)
        print(f"W[{i}] shape = {W.shape}, b[{i}] shape = {b.shape}")


    #loop to train every epoch
    for epoch in range(n_epochs):

        #permutation of the data for each epoch
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_true[permutation]

        #loop to train every batch
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            #passforward
            loss, Z_list, Activation_list = forward_pass(X_batch, weights, biases, y_batch)

            #backpropagation
            dW_list, db_list = backpropagation(y_batch, weights, biases, Z_list, Activation_list)

            #updating weights and biases
            for l in range(len(weights)):
                weights[l] -= learning_rate * dW_list[l]
                biases[l] -= learning_rate * db_list[l]

        print(f"Epoch {epoch+1}/{n_epochs}, Loss final = {loss:.4f}")

    return weights, biases

if __name__ == "__main__":

    print("=== Versão nova do script ===")

    x_train, x_test, y_train, y_test = load_MNIST()

    input_dimension = 784
    hidden_layer1 = 128
    hidden_layer2 = 64
    out_layer = 10

    num_layers = [input_dimension, hidden_layer1, hidden_layer2, out_layer]

    n_epochs = 10
    batch_size = 32
    learning_rate = 0.01

    weights, biases = train_mlp(x_train, y_train, n_epochs, batch_size, learning_rate, num_layers)

    for i, W in enumerate(weights):
        print(f"W[{i}] shape = {W.shape}, b[{i}] shape = {biases[i].shape}")

    # 5) evaluate the training data
    test_acc = accuracy(x_test, y_test, weights, biases)
    print(f"Acurácia no teste = {test_acc:.2%}")