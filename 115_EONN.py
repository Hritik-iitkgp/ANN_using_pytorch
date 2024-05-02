import csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import collections
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read CSV file into a list and convert elements to int or float
with open('neo.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    str_data = list(csv_reader)[1:]  # Skip header row
	
    # Mapping for boolean values
    m1 = {'False': 0, 'True': 1}
    
    # Convert data into dataset with specified columns as float and the last column as integer
    dataset = [[
        float(i[2]),
        float(i[3]),
        float(i[4]),
        float(i[5]),
        float(i[8]),
        m1.get(i[9], i[9]),
    ] for i in str_data]

# Separate features and labels for each data item
X, y = [row[:-1] for row in dataset], [row[-1] for row in dataset]

# Shuffle the dataset
X, y = shuffle(X, y, random_state=0)

# Generate training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)

class NeuralNetwork:
    def __init__(self, layer_dimensions=[25,16,16,1],learning_rate=0.00001):
        """
        Parameters
        ----------
        layer_dimensions : list
            Dimension of each layer
        learning_rate :  float
            learning rate of the network.
        """
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        
    def initialize_parameters(self):
        """initializes the parameters"""
        np.random.seed(3)
        self.n_layers =  len(self.layer_dimensions)
        for l in range(1, self.n_layers):
            vars(self)[f'W{l}'] = np.random.randn(self.layer_dimensions[l], self.layer_dimensions[l-1]) * 0.01
            vars(self)[f'b{l}'] = np.zeros((self.layer_dimensions[l], 1))
    
    def _forward_propagation(self,A_prev ,W ,b , activation):
        """
        forward propagation for single layer
        Arguments:
        A_prev : activations from previous layer
        W -- shape : (size of current layer, size of previous layer)
        b -- shape : (size of the current layer, 1)
        activation -- the activation to be used in this layer

        Returns:
        A -- the output of the activation function 
        cache -- tuple containing "linear_cache" () and "activation_cache" for backpropagation
        """
        
        # Compute Z using the function defined above, compute A using the activaiton function
        if activation == "sigmoid":
            Z, linear_cache = np.dot(W,A_prev) + b,(A_prev, W, b)
            A, activation_cache = sigmoid(Z) 
        elif activation == "relu":
            Z, linear_cache = np.dot(W,A_prev) + b,(A_prev, W, b)
            A, activation_cache = relu(Z) 
            # cache for backpropagation
        cache = (linear_cache, activation_cache)
        return A, cache
    
    
    def forward_propagation(self, X):
        """
        forward propagation for the whole network
        Arguments:
        X --  shape : (input size, number of examples)
        Returns:
        AL -- last post-activation value
        caches -- list of cache returned by _forward_propagation helper function
        """
        # Initialize empty list to store caches
        caches = []
        # Set initial A to X 
        A = X
        L =  self.n_layers -1
        for l in range(1, L):
            A_prev = A 
            # Forward propagate through the network except the last layer
            A, cache = self._forward_propagation(A_prev, vars(self)['W' + str(l)], vars(self)['b' + str(l)], "relu")
            caches.append(cache)
        # Forward propagate through the output layer and get the predictions
        predictions, cache = self._forward_propagation(A, vars(self)['W' + str(L)], vars(self)['b' + str(L)], "sigmoid")
        # Append the cache to caches list recall that cache will be (linear_cache, activation_cache)
        caches.append(cache)
        return predictions, caches
    
    def compute_cost(self, predictions, y):
        """
        Implements the cost function 
        Arguments:
        predictions -- The model predictions, shape : (1, number of examples)
        y -- The true values, shape : (1, number of examples)
        Returns:
        cost -- cross-entropy cost
        """
        # Get number of training examples
        m = y.shape[0]
        # Compute cost we're adding small epsilon for numeric stability
        cost = (-1/m) * (np.dot(y, np.log(predictions+1e-9).T) + np.dot((1-y), np.log(1-predictions+1e-9).T))
        # squeeze the cost to set it into the correct shape 
        cost = np.squeeze(cost)
        return cost   
    
            
    def _back_propagation(self, dA, cache, activation):
        """
        Implements the backward propagation for a single layer.

        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) 
        activation -- the activation to be used in this layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        # get the cache from forward propagation and activation derivates function
        linear_cache, activation_cache = cache
        # compute gradients for Z depending on the activation function
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        # Compute gradients for W, b and A 
        
        A_prev, W, b = linear_cache
        m=A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True, initial=0)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev, dW, db

    def back_propagation(self, predictions, Y, caches):
        """
        Implements the backward propagation for the NeuralNetwork
        Arguments:
        Prediction --  output of the forward propagation 
        Y -- true label
        caches -- list of caches
        """
        L =  self.n_layers - 1
        # Get number of examples
        m = predictions.shape[1]
        Y = Y.reshape(predictions.shape) 
        # Initializing the backpropagation we're adding a small epsilon for numeric stability 
        dAL = - (np.divide(Y, predictions+1e-9) - np.divide(1 - Y, 1 - predictions+1e-9))
        current_cache = caches[L-1] # Last Layer
        # Compute gradients of the predictions
        vars(self)[f'dA{L-1}'], vars(self)[f'dW{L}'], vars(self)[f'db{L}'] = self._back_propagation(dAL, current_cache, "sigmoid")
        for l in reversed(range(L-1)):
            # update the cache
            current_cache = caches[l]
            # compute gradients of the network layers 
            vars(self)[f'dA{l}'] , vars(self)[f'dW{l+1}'], vars(self)[f'db{l+1}'] = self._back_propagation(vars(self)[f'dA{l + 1}'], current_cache, activation = "relu")
            
    def update_parameters(self):
            """
            Updates parameters using gradient descent
            """
            L = self.n_layers - 1
            # Loop over parameters and update them using computed gradients
            for l in range(L):
                vars(self)[f'W{l+1}'] = vars(self)[f'W{l+1}'] - self.learning_rate * vars(self)[f'dW{l+1}']
                vars(self)[f'b{l+1}']  = vars(self)[f'b{l+1}'] - self.learning_rate * vars(self)[f'db{l+1}']
                
    def fit1(self, X, y,X_test,y_test, epochs=10, batch_size=128, print_cost=True):
        """
        Trains the Neural Network 

        Arguments:
        X -- input data
        y -- true "label" 
        epochs -- number of iterations of the optimization loop
        batch_size -- size of each batch
        print_cost -- If set to True, this will print the cost every 100 iterations 
        """
        # Transpose X to get the correct shape
        X = X.T
        X_test = X_test.T
        np.random.seed(1)
        # Create empty array to store the costs
        costs = [] 
        # Get number of training examples
        m = X.shape[1]  
        m1 = X_test.shape[1]  
        # Initialize parameters 
        self.initialize_parameters()
        # Loop for the specified number of epochs
        for epoch in range(epochs):
            epoch_cost = 0
            epoch_cost1=0
            # Shuffle the training data
            permutation = np.random.permutation(m)
            shuffled_X = X[:, permutation]
            shuffled_Y = y[permutation]
            permutation_t = np.random.permutation(m1)
            shuffled_X_t = X_test[:, permutation_t]
            shuffled_Y_t = y_test[permutation_t]
            print("epoch: {}".format(epoch))
            print('-'*60)
            # Iterate over each batch
            for i in range(0, m, batch_size):
                # Get the current batch
                X_batch = shuffled_X[:, i:i+batch_size]
                Y_batch = shuffled_Y[i:i+batch_size]
            
                # Forward propagate and get the predictions and caches
                predictions, caches = self.forward_propagation(X_batch)
            
                # Compute the cost function
                cost = self.compute_cost(predictions, Y_batch)
                epoch_cost += cost
            
                # Backpropagation
                self.back_propagation(predictions, Y_batch, caches)
            
                # Update parameters
                self.update_parameters()
                if i % 100 == 0:
                    current = i+len(X_batch[0])
                    print(f"Loss: {cost}  [{current:>5d}/{m:>5d}]")
            # Compute average cost for the epoch
            epoch_cost /= (m / batch_size)
        
            # Print the cost every 10000 training examples
            if print_cost :
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if print_cost:
                costs.append(epoch_cost)
            correct=0
            for i in range(0, m1, batch_size):
                # Get the current batch
                X_t_batch = shuffled_X_t[:, i:i+batch_size]
                Y_t_batch = shuffled_Y_t[i:i+batch_size]
            
                # Forward propagate and get the predictions and caches
                predictions, caches = self.forward_propagation(X_t_batch)
                correct += (predictions.argmax(1) == Y_t_batch).sum()
                # Compute the cost function
                cost = self.compute_cost(predictions, Y_t_batch)
                epoch_cost1 += cost
                
                # Backpropagation
                #self.back_propagation(predictions, Y_batch, caches)
            
                # Update parameters
                #self.update_parameters()
                if i % 100 == 0:
                    current = i+len(X_t_batch[0])
                    print(f"Loss: {cost}  [{current:>5d}/{m1:>5d}]")
            # Compute average cost for the epoch
            epoch_cost1 /= (m1 / batch_size)
            correct /= m1
            if print_cost :
                print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {epoch_cost1:>8f} \n")
    def predict(self,X,y):
        """
        uses the trained model to predict given X value

        Arguments:
        X -- data set of examples you would like to label
        y -- True values of examples; used for measuring the model's accuracy
        Returns:
        predictions -- predictions for the given dataset X
        """
        X = X.T
        # Get predictions from forward propagation
        predictions, _ = self.forward_propagation(X)
        # Predictions Above 0.5 are True otherwise they are False
        predictions = (predictions > 0.5)
        # Squeeze the predictions into the correct shape and cast true/false values to 1/0
        predictions = np.squeeze(predictions.astype(int))
        #Print the accuracy
        return np.sum((predictions == y)/X.shape[1]), predictions.T
def relu_backward(dA, cache):
    """
    backward propagation for a single ReLU unit.
    Arguments:
    dA -- post-activation gradient
    cache -- 'Z'  stored for backpropagation
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) 
    # When z <= 0, dz is equal to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    backward propagation for a single sigmoid unit.
    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' stored during forward pass

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
def sigmoid( Z):
    """
    single sigmoid funtion .
    Arguments:
    Z -- input

    Returns:
    Z -- input as cache for backword propagation
    A -- Post -activation value
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    single relu funtion .
    Arguments:
    Z -- input

    Returns:
    Z -- input as cache for backword propagation
    A -- Post -activation value
    """
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache


print("Training Neural Network using 0 hidden Layer")
# create model instance with the given hyperparameters
layers = [5,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
print("Training Neural Network using 1 hidden Layer with 16 ")
layers = [5,16,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
print("Training Neural Network using 1 hidden Layer with 32")
layers = [5,32,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
def relu_backward(dA, cache):
    """
    backward propagation for a single ReLU unit.
    Arguments:
    dA -- post-activation gradient
    cache -- 'Z'  stored for backpropagation
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) 
    # When z <= 0, dz is equal to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    backward propagation for a single sigmoid unit.
    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' stored during forward pass

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
def sigmoid( Z):
    """
    single sigmoid funtion .
    Arguments:
    Z -- input

    Returns:
    Z -- input as cache for backword propagation
    A -- Post -activation value
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    single relu funtion .
    Arguments:
    Z -- input

    Returns:
    Z -- input as cache for backword propagation
    A -- Post -activation value
    """
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache


print("Training Neural Network using 0 hidden Layer")
# create model instance with the given hyperparameters
layers = [5,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
print("Training Neural Network using 1 hidden Layer with 16 ")
layers = [5,16,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
print("Training Neural Network using 1 hidden Layer with 32")
layers = [5,32,1]
model = NeuralNetwork(learning_rate=0.001,layer_dimensions=layers)
model.fit1(np.array(X_train),np.array(y_train),np.array(X_test), np.array(y_test),epochs=5,print_cost=True)
accuracy, predictions = model.predict(np.array(X_test), np.array(y_test) )
print(classification_report(y_test, predictions, zero_division=1))
