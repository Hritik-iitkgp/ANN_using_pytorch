# Objective:
- There are an infinite number of objects in outer space. Some of them are closer than we think.
Even though we might think that a distance of 70,000 km can not potentially harm us, but at an
astronomical scale, this is a very small distance and can disrupt many natural phenomena. These
objects/asteroids can thus prove to be harmful. Hence, it is wise to know what is surrounding us
and what can harm us amongst those. Thus, this dataset compiles the list of NASA certified
asteroids that are classified as the Near Earth Objects.
- In This We will implement some simple multi-layer perceptrons and neural networks.
- In particular, we shall be doing the following tasks.

1. Starter code provided does this using python, can be used: Randomly divide the dataset
into 80% training set and the rest as test set. Choose the important features from the
dataset by modifying relevant parts of the starter code. Choose a mini-batch size to divide
the dataset into batches.
2. Build the ANN model. These operations have been demonstrated in the starter code using
Pytorch.
a. Build the MLP classifiers by identifying the number of input and output nodes
required for the problem, and specifying the number of hidden layers as:
i. 0 hidden layers
ii. 1 hidden layer with 16 nodes
iii. 1 hidden layer with 32 nodes
b. Use Sigmoid or ReLU activation function for the input and hidden layers. Use ReLU
activation for the output layer.
c. Define the forward and backward operations for your network. They are required
for inference and weight updation of your model.
d. Define the training function to train the model using a forward and a backward
pass. Define the prediction function for obtaining the outputs from the network.
e. Compare the implementation of your model compared to that using the Pytorch
library, on the same dataset (code snippet provided).
4. Hyper-parameter tuning.
a. For each of the architectures, vary the learning rates in the order of 0.1, 0.01,
0.001, 0.0001, 0.00001. Plot graph for the results with respect to accuracy and
loss. (Learning rate vs accuracy/loss for each model).
b. Report test set accuracy for all the learning rates in a tabular form and identify the
best model.
5. Classification Report
a. Create a classification report for comparing the performance of your algorithm, for
your best performing algorithm in terms of accuracy, with that of the Pytorch
algorithm.
