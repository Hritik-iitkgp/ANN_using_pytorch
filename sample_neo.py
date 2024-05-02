import csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read csv file into list and convert elements to int
with open('neo.csv', 'r') as read_obj:
	csv_reader = csv.reader(read_obj)
	str_data = list(csv_reader)[1:]
	
	
	m1 = {'False': 0, 'True': 1}
	
	'''
	# View data distribution
	l1 = np.transpose(str_data)
	for i, row in enumerate(l1):
		if i in [9]:
			print("Row:", i)
			print(collections.Counter(row))
	'''
	
	dataset = [[
					# m1.get(i[0], i[0]),
					float(i[2]),
					float(i[3]),
					float(i[4]),
					float(i[5]),
					float(i[8]),
					m1.get(i[9], i[9]),
				] for i in str_data]
	
# Separate labels and features for each data item
# print(dataset[:5])
X, y = [row[:-1] for row in dataset], [row[-1] for row in dataset]
X, y = shuffle(X, y, random_state=0)
# Generate training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)

# Pytorch code starts here
# web: https://pytorch.org/tutorials/beginner/basics

# Dataset creation
class MyDataset(Dataset):
	def __init__(self, data, targets):
		self.data = torch.FloatTensor(data)
		self.targets = torch.LongTensor(targets)
		
	def __getitem__(self, index):
		x = self.data[index]
		y = self.targets[index]
		
		return x, y
	
	def __len__(self):
		return len(self.data)


batch_size = 128
train_dataset = MyDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MyDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Not reusable section starts here

# Neural network
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		
		self.l1 = nn.Linear(5, 2)

	def forward(self, x):
		x = self.l1(x)
		return F.log_softmax(x, dim=1)

model = NeuralNetwork()
model = model.to(device)

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X.to(device))
		loss = loss_fn(pred, y.to(device))

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Not reusable section ends here

def test_loop(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X.to(device))
			test_loss += loss_fn(pred, y.to(device)).item()
			correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Training
epochs = 4
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(train_dataloader, model, loss_fn, optimizer)
	test_loop(test_dataloader, model, loss_fn)
print("Training done!")

# Prediction
y_pred = []
with torch.no_grad():
	for X, y in test_dataloader:
		y = model(X.to(device)).argmax(1).cpu().tolist()
		y_pred.extend(y)
		
# Pytorch code ends here

print(classification_report(y_test, y_pred, zero_division=1))
