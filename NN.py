import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


data = pd.read_excel("GHED_data.xlsx", 'Data')


rfdata = pd.get_dummies(data = data.filter(data.columns[np.r_[0:6,10:25]]), columns = [data.columns[3]], drop_first=True, dtype=int)
rfdata = rfdata.dropna(how="any")
rfdata
# the value we want to predict
rfdata_label = np.array(pd.factorize(rfdata.iloc[:,1])[0]+1)
rfdata_label
rfdata_features = rfdata.filter(rfdata.columns[3:24])
rfdata_features["region"] = pd.factorize(rfdata.iloc[:,2])[0]+1
rfdata_features
feature_names = list(rfdata_features)
feature = np.array(rfdata_features)


X_train, X_test, y_train, y_test = train_test_split(feature, rfdata_label, test_size = 0.25, random_state = 42)

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break

import torch
from torch import nn
from torch import optim

input_dim = 21
hidden_dim = 100
output_dim = 166


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x
       
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print(model)


learning_rate = 0.1

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.type(torch.LongTensor))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")