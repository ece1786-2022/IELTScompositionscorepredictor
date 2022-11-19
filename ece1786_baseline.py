# -*- coding: utf-8 -*-
"""ECE1786_Baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14D-sLnt6Czdj1Y1F7z-KojnvvPvSt1vP
"""

import torch
import torchtext
from torchtext import data
import torch.optim as optim
import argparse
import os
import pandas as pd
import csv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Colab\ Notebooks/

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Using device:", device)

tsv_file = open("IELTS_dataset.csv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
df = pd.read_csv('IELTS_dataset.csv')
tsv_file = open("IEL.csv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
df = pd.read_csv('IEL.csv')
df = df.iloc[:, 0:3]
df=df.dropna()
print(df.head)
class2idx = {
    '<4':0,
    '4':1,
    '4.5':2,
    '5':3,
    '5.5':4,
    '6':5,
    '6.5':6,
    '7':7,
    '7.5':8,
    '8':9,
    '8.5':10,
    '9':11
}
for i in class2idx.values():
    print (type(i))
idx2class = {v: k for k, v in class2idx.items()}
df['score'].replace(class2idx, inplace=True)
df = df.astype({'score': int})
print(df.head)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, df):
        # X: torch.tensor (maxlen, batch_size), padded indices
        # Y: torch.tensor of len N
        X1, X2, Y = [], [], []
        V = len(vocab.vectors)
        for i, row in df.iterrows():
            L1 = row["Topic"].split()
            X1.append(torch.tensor([vocab.stoi.get(w, V-1) for w in L1]))
            L2 = row["Content"].split()
            X2.append(torch.tensor([vocab.stoi.get(w, V-1) for w in L2]))
            Y.append(row["score"])
        self.X1 = X1 
        self.X2 = X2
        self.Y = torch.tensor(Y)
        
    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx]

train_ratio = 0.80
validation_ratio = 0.05
test_ratio = 0.15
x_train, x_test = train_test_split(df, test_size=1 - train_ratio)
x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio))

print(x_train)

def my_collate_function(batch, device):
    batch_x1, batch_x2, batch_y = [], [], []
    max_len_x1 = 0
    max_len_x2 = 0
    for x1,x2,y in batch:
        batch_y.append(y)
        max_len_x1 = max(max_len_x1, len(x1))
        max_len_x2 = max(max_len_x2, len(x2))
    for x1,x2,y in batch:
        x_p = torch.concat(
            [x1, torch.zeros(max_len_x1 - len(x1))]
        )
        x_p2 = torch.concat(
            [x2, torch.zeros(max_len_x2 - len(x2))]
        )
        batch_x1.append(x_p)
        batch_x2.append(x_p2)
    return torch.stack(batch_x1).t().int().to(device), torch.stack(batch_x2).t().int().to(device), torch.tensor(batch_y).to(device)

train_dataset = TextDataset(glove, x_train)
val_dataset = TextDataset(glove, x_val)
test_dataset = TextDataset(glove, x_test)
batch_size = 10
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=lambda batch: my_collate_function(batch, device))

validation_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=lambda batch: my_collate_function(batch, device))

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: my_collate_function(batch, device))

class cnn(torch.nn.Module):
    def __init__(self,k1,n1,k2,n2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors,freeze = True)
        self.cnn1 = nn.Conv2d(1,n1,(100,k1),bias = False)
        self.cnn2 = nn.Conv2d(1,n2,(100,k2),bias = False)
        self.hidden=nn.Linear(n1+n2,100)
        self.output=nn.Linear(100,12)
    def forward(self,x1,x2):
        e1 = self.embedding(x1)
        e1 = torch.transpose(e1,1,2)
        e1 = e1.unsqueeze(1)
        e1 = F.relu(self.cnn1(e1))
        e1 = e1.squeeze(2)
        e2 = self.embedding(x2)
        e2 = torch.transpose(e2,1,2)
        e2 = e2.unsqueeze(1)
        e2 = F.relu(self.cnn2(e2))
        e2 = e2.squeeze(2)
        pool1 = nn.MaxPool1d(e1.shape[-1])
        pool2 = nn.MaxPool1d(e2.shape[-1])
        n1 = pool1(e1).squeeze(-1)
        n2 = pool2(e2).squeeze(-1)
        c = torch.cat((n1,n2),dim=1)
        c = self.hidden(c)
        c = F.relu(c)
        prediction = self.output(c)
        return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_cnn(learning_rate,epoches,k1,n1,k2,n2):
    model = cnn(k1,n1,k2,n2)
    model = model.to(device)
    print(count_parameters(model))
    loss_function = nn.CrossEntropyLoss()
    accuracy = Accuracy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iters_epoch  = []    
    losses_epoch = []
    accu_epoch = []
    validation_losses_epoch = []
    validation_accu_epoch = []
    for epoch in range(epoches):
        n_epoch=0
        n_epoch_test=0
        loss_train=0
        accu_train=0
        loss_test=0
        accu_test=0
        model.train()
        for batch in iter(train_dataloader):
            optimizer.zero_grad()
            x1 = batch[0]
            x1 = torch.transpose(x1,0,1)
            x2 = batch[1]
            x2 = torch.transpose(x2,0,1)
            y_pred = model(x1,x2)
            y = torch.tensor(batch[2])
            #y = y.unsqueeze(1)
            y_binary = y
            #y = y.float()
            loss = loss_function(input=y_pred, target=y)
            accu = accuracy(y_pred,y_binary)
            loss_train+=loss
            accu_train+=accu
            n_epoch += 1
            loss.backward()
            optimizer.step()
        iters_epoch.append(epoch)
        losses_epoch.append(float(loss_train)/n_epoch)
        accu_epoch.append(float(accu_train)/n_epoch)
        model.eval()
        for batch in iter(test_dataloader):
            x1 = batch[0]
            x1 = torch.transpose(x1,0,1)
            x2 = batch[1]
            x2 = torch.transpose(x2,0,1)
            y_pred = model(x1,x2)
            y = torch.tensor(batch[2])
            #y = y.unsqueeze(1)
            y_binary = y
            #y = y.float()
            loss = loss_function(input=y_pred, target=y)
            accu = accuracy(y_pred,y_binary)
            loss_test+=loss
            accu_test+=accu
            n_epoch_test+=1
        validation_losses_epoch.append(float(loss_test)/n_epoch_test)
        validation_accu_epoch.append(float(accu_test)/n_epoch_test)
    fig, ax = plt.subplots(2)
    ax[0].plot(iters_epoch, losses_epoch, label="Train Loss")
    ax[0].plot(iters_epoch, validation_losses_epoch, label="Test Loss")
    ax[1].plot(iters_epoch, accu_epoch, label="Train Accuracy")
    ax[1].plot(iters_epoch, validation_accu_epoch, label="Test Accuracy")
    ax[0].legend(loc="center right",bbox_to_anchor=(1,0.5))
    ax[0].legend()
    ax[1].legend(loc="center right",bbox_to_anchor=(1,0.5))
    ax[1].legend()
    return model
model_cnn_train = train_cnn(0.001,25,4,50,20,200)

torch.set_printoptions(precision=10)
loss_function = nn.CrossEntropyLoss()
accuracy = Accuracy().to(device)
for batch in iter(validation_dataloader):
    x1 = batch[0]
    x1 = torch.transpose(x1,0,1)
    x2 = batch[1]
    x2 = torch.transpose(x2,0,1)
    y_pred = model_cnn_train(x1,x2)
    y = torch.tensor(batch[2])
    print(y)
    print(y_pred)
    y_binary = y
    loss = loss_function(input=y_pred, target=y)
    accu = accuracy(y_pred,y_binary)
