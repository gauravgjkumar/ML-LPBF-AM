from os.path import join as pjoin
import scipy.io as sio
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from cv_fcnn import FCNN
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from time import time
import matplotlib.pyplot as plt

def Smooth(data, normal=True):
  if normal:
    minm = torch.min(data)
    maxm = torch.max(data)
    normalized = torch.div(data - minm, maxm - minm)
  else:
    mean = torch.mean(data)
    std = torch.std(data)
    normalized = torch.div(data - mean, std)
  return normalized


data = np.load('XY_test_0.2_5x5x5.npz')
X, Y = data['X'], data['Y']
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
input, output = X, Y
dataset = TensorDataset(torch.FloatTensor(input.float()), torch.FloatTensor(output.float()))
test_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=2000, shuffle=True)

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)    
    return acc

model = FCNN()
model.load_state_dict(torch.load('model-cnn-0.2-5x5x5-fold-1.pth'))

predicted_list = []
Target = []
model.eval()
with torch.no_grad():

  # Iterate over the test data and generate predictions
  for i, data in enumerate(test_loader, 0):    
    # Get inputs        
    inputs, targets = data
    #inputs, targets = inputs.to(device), targets.to(device)

    # Generate outputs
    outputs = model(inputs)

    # Set total and correct
    #_, predicted = torch.max(outputs.data, 1)
    #outputs = torch.sigmoid(outputs)
    #outputs_tag = torch.round(outputs)
    predicted = torch.round(torch.sigmoid(outputs.data))
    predicted_list.append(predicted.cpu().numpy())
    Target.append(targets.cpu().numpy())    
  predicted_list = [a.squeeze().tolist() for a in predicted_list]
  Target = [a.squeeze().tolist() for a in Target]   
  predicted_list = flatten_list(predicted_list)
  Target = flatten_list(Target)
  predicted_list = np.array(predicted_list)  
  Target = np.array(Target)
  #a = np.count_nonzero(Target)
  #print(a)  
  #print(predicted_list.shape)
  #print(Target.shape)
a = np.count_nonzero(Target)
b = np.count_nonzero(predicted_list)
print(a,b)
CM = confusion_matrix(Target, predicted_list)
print(CM)
sc = roc_auc_score(Target, predicted_list)
print(sc)
print(predicted_list.shape)
ns_probs = [0 for _ in range(len(Target))]
predicted_list = torch.from_numpy(predicted_list)
lr_probs = torch.sigmoid(predicted_list)
ns_auc = roc_auc_score(Target, ns_probs)
lr_auc = roc_auc_score(Target, lr_probs)
ns_fpr, ns_tpr, _ = roc_curve(Target, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Target, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr,label='mlp_5x5x5_0.2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.savefig('MLPNN_5x5x5_0.2_ROC.png')     