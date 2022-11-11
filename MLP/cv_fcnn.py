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
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

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
    
class FCNN(nn.Module):
  '''
    Simple Fully Connected Neural Network
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(250, 2000),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(2000),
      nn.Dropout(0.3),
      nn.Linear(2000, 1000),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(1000),
      nn.Dropout(0.3),
      nn.Linear(1000, 500),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(500),
      nn.Dropout(0.3),
      nn.Linear(500, 250),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(250),
      nn.Linear(250, 1)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':  
  # Configuration options
  k_folds = 5
  num_epochs = 100
  loss_function = nn.BCEWithLogitsLoss()  
  # For fold results
  results = {}  
  # Set fixed random number seed
  torch.manual_seed(5467178)
  data = np.load("XYu5x5x5.npz")
  X, Y = data['X'], data['Y']
  X = MinMaxScaler().fit_transform(X)
  #X = StandardScaler().fit_transform(X)
  over = BorderlineSMOTE(random_state=46, k_neighbors=1)
  input_sm, output_sm = over.fit_resample(X, Y)
  X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(input_sm, output_sm, test_size=0.2, random_state=450)
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=13422349)
  np.savez("XY_test_0.2_5x5x5.npz", X=X_test, Y=y_test)
  input, output = X_train_sm, y_train_sm  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
  #kfold = ShuffleSplit(n_splits=k_folds, test_size=.30, random_state=0)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(input, output)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    dataset = TensorDataset(torch.FloatTensor(input), torch.FloatTensor(output))        
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=1000, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=2000, sampler=test_subsampler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Init the neural network
    model = FCNN().to(device)
    print(model)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        model.train()
        # Get inputs      
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients
        model.zero_grad()
        
        # Perform forward pass
        outputs = model(inputs)
        
        targets = targets.unsqueeze(1)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0
      loss_final = current_loss/len(trainloader) 
      scheduler.step(loss_final)      
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-cnn-0.2-5x5x5-fold-{fold}.pth'
    #save_path = 'entire_model.pt'
    #torch.save(network, save_path)
    torch.save(model.state_dict(), save_path)

    # Evaluationfor this fold
    #correct, total = 0, 0
    predicted_list = []
    Target = []
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):
        model.eval()
        # Get inputs        
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate outputs
        outputs = model(inputs)

        # Set total and correct
        #_, predicted = torch.max(outputs.data, 1)
        predicted = torch.round(torch.sigmoid(outputs.data))
        #total += targets.size(0)
        #correct += (predicted == targets.long()).sum().item()
        predicted_list.append(predicted.cpu().numpy())
        Target.append(targets.cpu().numpy())
        #print(len(predicted_list))
        #print(len(Target))

      # Print accuracy
      #print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('----------------------predicted_list---------')
      #results[fold] = 100.0 * (correct / total)
    predicted_list = [a.squeeze().tolist() for a in predicted_list]
    Target = [a.squeeze().tolist() for a in Target]
    #print(len(predicted_list))
    #print(len(Target))      
    predicted_list = flatten_list(predicted_list)
    Target = flatten_list(Target)
    #print(len(predicted_list))
    #print(len(Target))    
    predicted_list = np.array(predicted_list)  
    Target = np.array(Target)
    a = np.count_nonzero(Target)
    b = np.count_nonzero(predicted_list)
    print(a,b)  
    #print(predicted_list.shape)
    #print(Target.shape)     
    CM = confusion_matrix(Target, predicted_list)
    print(CM)
    sc = roc_auc_score(Target, predicted_list)
    print(sc)    
    fpr, tpr, thresholds = roc_curve(Target, predicted_list)
    #plt.plot(fpr, tpr, linestyle='--', label='No Skill')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.legend()
    #plt.savefig('ROC.png')    
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  print('Finished 5-fold CV successfully')
  #sum = 0.0
  #for key, value in results.items():
    #print(f'Fold {key}: {value} %')
    #sum += value
  #print(f'Average: {sum/len(results.items())} %')