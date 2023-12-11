# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:38:45 2023

@author: Pooya
"""

import torch
import numpy as np
from torch import nn
from math import pi
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

#############    Question:
    ####      How this model knows which weights to change in back
    ####   propagation? Which rows of item and user should change to 
    ####   decrease MSE? How does the model understand these? bc in each
    ####   batch there are different rows of user and item are selected.
    ####   In fact how does it calculate eij(error of predicted rate of
    ####   user i and item j) when it is not available?
    ####   model(batch_features) only calculates error for users and
    ####   items available in the batch. How does it calculate for
    ####   those which are not available to update the embeddings?
######   Regression of Sin(x):
    
class WideNet(nn.Module):
  """
   A Wide neural network with a two hidden layers
   Structure.
  """
  
  def __init__(self, n_user, n_item, k=10, c_vector=1.0, c_bias=1.0,):
      super(WideNet, self).__init__()

      # These are the hyper-parameters
      self.k = k
      self.n_user = n_user
      self.n_item = n_item
      self.c_vector = c_vector

      # The embedding matrices for user and item are learned and
      # fit by PyTorch
      
      # biases are used to remove the effect of some users give high
      # retes or low rates. Some items get higher rate
      
      self.user = nn.Embedding(n_user, k)
      self.item = nn.Embedding(n_item, k)
      self.item_bias = nn.Embedding(n_item,1)
      self.user_bias = nn.Embedding(n_user,1)
      
      self.bias = nn.Parameter(torch.ones(1))
      
  def forward(self, train_x):
      
      user_id = train_x[:, 0]
      # These are the item indices, correspond to the "i" variable
      item_id = train_x[:, 1]

      # Initialize a vector user = p_u using the user indices
      vector_user = self.user(user_id)
      # Initialize a vector item = q_i using the item indices
      vector_item = self.item(item_id)
      
      bias_vec_user = self.user_bias(user_id).squeeze()
      bias_vec_item = self.item_bias(item_id).squeeze()

      # The user-item interaction: p_u * q_i is a dot product
      # between the 2 vectors above
      ui_interaction = torch.sum(vector_user * vector_item, dim=1)
      prediction = ui_interaction + bias_vec_user + bias_vec_item + \
          self.bias
      return prediction
  
  def loss(self, prediction, target):
            """
            Function to calculate the loss metric
            """
            # Calculate the Mean Squared Error between target = R_ui
            # and prediction = p_u * q_i
            loss_mse = F.mse_loss(prediction, target.squeeze())

            # Compute L2 regularization over user (P) and item (Q)
            # matrices
            
            # for c_vector explore values between .001, .1 to 1000
            # do exploration with grid search, cross validation
            prior_user = l2_regularize(self.user.weight) * self.c_vector
            prior_item = l2_regularize(self.item.weight) * self.c_vector

            prior_it_bias = l2_regularize(self.item_bias.weight) * \
                self.c_vector
            prior_us_bias = l2_regularize(self.user_bias.weight) * \
                self.c_vector
            # Add up the MSE loss + user & item regularization
            total = loss_mse + prior_user + prior_item + prior_it_bias \
                + prior_us_bias

            return total
###### l2-regularization:
    ##  "l2_regularize" typically refers to L2 regularization, also
    ## known as weight decay. L2 regularization is a technique used to
    ## prevent overfitting in a model by adding a penalty term to the
    ## loss function based on the squared magnitude of the model's
    ## weights. It penalizes large weights in the model, discouraging
    ## the model from relying too much on any particular feature and
    ## promoting a more balanced use of features.
    
def l2_regularize(array):
    """
    Function to do L2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss





def train(features, labels, model, optimizer, scheduler,
             n_epochs, batch_size):

  loss_record = []  # Keeping recods of loss

  for epoch in range(n_epochs):
      epoch_loss = 0.0
      for batch_start in range(0, len(features), batch_size):
          batch_features = features[batch_start:batch_start +\
                                    batch_size]
          batch_labels = labels[batch_start:batch_start +\
                                batch_size]

          optimizer.zero_grad()
          predictions = model(batch_features)
          loss = model.loss(predictions, batch_labels)
          loss.backward()
          optimizer.step()

          epoch_loss += loss.item()

      scheduler.step()  # Update the learning rate scheduler
      loss_record.append(\
                      epoch_loss / (len(features) / batch_size))
      print('epoch is: ',epoch+1, 'loss is: ',
            epoch_loss / (len(features) / batch_size))

  return loss_record
  


full_data = np.load('dataset.npz')
train_x = full_data['train_x'].astype(np.int64)
train_y = full_data['train_y']

test_x = full_data['test_x'].astype(np.int64)
test_y = full_data['test_y']

n_user = int(full_data['n_user'])
n_item = int(full_data['n_item'])
k = 10  # Number of dimensions per user, item
c_vector = 1e-6  # regularization constant
c_bias = 1e-6 
lr = 0.003
batch_size = 1024

model = WideNet(n_user, n_item, k=k, c_vector=c_vector, c_bias = c_bias)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

torch.manual_seed(1)
epochs = 50
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

losses = train(train_x, train_y, model, optimizer, scheduler,
               epochs, batch_size)

# Save the model to a separate folder

# torch.save(model.state_dict(), '../models/vanilla_mf.pth')
    

#########    Evaluation:
    
    
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
predictions = model(test_x)
loss = model.loss(predictions,test_y)

print('Test lost is: ', loss)

########     To obtain the estimated rating for users and all movies:

entra = torch.arange(n_user)

prod = torch.arange(n_item)
    
with torch.no_grad():
    user_vectors_to_predict = model.user(entra)
    item_vectors_to_predict = model.item(prod)
    predictions = user_vectors_to_predict @ item_vectors_to_predict.T

print("Predictions:", predictions)

predictions.max()

predictions.min()