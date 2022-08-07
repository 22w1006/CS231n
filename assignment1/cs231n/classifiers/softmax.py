import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  score = X.dot(W) # (N, C) numpy array
  nor_score = score - np.max(score, axis = 1).T.reshape(num_train, 1)
  exp_nor_score = np.exp(nor_score)
  
  # compute loss
  for i in range(num_train):
      sum_i = np.sum(exp_nor_score[i])
      loss += -np.log(exp_nor_score[i][y[i]]*1.0/sum_i)
      
  loss /= num_train
  loss += reg * np.sum(np.square(W))
  
  # compute gradient
  for i in range(num_train):
      sum_i = np.sum(exp_nor_score[i])
      for j in range(num_classes):
          dW[:, j] += (exp_nor_score[i][j]/sum_i * X[i]).T
          if j == y[i]:
              dW[:, j] -= X[i].T
              
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # preparation
  num_train = X.shape[0] # N
  num_classes = W.shape[1] # C
  score = X.dot(W) # (N, C) numpy array
  exp_nor_score = np.exp(score - np.max(score, axis = 1).T.reshape(num_train, 1))
  
  # compute loss
  loss = np.sum(-np.log(exp_nor_score[xrange(num_train), y]*1.0/np.sum(exp_nor_score, axis = 1)))
  loss /= num_train
  loss += reg * np.sum(np.square(W))

  # compute gradient
  eff = (exp_nor_score[xrange(num_train), :].T*1.0/np.sum(exp_nor_score, axis = 1)).T
  eff[xrange(num_train), y] -= 1
  dW = X.T.dot(eff)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

