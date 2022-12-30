from builtins import range
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
    num_train=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
      score=X[i].dot(W)
      dscore=X[i]
      score+=-np.max(score)
      sum1=np.sum(np.exp(score))
      loss+=np.log(sum1)-score[y[i]]
      dW[:,y[i]]+=-X[i]
      for j in range(num_class):
        dW[:,j]+=X[i]*np.exp(score[j])/sum1
    loss/=num_train
    dW/=num_train
    loss+=reg*np.sum(np.square(W))
    dW+=2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores=X.dot(W)
    scores+=-np.max(scores,axis=1).reshape(-1,1)
    loss=np.sum(np.log(np.sum(np.exp(scores),axis=1)))
    loss+=-np.sum(scores[range(num_train),y])
    loss/=num_train
    loss+=reg*np.sum(np.square(W))
    M1=np.zeros_like(scores)
    M1[range(num_train),y]=-1
    M2=np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
    dW=X.T.dot(M1+M2)/num_train+2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
