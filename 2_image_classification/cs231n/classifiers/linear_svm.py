from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        grad = 0.0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            # print (margin)
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                grad += 1.0
        # print(len(dW[i]))
        dW[:,y[i]] -= grad * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_train
    dW += 2 * reg * W # Regularisation
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    print (X.shape)
    print (W.shape)
    margin = 0.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        grad = 0.0
        margin = scores - correct_class_score + 1
        loss_margin = np.delete(margin, y[i])
        # print(margin)
        # print()/
        loss += np.sum(loss_margin[np.where(loss_margin > 0)])
        # print(np.where(margin > 0))
        dW[np.where(margin > 0)] += X[i]
        dW[y[i]] -= (len(margin[np.where(margin > 0)])) * X[i]

    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW = dW.T
    dW += 2 * reg * W # Regularisation
    """

    # scores = X.dot(W)
    # print(scores.shape)
    # print(list(range(num_train)))
    # correct_class_scores = scores[:,y]
    #
    #
    # margin = scores - correct_class_scores + 1
    # print(margin)
    # loss = np.sum(margin[np.where(margin > 0)])

    s = X.dot(W)
    # read correct scores into a column array of height N
    correct_score = s[list(range(num_train)), y]
    print(correct_score.shape)
    correct_score = correct_score.reshape(num_train, -1)
    print(correct_score.shape)
    # subtract correct scores from score matrix and add margin
    s += 1 - correct_score
    # make sure correct scores themselves don't contribute to loss function
    s[list(range(num_train)), y] = 0
    # construct loss function
    loss = np.sum(np.fmax(s, 0)) / num_train
    loss += reg * np.sum(W * W)

    # loss = np.sum(margin[np.where(margin > 0)])
    # loss /= num_train

    # Add regularization to the loss.
    # loss += reg * np.sum(W * W)
    # print(margin.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
