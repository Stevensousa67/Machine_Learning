import numpy as np

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z)= 1 / (1 + e^(-z)) entrywise
    """

    p = 1 / (1 + np.exp(-z))     
    
    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # the dot product of X and theta
    result = np.dot(X,theta)
    # the logistic function of the dot product
    func = logistic(result)
    # the log of the logistic function
    log1 = np.log(func)
    log2 = np.log(1-func)
    # the different parts of the cost function
    part1 = np.dot(log1.T,-y)
    part2 = np.dot(log2.T,1-y)
    # finding the cost
    cost = part1 - part2
    
    return cost

def gradient_descent( X, y, theta, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    
    
    # the array for the cost values
    J_history = np.zeros(iters)
    
    for i in range(0,iters):
        
        # the dot product of X and theta
        result = np.dot(X,theta)
        # the logistic function of the dot product
        func = logistic(result)
        
        # the difference between the model and the actual values
        cost = func - y
        
        # calculating the partial derivative
        partial = np.dot(X.T,cost)
        
        # TODO: performing gradient descent on theta
        theta = theta - (alpha * partial)
               
        # TODO: adding the cost to the cost array items (J-history[i]) by calling cost_function from top
        J_history[i] = cost_function(X, y, theta)
    
    return theta, J_history