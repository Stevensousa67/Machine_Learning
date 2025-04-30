import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings

def train_one_vs_all(X, y, num_classes, lambda_val):
    '''
    Train a one vs. all logistic regression
    
    Inputs: 
      X                data matrix (2d array shape m x n)
      y                label vector with entries from 0 to 
                       num_classes - 1 (1d array length m)
      num_classes      number of classes (integer)
      lambda_val       regularization parameter (scalar)

    Outputs:
      weight_vectors   matrix of weight vectors for each class 
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)                       
            
    '''
    
        
    # getting the size of the X matrix
    m,n = X.shape
    
    # the vector for the intercepts
    intercepts = np.zeros(num_classes)
    # the matrix for the weight vectors
    weight_vectors = np.zeros((n,num_classes))
    # the vector for the classes
    y_class = np.zeros(m)
    
    for i in range(num_classes):
        # checking if the class value is present in the y label vector and updating the class vector accordingly
        for j in range(len(y)):
            if y[j] == i:
                y_class[j] = 1
            else:
                y_class[j] = 0
        
        # TODO: calculating the respective weight vectors and the intercept values for every iteration
        # i by calling train_logistic_regression function:
        weights, intercept = train_logistic_regression(X, y_class, lambda_val)
        weight_vectors[:, i] = weights
        intercepts[i] = intercept    
    
    return weight_vectors, intercepts


def predict_one_vs_all(X, weight_vectors, intercepts):
    '''
    Train a one vs. all logistic regression
    
    Inputs: 
      X                data matrix (2d array shape m x n)
      weight_vectors   matrix of weight vectors for each class 
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)   
                       
    Outputs:
      predictions      vector of predictions for examples in X
                       (1d array length m)            
    '''    
    
    
    # getting the size of the X matrix
    m,n = X.shape
    
    # the predictions vector
    predictions = np.zeros(m)
    
    # the dot product of X and the weight vectors
    probs = np.dot(X,weight_vectors)
    # adding the intercept values
    result = probs + intercepts
    # getting the dot product of the result
    func = 1 / (1 + np.exp(-result))
    
    # TODO: finding the prediction for each row in the matrix
    # Hint: look up the np.argmax function. It can find the index of
    # the largest value in an array, or in each row/column of an array. You can use that function on func
    # matrix to get predictions[i] in every iteration.
    predictions = np.argmax(func, axis=1)
    
    return predictions


def train_logistic_regression(X, y, lambda_val):
    '''
    Train a regularized logistic regression model
    
    Inputs:
      X           data matrix (2d array shape m x n)
      y           label vector with 0/1 entries (1d array length m)
      lambda_val  regularization parameter (scalar)

    Outputs:
      weights     weight vector (1d array length n)
      intercept   intercept parameter (scalar)
    '''
    model = linear_model.LogisticRegression(C=2./lambda_val, solver='lbfgs')

    # call model.fit(X, y) while suppressing warnings about convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    weight_vector = model.coef_.ravel()
    intercept = model.intercept_
    return weight_vector, intercept

def display_data(X, im_width=None, return_mosaic=False):
    '''
    Display data rows as mosaic image
    '''
    
    m, n = X.shape
    
    if im_width is None:
        im_width = int(np.sqrt(n))
    
    im_height = int(n / im_width)
    
    if not im_width * im_height == n:
        raise ValueError('cannot determine image dimensions')

    X = X / (2*np.max(np.abs(X), axis=1, keepdims=True)) + 0.5

    # Compute rows, cols
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    display_rows = display_rows.astype('int')
    display_cols = display_cols.astype('int')
    
    fig = plt.figure(1, (6., 6.))
    
    # convert each row to image
    images = [X[i,:].reshape([im_height, im_width]) for i in range(m)]

    # pad images for nice display
    pad = 1
    images = [np.pad(images[i], (pad,0), 'constant') for i in range(m)]

    # Assemble the image into a mosaic
    rows = []
    for i in range(display_rows):
        row_start = i * display_cols
        row_end = (i+1) * display_cols

        im = np.concatenate(images[row_start:row_end], axis=1)

        # Build the row first as an array of the correct size
        row = np.zeros((im_height + pad, (im_width + pad)*display_cols))
        h,w = im.shape

        # Now populate it with the image
        row[:h, :w] = im
        rows.append(row)

    # Concatenate rows to get the final result
    mosaic = np.concatenate(rows, axis=0)

    plt.imshow(mosaic, cmap='gray', clim=[0,1])
    plt.axis('off')
    plt.show()

    if return_mosaic:
        return mosaic
    else:
        return