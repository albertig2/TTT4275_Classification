import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay 
from common_utilities.evaluation import generate_confusion_matrix_and_error_rates


def MSE(G, T):
    """
    Input: C x N matrices G and T 
    Output: The MSE scalar
    """
    MSE = 0
    for col in range(np.shape(G)[1]):
        MSE += (G[:,col] - T[:,col]).T @ (G[:,col] - T[:,col])
    return 1/2 * MSE

def grad_MSE(G, T, X):
    """
    Input: (DIM + 1) x N matrix X and C x N matrix T 
    Output: C x (DIM + 1) matrix dMSE/dW
    """
    # DIM = np.shape(X)[0] - 1; C = np.shape(G)[0]
    # grad_MSE = np.zeros((C, DIM + 1)) # Same shape as G[i] x X[i]^T = C x 1 x 1 x (DIM+1) = C x (DIM + 1)
    # N_tr = np.shape(X)[1] # The amount of datapoints available for training
    # for i in range(N_tr):
    #     grad_MSE += np.reshape((G[:, i]-T[:, i]) * G[:, i] * (1-G[:, i]), (C,1)) @ np.reshape(X[:, i], (DIM+1, 1)).T
    # return grad_MSE
    delta = (G - T) * G * (1 - G)  # (C, N) elementwise
    return delta @ X.T 

#Model-training
def train_model(X_tr, T):
    """
    Trains the model W based on the input data X and the solution provided by T.
    Input: (DIM + 1) x N matrix X and C x N matrix T
    Output: C x (DIM + 1) matrix W 
    """
    DIM = np.shape(X_tr)[0] - 1; C = np.shape(T)[0]
    W = np.zeros((C, DIM + 1)) # G[i] = C x 1 and X[i] = (DIM + 1) x 1 and G[i]= WX[i] => W = C x (DIM + 1)
    G = 1 / (1 + np.exp(-W @ X_tr)) # Same shape as W @ X naturally. Start with the G-values that correspond to W0 

    alpha = 0.001 # Tuned this value...
    tresh = 10**(-6) #* 10 # Stop when we no longer get that previous MSE is more than tresh * 100% greater than the new one..
    prev_err = 1000000; curr_err = 0.9*prev_err
    rel_err = (prev_err - curr_err) / curr_err #Checks improvement in MSE

    while (rel_err > tresh):# > tresh):
        W = W - alpha * grad_MSE(G, T, X_tr)
        Z = W @ X_tr
        G = 1 / (1 + np.exp(-Z))
        prev_err = curr_err
        curr_err = MSE(G, T)
        rel_err = (prev_err - curr_err) / curr_err #Checks improvement in MSE
        #print(rel_err*100, "%")
    return W


#Model-testing


