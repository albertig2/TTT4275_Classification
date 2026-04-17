import numpy as np

def get_predicted_classes(X, W):
    """
    Gets the predicted class for a given classifier W and input data X. 
    Output data is given as a C x N matrix, where C is the amount of classes to pick from, and N is the amount of points given in X.
    Each column represents one prediction, and only one row is 1, symbolizing the prediction lands at that corresponding class. 
    So [0,0,1]^T for instance means class 3 was predicted...
    """
    Z = W @ X
    G = 1 / (1 + np.exp(-Z))
    C = np.argmax(G, axis=0) 
    return C
