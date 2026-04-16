import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay 


c1 = np.loadtxt("Iris/class_1", delimiter=",")
c2 = np.loadtxt("Iris/class_2", delimiter=",")
c3 = np.loadtxt("Iris/class_3", delimiter=",")


N_tr = 30
c1_tr = c1[:N_tr]; c1_te = c1[N_tr:]
c2_tr = c2[:N_tr]; c2_te = c2[N_tr:]
c3_tr = c3[:N_tr]; c3_te = c3[N_tr:]


x1_tr = np.column_stack((c1_tr, np.ones(N_tr))).T # X = (DIM + 1) x N
x2_tr = np.column_stack((c2_tr, np.ones(N_tr))).T
x3_tr = np.column_stack((c3_tr, np.ones(N_tr))).T

C = 3
DIM = 4

X_tr = np.concatenate((x1_tr, x2_tr, x3_tr), axis=1) # X[i] = (DIM + 1) x 1 => X = (DIM + 1) x N
T_tr = np.zeros((C, np.shape(X_tr)[1])); T_tr[0, :N_tr] = 1; T_tr[1, N_tr:2*N_tr] = 1; T_tr[2, 2*N_tr:] = 1 # Same shape as G naturally, C x N

def MSE(G, T):
    MSE = 0
    for col in range(np.shape(G)[1]):
        MSE += (G[:,col] - T[:,col]).T @ (G[:,col] - T[:,col])
    return 1/2 * MSE

def grad_MSE(G, T, X):
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
        print(rel_err*100, "%")
    return W


#Model-testing

def get_conf_and_err(X, W): # Gets confusion matrix and probability of error from given X. The assumed correct result is given by the structure of X, as made earlier.
    x1_te, x2_te, x3_te = np.split(X, 3, axis=1)
    conf_matr = np.zeros((3,3))
    tot_errs = 0
    for i, xi_te in enumerate([x1_te, x2_te, x3_te]):
        Z_i = W @ xi_te
        G_i = 1 / (1 + np.exp(-Z_i))
        C_i = np.argmax(G_i, axis=0) # Holds the disciminatory vote for our given test set.
        for j in range(C):
            conf_matr[i][j] = np.sum(C_i == j) # Update the confusion matrix...
            if (i != j): tot_errs += conf_matr[i][j]
    P_e = tot_errs / (np.shape(X)[1])
    return conf_matr, P_e

#Test the models
#W = train_model(X_tr, T_tr)

N_te = len(c1_te)
x1_te = np.column_stack((c1_te, np.ones(N_te))).T # X = (DIM + 1) x N
x2_te = np.column_stack((c2_te, np.ones(N_te))).T
x3_te = np.column_stack((c3_te, np.ones(N_te))).T
X_te = np.concatenate((x1_te, x2_te, x3_te), axis=1)

# conf_te, p_err_te = get_conf_and_err(X_te, W)
# conf_tr, p_err_tr = get_conf_and_err(X_tr, W)
# print(conf_te, p_err_te * 100, "%")
# print(conf_tr, p_err_tr * 100, "%")

DEL_TRAIT = 0; SEL_TRAIT = 3
X_tr_red = X_tr[[SEL_TRAIT, 4], :]
print(np.shape(X_tr_red))
#X_tr_red = np.delete(X_tr, DEL_TRAIT, axis=0)
W_red = train_model(X_tr_red, T_tr)
X_te_red = X_te[[SEL_TRAIT,4], :]
#X_te_red = np.delete(X_te, DEL_TRAIT, axis=0)


conf_te_red, p_err_te_red = get_conf_and_err(X_te_red, W_red)
conf_tr_red, p_err_tr_red = get_conf_and_err(X_tr_red, W_red)
print(conf_te_red, p_err_te_red * 100, "%")
print(conf_tr_red, p_err_tr_red * 100, "%")

