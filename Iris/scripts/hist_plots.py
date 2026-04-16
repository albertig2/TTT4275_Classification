import numpy as np
import matplotlib.pyplot as plt


c1 = np.loadtxt("Iris/data/class_1", delimiter=",")
c2 = np.loadtxt("Iris/data/class_2", delimiter=",")
c3 = np.loadtxt("Iris/data/class_3", delimiter=",")

print(np.shape(c1))
DIM = 4

for i in range(DIM):
    plt.hist(c1[:, i], alpha=0.5, label="Class 1")
    plt.hist(c2[:, i], alpha = 0.5, label="Class 2")
    plt.hist(c3[:, i], alpha = 0.5, label="Class 3")
    plt.title("Feature " + str(i))
    plt.legend()
    plt.savefig("../TTT4275_CLASSIFICATION/Iris/results/Feature " + str(i))
    plt.cla()

