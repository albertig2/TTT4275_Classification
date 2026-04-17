import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join


def generate_histograms_for_fetures(class1_features, class2_features, class3_features, dimention):
    for i in range(dimention):
        plt.hist(class1_features[:, i], alpha=0.5, label="Class 1")
        plt.hist(class2_features[:, i], alpha = 0.5, label="Class 2")
        plt.hist(class3_features[:, i], alpha = 0.5, label="Class 3")
        plt.title("Feature " + str(i))
        plt.legend()
        plt.savefig(join(dirname(dirname(__file__)), 'results/Feature_' + str(i) + ".png"))
        plt.cla()
        
        
