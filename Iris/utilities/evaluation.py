import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join


def generate_histograms_for_fetures(class_features, dimention, classes):
    for i in range(dimention): #Plot histograms for each feature
        for j in range(classes): # and each class
            plt.hist(class_features[j, :, i], alpha=0.8, label="Class" + str(j))
        plt.title("Feature " + str(i))
        plt.legend()
        plt.savefig(join(dirname(dirname(__file__)), "results/seperability_histograms/Histogram_feature_" + str(i) + ".png" ))
        plt.cla()
        