from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

data_dir = dirname(__file__)
mat_fname = pjoin(data_dir, 'data_all.mat')

mat_contents = sio.loadmat(mat_fname, spmatrix=False, mat_dtype=True)


# Task 2.1.2c
# Clustering training vectors from class w_i into M templates given by matrix C_i
M = 64
train = mat_contents['trainv']
train_labels = mat_contents['trainlab'].flatten()
num_train = int(mat_contents['num_train'][0][0])

# Separated classes in the training set 
train_dictionary = dict()
for i, sample in enumerate(train):
    label = train_labels[i]
    if label not in train_dictionary.keys():
        train_dictionary[label] = []
    train_dictionary[label].append(sample)


# Task 2.1.2b
num_classes = 10

templates = []
template_labels = []

for i in range(num_classes):
    X = np.array(train_dictionary[i])
    kmeans = KMeans(n_clusters=M, random_state=42)
    kmeans.fit(X)
    templates.append(kmeans.cluster_centers_)
    template_labels.extend([i]*M)
    
templates = np.vstack(templates) # (10*M, 784)
template_labels = np.array(template_labels) # (10*M,)


test = mat_contents['testv']
test_labels = mat_contents['testlab'].flatten()
num_test = int(mat_contents['num_test'][0][0])

knn = KNeighborsClassifier(n_neighbors=7,  metric='euclidean')
knn.fit(templates, template_labels)

y_pred = knn.predict(test)
confusion = confusion_matrix(test_labels, y_pred, labels=range(10))

plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for k-Nearest-Neighbor Classifier (k=7)")
plt.show()

error_rate = np.mean(y_pred != test_labels)
print(f"Error Rate: {error_rate:.4f}")
