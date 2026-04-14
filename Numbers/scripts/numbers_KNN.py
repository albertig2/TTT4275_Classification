import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
import scipy.io as sio
import seaborn as sns
from sklearn import metrics, cluster, neighbors

data_all_mat = join(dirname(dirname(__file__)), 'data', 'data_all.mat')
data_all = sio.loadmat(data_all_mat, spmatrix=False, mat_dtype=True)

# Task 2.1 Nearest-Neighborhood Classifier
## Task 2.1.2 Nearest-Neighborhood Classifier with 10*64 templates
### Task 2.1.2c K-Nearest_Neighborhood 
M = 64
train = data_all['trainv']
train_labels = data_all['trainlab'].flatten()
num_train = int(data_all['num_train'][0][0])

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
    kmeans = cluster.KMeans(n_clusters=M, random_state=42)
    kmeans.fit(X)
    templates.append(kmeans.cluster_centers_)
    template_labels.extend([i]*M)
    
templates = np.vstack(templates) # (10*M, 784)
template_labels = np.array(template_labels) # (10*M,)


test = data_all['testv']
test_labels = data_all['testlab'].flatten()
num_test = int(data_all['num_test'][0][0])

knn = neighbors.KNeighborsClassifier(n_neighbors=7,  metric='euclidean')
knn.fit(templates, template_labels)

y_pred = knn.predict(test)
confusion = metrics.confusion_matrix(test_labels, y_pred, labels=range(10))

plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for k-Nearest-Neighbor Classifier (k=7)")
plt.show()

error_rate = np.mean(y_pred != test_labels)
print(f"Error Rate: {error_rate:.4f}")
