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
### Task 2.1.2a Clustering
train = data_all['trainv']
train_labels = data_all['trainlab'].flatten()

#### Separating classes in the training set
train_dictionary = dict()
for i, sample in enumerate(train):
    label = train_labels[i]
    if label not in train_dictionary.keys():
        train_dictionary[label] = []
    train_dictionary[label].append(sample)

M = 64
num_classes = 10
template = []
template_labels = []

for i in range(num_classes):
    X = np.array(train_dictionary[i])
    kmeans = cluster.KMeans(n_clusters=M, random_state=42)
    kmeans.fit(X)
    template.append(kmeans.cluster_centers_)
    template_labels.extend([i]*M)
    
template = np.vstack(template)
template_labels = np.array(template_labels)


### Task 2.1.2b
test = data_all['testv']
test_labels = data_all['testlab'].flatten()
num_test = int(data_all['num_test'][0][0])

chunk_size = 1000
predicted_labels = np.zeros(num_test, dtype=int)
template_norms = np.sum(template**2, axis=1)

for i in range(0, num_test, chunk_size):
    test_chunk = test[i:i+chunk_size]
    test_norms = np.sum(test_chunk**2, axis=1, keepdims=True)
    distances = template_norms + test_norms - 2 * test_chunk @ template.T
    nearest_idx = np.argmin(distances, axis=1)
    predicted_labels[i:i+chunk_size] = template_labels[nearest_idx]

labels = [0,1,2,3,4,5,6,7,8,9]
confusion = metrics.confusion_matrix(test_labels, predicted_labels, labels=labels)
plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Nearest-Neighbor Classifier with 10*64 templates")
plt.show()

error_rate = np.mean(predicted_labels != test_labels)
print(f"Error Rate: {error_rate:.4f}")

### Task 2.1.2c K-NN

knn = neighbors.KNeighborsClassifier(n_neighbors=7,  metric='euclidean')
knn.fit(template, template_labels)

predicted_labels = knn.predict(test)
confusion = metrics.confusion_matrix(test_labels, predicted_labels, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for K-Nearest-Neighbor Classifier (K=7)")
plt.show()

error_rate = np.mean(predicted_labels != test_labels)
print(f"Error Rate: {error_rate:.4f}")

