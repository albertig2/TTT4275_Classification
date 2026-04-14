import numpy as np
from sklearn import cluster

def separate_classes(train, train_labels) -> dict:
    train_dictionary = dict()
    for i, sample in enumerate(train):
        label = train_labels[i]
        if label not in train_dictionary.keys():
            train_dictionary[label] = []
        train_dictionary[label].append(sample)
    return train_dictionary


def create_templates(train_dictionary, M=64, num_classes=10) -> tuple[np.ndarray, np.ndarray]:
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
    return template, template_labels