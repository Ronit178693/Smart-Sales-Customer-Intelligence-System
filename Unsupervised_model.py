from Data_Preprocessing import train_x, test_x, train_y, test_y
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Using the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(train_x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize = (10,7))
sns.lineplot(x = range(2, 11), y = wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# The optimal K will be either 3, 4

#Initializing the K-Means Algorithm
Kmeans = KMeans(
    n_clusters = 3,
    random_state = 42,
    init = 'k-means++', # Using k-means++ for better initialization
    n_init = 10 # Number of times the algorithm will run with different centroid seeds
)
#Model Training on the training dataset
Kmeans.fit(train_x)
#Predicting the clusters
label = Kmeans.predict(train_x)
print("Silhouette Score: ", silhouette_score(train_x, label))
#Getting the Centroids
centroid = Kmeans.cluster_centers_
print("Centroids: ", centroid)

#Visualizing the clusters
plt.figure(figsize = (10,7))
sns.scatterplot(x = train_x[:, 0], y = train_x[:, 1], hue = Kmeans.labels_)
plt.scatter(centroid[:, 0], centroid[:, 1], marker = 'x', s = 100, color = 'red')
plt.title('Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
