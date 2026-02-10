from Data_Preprocessing import train_x, test_x, train_y, test_y
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# The optimal K will be either 4, 5

#Initializing the K-Means Algorithm
Kmeans = KMeans(
    n_clusters = 4,
    random_state = 42,
    init = 'k-means++', # Using k-means++ for better initialization
    n_init = 10 # Number of times the algorithm will run with different centroid seeds
)
#Model Training on the training dataset
Kmeans.fit(train_x)
#Predicting the clusters
Kmeans.predict(train_x)
Kmeans.predict(test_x)
#Getting the Centroids
centroid = Kmeans.cluster_centers_
print("Centroids: ", centroid)

