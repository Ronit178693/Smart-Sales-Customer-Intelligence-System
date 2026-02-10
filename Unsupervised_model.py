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
