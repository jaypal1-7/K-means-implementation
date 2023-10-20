//
//  python.cpp
//  
//
//  Created by jaypal singh on 3/1/21.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset and rename columns
df = pd.read_csv('leaf.csv')
df.columns = ['Class (Species)', 'Specimen Number', 'Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity',
              'Stochastic Convexity', 'Isoperimetric Factor', 'Maximal Indentation Depth', 'Lobedness',
              'Average Intensity', 'Average Contrast', 'Smoothness', 'Third moment', 'Uniformity', 'Entropy']

# Display the first few rows of the dataset
print(df.head())

# Choose two variables for clustering: 'Solidity' and 'Uniformity'
data = df.loc[:, ['Solidity', 'Uniformity']].values

# Function to compute Euclidean distance between two points
def distance(x1, x2):
    return np.sum((x1 - x2) ** 2)

# Initialize the K-Means algorithm
def initialize_kmeans(data, k):
    # Initialize centroids list by randomly selecting a data point
    centroids = [data[np.random.randint(data.shape[0]), :]]
    
    # Loop to find the remaining centroids
    for c_id in range(k - 1):
        # Initialize a list to store distances of data points from the nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = np.inf  # Set to positive infinity
            
            # Compute distance of 'point' from each of the previously selected centroids
            # and store the minimum distance
            for j in range(len(centroids):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            
            dist.append(d)
          
        # Select the data point with the maximum distance as the next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    
    return centroids

# Call the initialization function to get the initial centroids
initial_centroids = initialize_kmeans(data, k=6)

# Plot the data and centroids to visualize the initialization
def plot_data_and_centroids(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.', color='orange', label='Data Points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1], color='black', label='Previously Selected Centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1], color='green', label='Next Centroid')
    plt.title('Select %d th centroid' % centroids.shape[0])
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0.5, 1.5)
    plt.show()

# Visualize the initialization
plot_data_and_centroids(data, initial_centroids)
