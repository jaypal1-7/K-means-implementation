//
//  python.cpp
//  
//
//  Created by jaypal singh on 3/1/21.
//

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('leaf.csv')
df.columns = ['Class (Species)','Specimen Number','Eccentricity','Aspect Ratio','Elongation', 'Solidity','Stochastic Convexity','Isoperimetric Factor','Maximal Indentation Depth','Lobedness','Average Intensity','Average Contrast','Smoothness','Third moment','Uniformity','Entropy']
df.head()
#We will choose 2 variables 'Solidity' and 'Uniformity'
# calculate mean and covarience for the two variables
# so that multivariate_normal distibution can be applied

mean = np.array(df.loc[:,'Solidity':'Uniformity'].mean())
covariance = np.array(df.loc[:,'Solidity':'Uniformity'].cov())
data = np.random.multivariate_normal(mean, covariance, 300)

           
# function to compute euclidean distance
def distance(x1, x2):
    return np.sum((x1 - x2)**2)
   
#Here we initialize the algorithm
def algo_initial(data, k):
    ''' inputs:
        data - numpy array of data points having shape (300, 2)
        k - number of clusters to be formed
    '''
## initializing the centroids list by
## randomly adding a data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
    plot(data, np.array(centroids))
    
## To compute remaining k-1 centroids we initilize a list 'dist'
##to keep the track of max.distance of a point from previous centroids using
##the distance function. This list will give a point from data array
## which have max. distance from centroid as compared to other data points and that will be our next
## centroid for the cluster
    for c_id in range(k - 1):
          
    ## initialize a list to store distances of data
    ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
              
    ## compute distance of 'point' from each of the previously
    ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
              
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        plot(data, np.array(centroids))
    return centroids
   
# call the initialize function to get the centroids
centroids = initialize(data, k = 6)

#plot the data and cetroids label with different colors to
# distinguish the prev, next centroids
def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker = '.',
                color = 'orange', label = 'data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color = 'black', label = 'previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color = 'green', label = 'next centroid')
    plt.title('Select % d th centroid'%(centroids.shape[0]))
      
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0.5, 1.5)
    plt.show()
