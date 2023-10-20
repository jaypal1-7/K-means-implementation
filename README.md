# Machine Learning Assignment 1: Leaf Class Prediction and Breast Cancer Classification
## Prerequisites
### Python 3.7.3
### Jupyter Notebook
### Pandas Library
### Numpy Library
### Matplotlib
### Seaborn Library

## Introduction
This project focuses on the application of unsupervised learning on two distinct datasets: Leaf Classification and Breast Cancer Classification.

## a) Multivariate Datasets
Leaf Classification
The Leaf dataset contains information about the classification of leaf types based on various attributes derived from the texture and dimensions of the leaves. Some of the attributes include 'Class (Species),' 'Specimen Number,' 'Eccentricity,' 'Aspect Ratio,' 'Elongation,' 'Solidity,' 'Stochastic Convexity,' 'Isoperimetric Factor,' 'Maximal Indentation Depth,' 'Lobedness,' 'Average Intensity,' 'Average Contrast,' 'Smoothness,' 'Third moment,' 'Uniformity,' and 'Entropy.'

Breast Cancer Classification
The Breast Cancer dataset helps classify whether a person has breast cancer or not. Its attributes include "Age (years), BMI (kg/m2), Glucose (mg/dL), Insulin (µU/mL), HOMA, Leptin (ng/mL), Adiponectin (µg/mL), Resistin (ng/mL), MCP-1(pg/dL)."

## b) Data Preparation for the Model
Using Jupyter Notebook, we import the necessary libraries, including Pandas, Numpy, and Matplotlib, and read the CSV file into a Pandas DataFrame. We will implement the model with the Leaf Classification dataset and later test it on the Breast Cancer dataset. All data attributes should be quantitative in nature before training the model. In the case of categorical data, it needs to be encoded, e.g., using One-Hot Encoding.

For this model, we will select 'Solidity' and 'Uniformity' as the two random variables from the Leaf dataset for better understanding of how the algorithm works. We will calculate the mean and covariance for these variables, allowing us to apply a multivariate normal distribution.

## c) Algorithm Implementation
Step 1: Euclidean Distance
We define a function 'distance' to calculate the Euclidean distance between a point and a centroid.

Step 2: Algorithm Initialization
In the function 'algo_initialize(data, k)', we initialize the algorithm. Here, 'data' is a 2D array of shape (300, 2), and 'k' represents the number of clusters to be formed. We initiate an empty list called 'centroids' and randomly select a data point as the initial centroid for the entire dataset. This data point is appended to the 'centroids' list.

Step 3: Calculate Distances
We calculate the Euclidean distance between each data point and all the previous centroids using for loops. The distances are recorded in a list called 'dist,' with the index of the 'dist' list corresponding to the data point in the 'data' array.

Step 4: Next Centroid
We select the data point from the 'data' array as the next centroid, which has the maximum distance from the previous centroid, using the 'dist' list. This data point becomes the next centroid for the dataset. This process helps assign more data points to centroids while maintaining the minimum distance to data points.

![image](https://user-images.githubusercontent.com/52853399/109596482-f9a99880-7adb-11eb-8eef-fed7fec5cc08.gif)

Step 5: Repeat
We repeat steps 3 and 4 until we reach the specified number of 'k' values for the dataset. For example, if 'k' is defined as 5, we will assign 5 centroids to the dataset using the above steps.

![image](https://user-images.githubusercontent.com/52853399/109596571-2493ec80-7adc-11eb-861a-6debf243868f.gif)
![image](https://user-images.githubusercontent.com/52853399/109596580-2958a080-7adc-11eb-8bfb-d5b1a150e980.gif)
![image](https://user-images.githubusercontent.com/52853399/109596588-2d84be00-7adc-11eb-8c15-cbbd43ec5447.gif)
![image](https://user-images.githubusercontent.com/52853399/109596608-32e20880-7adc-11eb-94ae-52c8ef7690e9.gif)
![image](https://user-images.githubusercontent.com/52853399/109596617-383f5300-7adc-11eb-8ae1-1e0c111c3794.gif)


## d) Conclusion
This algorithm is an efficient way to assign centroids and helps reduce convergence time. It prevents the assignment of centroids to just one or two data points (outliers), which can happen if we use k-means directly. Furthermore, this algorithm works effectively with the Breast Cancer dataset, demonstrating its versatility for multiple datasets. 

![image](https://user-images.githubusercontent.com/52853399/109596642-44c3ab80-7adc-11eb-8f3c-0f7188d52f93.gif)
![image](https://user-images.githubusercontent.com/52853399/109596662-4ee5aa00-7adc-11eb-994d-e57a58495998.gif)
![image](https://user-images.githubusercontent.com/52853399/109596681-56a54e80-7adc-11eb-8dba-8594e840843c.gif)
![image](https://user-images.githubusercontent.com/52853399/109596695-5c9b2f80-7adc-11eb-9bac-9417c39b92c7.gif)
![image](https://user-images.githubusercontent.com/52853399/109596704-6329a700-7adc-11eb-9f0e-565c85a2afe0.gif)
![image](https://user-images.githubusercontent.com/52853399/109596715-6886f180-7adc-11eb-9a43-a8ac2427e493.gif)






