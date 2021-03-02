# K-means-implementation

1st assignment of EECS 738 machine learning - leaf class prediction and Breast cancer classification

Prerequisites Python 3.7.3, Jupyter Notebook, Pandas Library, Numpy Library, matplotlib, and searborn library

Introduction: 
a) Multivariate Data set:

In this project, we are going to use unsupervised learning using the data set of leaf classification and Breast cancer classification. Leaf data set holds the information about classification of type of leaf on the basis of several attributes.These attributes are calculated on the basis of leaf's texture and dimensions. Name of these attributes are 'Class (Species)','Specimen Number','Eccentricity','Aspect Ratio','Elongation', 'Solidity','Stochastic Convexity','Isoperimetric Factor','Maximal Indentation Depth','Lobedness','Average Intensity','Average Contrast','Smoothness','Third moment','Uniformity','Entropy'. 
Similarly, Breast cancer classification data attributes helps in classifying whether a person has breast cancer or not. The attributes for this data set are "Age (years), BMI (kg/m2), Glucose (mg/dL), Insulin (µU/mL), HOMA, Leptin (ng/mL), Adiponectin (µg/mL), Resistin (ng/mL), MCP-1(pg/dL)" 


b) Data preparation for model:

Using jupyter notebook, import pandas, Numpy and matplotlib and read csv file using pandas dataframe. We are going to implement the model with leaf classification data set and then we will test it on breast cancer data set. All the data attributes should be quantitative in nature before training the model, so check for catagorycal data. if catagorycal data is present then encode it accordingly using One Hot encoding or any other method. For this model, we will choose 2 from attributes leaf data set 'Solidity' and 'Uniformity' as random variables for better understanding of how the algorithm works. Calculate mean and covariance for these variables so that multivariate_normal distribution can be applied.  


c) Algorithm implemetation:
step1:firstly, we define a function 'distance' for calculating euclidean distance between point and a centroid.

step2: Secondly, Function 'algo_initialize(data,k)'is defined, Where 'data' is 2D array of shape(300,2) and K no. of clusters to be formed. Initiate an empty list named as 'centroids' and randomely assign a data point from the array as centroid for the entire data set. Append this data point in list named 'centroids'.

Step3: Here we calculate euclidean distance between each data point and all the previous centroids using for loops. All the distance measurement to be recorded in a list called 'dist' where the index of 'dist' list points to the data point in 'data' array.

Step4: Here,We choose the data point from 'data' array as our next centroid which has max. dist from previous centroid with the help of dist list. This data point will be the next centroid for the data set. This will help in assigning more no. of data points to centroid along with maintaining minimum distance towards data points.

Step5: Repeat step3 and step4 untill we reach the given no. of k values for the data set. For example, if k= 5 is defined then we will assign 5 centroids for the data set using all the steps mentioned above.

d)conclusion:
Using this method, we can make sure that centroids should be assigned in way where the algorithm takes less time to converge rather than assigning arbitarily to a single point. There is a possibility of a centroid being assigned to just one or two data points(outliers) if we use k-means directly. This algorithm works efficiently with the breast cancer data set as well. So, a single algorithm can be used for more than one data set
