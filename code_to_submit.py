#Author: Vyas Bhagwat
#Date: 12/16/2016
#This code predicts the final marks of a student based on various factors such as previous marks, age, sex, marital status of parents, alcohol consumption, etc.
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm, cross_validation, metrics, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import csv
import matplotlib.pyplot as plt 

def reduce_dimension(X, percent_covariance_to_account_for, percent_dimension_reduction): 
    """
    PRECONDITION: X is a 2D numpy array that has beeen normalized between
        -1 and 1. Use the scikit normalize function
                 
                 header is a list of column names of X. 
                 e.g., ["col 1 name", "col 2 name", ..., "col n name"]
                 
                 percent_covariance_to_account_for is the percentage of 
                 the covariance of the data that you wish to account for
                 in the dimension reduction. 
                 
                 percent_dimension_reduction  is the percentage of features
                 listed in a PC that you wish to keep and use when reducing 
                 the dimension.
                 
    POSTCONDITION: A tuple of the reduced data set (X_new) and the corresponding
                 header (header_new) is returned. That is (X_new, header_new) 
    
    SIDE EFFECTS:  None
    
    ERRORS:      None as of VERSION 1 of this code..
    """
    
    # Check if we should use PCA or SVD
    # PCA means Principal Component Analysis, 
    # SVD means Singular Value Decomposition
    X_shape = X.shape # Get dimension of data
    if X_shape[0] == X_shape[1]:  # If number samples equals dimension use PCA else use SVD 
        print
        print "The dimension of the data set is square, and PCA can be used."
        print "However, PCA is not implemented yet, so SVD will be used."
        print
             
    print "====================== Performing SVD ========================="
    num_dimensions = int(percent_dimension_reduction * float(X.shape[1]))
    print ">>>>> num_dimensions=", num_dimensions
    
    # End of section to write out original header & data
    
    print "Starting to fit SVD"
    svd = TruncatedSVD(n_components = num_dimensions) 
    svd.fit(X)
    print
   
    index = 0;
    temp = 0;
    while(temp + svd.explained_variance_ratio_[index] < percent_covariance_to_account_for):
        temp = temp + svd.explained_variance_ratio_[index]
        index += 1
        
    index -= 1
        
    print "percent_covariance_to_account_for: "
    print percent_covariance_to_account_for
    
    print "Number of components: "
    print index
    
    feature_avg = abs(svd.components_[:,0:index]).mean(axis=1)
        
    sorted_features = sorted((range(len(feature_avg))), key=lambda x: feature_avg[x])
        
    final_features = sorted(sorted_features[:int(len(sorted_features)*percent_dimension_reduction)])
        
    return final_features
    


#Read in the datasets
d1 = pd.read_csv("student-por.csv",sep = ';')   #This is the dataset for portugese
d2 = pd.read_csv("student-mat.csv",sep = ';')   #This is the dataset for math


#Make sure we have read the dataframes
print d1.head()
print d2.head()

#Check for missing values
print "Missing values in Math df:", d1.isnull().sum().sum()
print "Missing values in Portugese df:", d2.isnull().sum().sum()


#Find common students in each class

print "Finding common students from both classes:"
d3=pd.merge(d1,d2,on = ("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print d3.shape[0] # 382 students

#concatenate the two frames into one
frames = [d1,d2]
df = pd.concat(frames)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
X = df.drop(['G3'], axis = 1)
X = preprocessing.scale(X)   #  scale data between [-1, 1]
#print X

Y = df.G3
Y = preprocessing.scale(Y)
#print Y


#SVD
percent_covariance_to_account_for = 0.7
percent_dimension_reduction = 0.7

#final_features = reduce_dimension(X, percent_covariance_to_account_for, percent_dimension_reduction)
#print final_features
#X = X[:,final_features]
clf = svm.SVR(C=1.0, epsilon=0.2,kernel = 'rbf')
clf.fit(X,Y)
print "R^2 score: ", svm.SVR.score(clf,X,Y,sample_weight=None)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) 
clf = svm.SVR(C=1.0, epsilon=0.2,kernel = 'rbf')
clf.fit(X_train,y_train) 
y_train_pred = clf.predict(X_train) 
y_test_pred = clf.predict(X_test) 


#Visualize the prediction
plt.clf()
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data') 
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data') 
plt.xlabel('Predicted values') 
plt.ylabel('Residuals') 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') 
plt.xlim([-10, 50]) 
plt.show()

print
print 
print "=============  MSE  ====================="
# slide 45
from sklearn.metrics import mean_squared_error 
print('MSE train: %.3f, test: %.3f' % ( 
mean_squared_error(y_train, y_train_pred), 
mean_squared_error(y_test, y_test_pred)))

print
print

# slide 47
print "================  R^2 ==================="
from sklearn.metrics import r2_score 
print  'R^2 train: %.3f, test: %.3f' %  (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) 
print


