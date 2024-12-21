import os
import time
import warnings

import numpy as np

from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


###############################################################################
# read data from file

def getDataPoints(path):
    dataPointsList = []
    for dataPointName in os.listdir(path):
        if dataPointName[0] == ".":
            continue
        with open(os.path.join(path, dataPointName), encoding="utf-8") as fileIn:
            dp = fileIn.read()
        dataPointsList.append(dp)
    return dataPointsList


def getDataPointsInfo(projectBasePath, projectName):
    # get list of tokenized test methods
    projectPath = os.path.join(projectBasePath, projectName)
    flakyPath = os.path.join(projectPath, "flakyMethods")
    nonFlakyPath = os.path.join(projectPath, "nonFlakyMethods")
    return getDataPoints(flakyPath), getDataPoints(nonFlakyPath)


###############################################################################
# compute effectiveness metrics

def computeResults(testLabels, predictLabels):
    warnings.filterwarnings("error")  # to catch warnings, e.g., "prec set to 0.0"
    try:
        precision = precision_score(testLabels, predictLabels)
    except:
        precision = "-"
    try:
        recall = recall_score(testLabels, predictLabels)
    except:
        recall = "-"
    warnings.resetwarnings()  # warnings are no more errors
    return precision, recall


###############################################################################
# FLAST

def flastVectorization(dataPoints, dim=0, eps=0.3):
    countVec = CountVectorizer()
    Z_full = countVec.fit_transform(dataPoints)
    if eps == 0:
        Z = Z_full
    else:
        if dim <= 0:
            dim = johnson_lindenstrauss_min_dim(Z_full.shape[0], eps=eps)
        srp = SparseRandomProjection(n_components=dim)
        Z = srp.fit_transform(Z_full)
        # print("this is Z",Z)
    return Z


from sklearn.ensemble import RandomForestClassifier
import time

def flastRandomForest(trainData, trainLabels, testData):
    t0 = time.perf_counter()

    # Train the model
    rf = RandomForestClassifier()
    rf.fit(trainData, trainLabels)

    t1 = time.perf_counter()
    trainTime = t1 - t0

    # Test the model
    t0 = time.perf_counter()
    predictLabels = rf.predict(testData)

    t1 = time.perf_counter()
    testTime = t1 - t0

    return trainTime, testTime, predictLabels


from sklearn.naive_bayes import GaussianNB
import time

def flastNaiveBayes(trainData, trainLabels, testData):
    t0 = time.perf_counter()

    # Train the model
    gnb = GaussianNB()
    gnb.fit(trainData, trainLabels)
    
    t1 = time.perf_counter()
    trainTime = t1 - t0

    # Test the model
    t0 = time.perf_counter()
    predictLabels = gnb.predict(testData)

    t1 = time.perf_counter()
    testTime = t1 - t0

    return trainTime, testTime, predictLabels





def find_best_param(X, y):

    """
    X : Treatment Variables \n
    y : Response Variable  \n
    model : random forest classifer / regressor \n 

    Info : 
    
    Returns suitable values for : \n
                {n_estimators', \n
                'max_features', \n
                'max_depth', \n
                'min_samples_split', \n
                'min_samples_leaf', \n
                'bootstrap'}

    Uses the random grid to search for best hyperparameters.

    
    """
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # select samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)
    
    model = RandomForestClassifier()

    # Using 3 fold cross validation with 50 combinations
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42)
    rf_random.fit(X, y)
    
    return rf_random.best_params_



