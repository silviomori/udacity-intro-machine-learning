#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)//10]
# labels_train = labels_train[:len(labels_train)//10]


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# for C in [1, 10, 100, 1000, 10000]:
for C in [10000]:
    print('=== C:', C)
    svc = SVC(kernel='rbf', C=C)
    t0 = time()
    svc.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3), "s")

    t0 = time()
    predictions = svc.predict(features_test)
    print("predicting time:", round(time()-t0, 3), "s")

    accuracy = accuracy_score(labels_test, predictions)
    print('Accuracy:', accuracy)
