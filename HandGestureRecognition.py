# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:15:30 2020

@author: Faroud
"""
# %% imports
#importing all the needed package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simple_parser.Parser import Parser
#from scipy.spatial import distance

# %% Loading Data through the parser 
#%%
HAND_POSES_DIRECTORY = '../PositionHandJoints-20201021/PositionHandJoints/'

parser = Parser(HAND_POSES_DIRECTORY)    
hands_data = parser.parse()


# %% Stats on data (1)

"""Number of observation by class and this by hand as a DataFrame"""
# for left hand
left_hand_class_prop = hands_data.left_hand_data["classe"].value_counts()
# for right hand
right_hand_class_prop = hands_data.right_hand_data["classe"].value_counts()
print(f"""Classes proportions by hand:\n
      left hand: \n{left_hand_class_prop}\nright hand:\n\n{right_hand_class_prop}""")

# 
joined_class_prop = pd.concat([left_hand_class_prop, right_hand_class_prop], axis=1)
joined_class_prop.columns = ['left_hand', 'right_hand']
plt.figure()
joined_class_prop.plot.bar()

#
plt.figure()
joined_class_prop.plot.pie(subplots=True, figsize=(25, 25))

"""A remark here is that there are some configurationn only giveen by the right hand data 
so to integrate all the 32 classes we well combine all the data. 
Bellow are the conserned configuration.
"""
# not common config
not_common_config = joined_class_prop.loc[joined_class_prop.isna().any(axis=1)].index
print(f"{len(not_common_config)} configurations only from right hand:\n{not_common_config}")

# %% Fuze all the data together
all_data = hands_data.fuze_all_hands_data()
print(hands_data.left_hand_data.tail())
print(hands_data.right_hand_data.tail())
print(all_data.tail())

# %% general stats
# proportion of each configuration
config_count_by_class = all_data["classe"].value_counts()
print(f"""Classes proportions(total):\n{config_count_by_class}""")

plt.figure()
config_count_by_class.plot.bar() 

"""A quick remark is that Pi configuration represent a big part of the data 6432 out of 29415 frames. 
S blindly generating train data may induce wrong data proportion for training
"""

#%% 
# extracting labels names for later use.
# this is an array like: ['2', ...,'Xferme']
target_names_ = hands_data.classes_dict.keys()


# %% Feature extraction

#the Euclidean distance between two points
def euclidian_distance(point1, point2):
    #return np.sqrt(np.sum((point1 - point2) ** 2))
    #return distance.euclidean(point1, point2)
    return np.linalg.norm(point1 - point2) 


all_X = all_data.values
m,n = all_X.shape

def build_labels_y():
    classes_as_num = np.array(
        list(map(lambda x: hands_data.classes_dict[x], all_X[:, -1])),
        dtype=np.int32
    )
    return classes_as_num

"""Fingertips distance feature extraction"""
# this function is used to build the dataset we want to use
# NB: this is not a adaptable function: We asume that we want all the data 
# we could've set parameter to use only left or right hand data
def build_data_set_based_on_fingertips_distance():

    def build_X_based_on_fingertips_distance():
        fingertips_based_X = []
        
        for frame_index in range(m):
            new_observation = []
            for finger_tips_index in range(15, n, 15):
                new_observation = np.append(
                    new_observation,
                    [euclidian_distance(
                        all_X[frame_index, finger_tips_index: finger_tips_index+3], 
                        all_X[frame_index, next_finger_tips_index: next_finger_tips_index+3]
                        ) for next_finger_tips_index in range(finger_tips_index+15, n, 15)
                    ]
                )
            fingertips_based_X.append(new_observation)
        return np.asarray(fingertips_based_X)
    

    
    X = build_X_based_on_fingertips_distance();
    y = build_labels_y()
    return X, y

# %%
# using build_X_based_on_fingertips_distance
X1, y1 = build_data_set_based_on_fingertips_distance()
print(X1.shape, y1.shape)

# %% test
# quick test check: to make sure the logic 
# inside the build_X_based_on_fingertips_distance function is 
#correct 
# pinky_4_0_cords = all_data.loc[0, 'Pinky4_x':'Pinky4_z'].values
# ring_4_0_cords = all_data.loc[0, 'Ring4_x':'Ring4_z'].values

# distance1 = euclidian_distance(pinky_4_0_cords, ring_4_0_cords)
# print(distance1 == X1[0, 0])

#%% CLASSIFICATION

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

#from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

import Utils

class HandClassifier():
    def __init__(self, X_, y_):
        self.X = X_
        self.y = y_
        self.y_true = None
        self.Y_pred = None
    
    
    def _standardize(self):
        scaler = preprocessing.StandardScaler().fit(self.X)
        return scaler.transform(self.X)
    
    def _normalize(self, X_):
       #scaling features to a range
       min_max_scaler = preprocessing.MinMaxScaler()
       return min_max_scaler.fit_transform(X_)
   
    def _split_data(self, X_, y_):
        # 1. add X1 and y1 back together (y1 as column)
        X = np.column_stack((X_, y_))
        #
        X_train = np.empty((0, X.shape[-1]-1)); y_train = []
        X_test = np.empty((0, X.shape[-1]-1)); y_test = []
        for classe in np.unique(X[:, -1]):
            a_X_part = X[X[:,-1] == classe]
            a_X_train, a_X_test, a_y_train, a_y_test = train_test_split(a_X_part[:, :-1], a_X_part[:,-1], 
                                                                        test_size=0.20, 
                                                                        random_state=42)
            X_train = np.append(X_train, a_X_train, axis=0); y_train = np.append(y_train, a_y_train)
            X_test = np.append(X_test, a_X_test, axis=0); y_test = np.append(y_test, a_y_test)
        
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
      

    def plot_confusion_matrix(self, y_true, y_predict):
        pass
        
    def classify(self):
        X = self._normalize(self._standardize())
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.20, random_state=42)
        #X_train, X_test, y_train, y_test = self._split_data(X, self.y)
        
        #set y_true for later use
        self.y_true = y_test
        self.Y_pred = np.empty((0, y_test.shape[-1]), dtype=np.int32)
        
        def apply_logistic_regression():
            # Apply Logistic Regression
            lr = LogisticRegression(random_state=0, solver='saga', max_iter=200)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            # lr_scores = [precision_score(y_test, y_pred, average='weighted'),
            #              recall_score(y_test, y_pred, average='weighted'),
            #              f1_score(y_test, y_pred, average='weighted')]
            return y_pred
            # lr_report = {'conf_matrix': confusion_matrix(y_test, y_pred),
            #              'classif_report': classification_report(y_test, y_pred, target_names=target_names_)}
            
            # return lr_report
        
        def apply_knn():
            neigh = KNeighborsClassifier(n_neighbors=6)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)
            # neigh_scores = [precision_score(y_test, y_pred, average='weighted'),
            #                 recall_score(y_test, y_pred, average='weighted'),
            #                 f1_score(y_test, y_pred, average='weighted')]
            return y_pred
            
        def apply_bayes():
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            
            return gnb.predict(X_test)
            
        def apply_svm():
            svm_clf = svm.SVC(kernel='poly').fit(X_train, y_train)
            return svm_clf.predict(X_test)
        #
        self.Y_pred = np.append(
            self.Y_pred,
            [apply_logistic_regression(), 
             apply_knn(), 
             apply_bayes(), 
             apply_svm()],
            axis=0
        )
        

#%% CLASSIFICATIONS using X1, y1

handClassifier = HandClassifier(X1, y1)
handClassifier.classify()
# the above function apply the make the classification on the data 
# with 4 different classifier in he following order
# LogisticRegression, Knn, GNB and SVM

# %% Performance analysis

# model 2 performance

print('\nKNN performance evaluation:\n')
y_true = handClassifier.y_true
y_pred = handClassifier.Y_pred[1] # try with 0 or 1or 2 or 3 respectivly for 
# lr, kn, nb and svm

print(classification_report(y_true, y_pred, target_names=target_names_))
conf_matrix = confusion_matrix(y_true, y_pred)
Utils.plot_confusion_matrix(cm=conf_matrix, target_names=target_names_, normalize=True)


