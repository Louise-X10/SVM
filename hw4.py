#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:20:43 2022

@author: liuyilouise.xu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import decomposition

#create train and test data to use SVM light

features = pd.read_csv("features.txt", header=None) #normalized feature vector from hw2
features = features.values[:,:-1] #remove last column hash file name

train, test  = train_test_split(features, test_size = 0.2) 

# convert array of data into desired format for SVM
def convert_data (file, data):
    rows = data.shape[0]
    cols = data.shape[1]
    for row in range(rows):
        for col in range(cols):
            if col==0:
                file.write(str(data[row,col]))
                file.write(" ")
            else:
                if data[row,col] == 0:
                    continue
                file.write(str(col))
                file.write(":")
                file.write(str(data[row,col]))
                file.write(" ")
        file.write("\n")
    file.close()
    
file = open("train.dat", "w")
convert_data(file, train)

file = open("test.dat", "w")
convert_data(file, test)

# PCA
pca_model = decomposition.PCA(n_components=768)
pca_model.fit(features[:, 1:]) #strip away labels
cum_variance = pca_model.explained_variance_ratio_.cumsum()
print(cum_variance)

pca_data = pca_model.transform(features[:, 1:])
pca_data = np.append(features[:,0].reshape(features.shape[0],1), pca_data, axis=1)

'''
def keep_features(n, pca_data, cum_variance):
    reduce_pca_data = pca_data[:, 0:(n+1)] # keep n features
    train, test  = train_test_split(reduce_pca_data, test_size = 0.2) 
    file = open(f"pca_train_n{n}.dat", "w")
    convert_data(file, train)

    file = open(f"pca_test_n{n}.dat", "w")
    convert_data(file, test)
    
    print("Features kept: %d, retained data variance %f" %(n, cum_variance[n]))
'''

def reduce_to_threshold(threshold, pca_data, cum_variance):
    n = np.max(np.where(cum_variance <=threshold)) + 1
    reduce_pca_data = pca_data[:, 0:(n+1)] # keep n features
    train, test  = train_test_split(reduce_pca_data, test_size = 0.2) 
    file = open(f"pca_train_n{n}.dat", "w")
    convert_data(file, train)

    file = open(f"pca_test_n{n}.dat", "w")
    convert_data(file, test)
    
    print("Features kept: %d, retained data variance %f" %(n, cum_variance[n]))
    
for i in [0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    reduce_to_threshold(i, pca_data, cum_variance)