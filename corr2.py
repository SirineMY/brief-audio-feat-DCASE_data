# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import numpy as np
import scipy.io as sio
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from features_functions import compute_features

from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import torch
import torchaudio

# Set the paths to the files 
data_path = "Data/"

# Names of the classes
classes_paths = ["Cars/", "Trucks/"]
classes_names = ["car", "truck"]
cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
nbr_of_sigs = 20 # Nbr of sigs in each class
seq_length = 0.2 # Nbr of second of signal for one sequence
nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

# Go to search for the files
learning_labels = []
for i in range(2*nbr_of_sigs):
    if i < nbr_of_sigs:
        name = f"{classes_names[0]}{cars_list[i]}.wav"
        class_path = classes_paths[0]
    else:
        name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
        class_path = classes_paths[1]

    # Read the data and scale them between -1 and 1
    fs, data = sio.wavfile.read(data_path + class_path + name)
    data = data.astype(float)
    data = data/32768

    # Cut the data into sequences (we take off the last bits)
    data_length = data.shape[0]
    nbr_blocks = int((data_length/fs)/seq_length)
    seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

    for k_seq, seq in enumerate(seqs):
        # Compute the signal in three domains
        sig_sq = seq**2
        sig_t = seq / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))

        # Compute the features and store them
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
        features_vector = np.array(features_list)[np.newaxis,:]

        if k_seq == 0 and i == 0:
            learning_features = features_vector
            learning_labels.append(classes_names[0])
        elif i < nbr_of_sigs:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[0])
        else:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[1])

print(learning_features.shape)
print(len(learning_labels))

# Separate data in train and test
X_train, X_test, y_train, y_test = train_test_split(learning_features, learning_labels, test_size=0.1, random_state=42)

# Standardize the labels
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)
testLabelsStd = labelEncoder.transform(y_test)



# Learn the model
#model= LogisticRegression(solver="lbfgs")
#model= svm.SVC(C=300, kernel='rbf', class_weight=None, probability=False)
#model= svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
#model= svm.SVC(C=10, kernel='sigmoid', class_weight=None, probability=False)
#model= MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=500, activation= 'relu', solver= 'adam')
#model= MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=500, activation= 'tanh', solver= 'adam')
#model= KNeighborsClassifier(n_neighbors = 5)
#model= RandomForestClassifier(random_state = 150,  max_features = 'auto',criterion= "entropy", bootstrap = True)
#model= DecisionTreeClassifier()




scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
learningFeatures_scaled = scaler.transform(X_train)

#model.fit(learningFeatures_scaled, learningLabelsStd)

# Test the model
testFeatures_scaled = scaler.transform(X_test)

#print the score of the model 

#print(name , "Score :" , model.score(testFeatures_scaled, testLabelsStd))
    
# # Matrix confusion
# plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
# plt.show()


# max_iters = [100, 200, 300, 400, 500, 600]
# activations = ['identity', 'logistic', 'tanh', 'relu']

# for activation in activations:
#     test_score = []
#     for max_iter in max_iters:
#         model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=max_iter, activation=activation, solver="adam")
#         model.fit(learningFeatures_scaled, learningLabelsStd)
#         test_score.append(model.score(testFeatures_scaled, testLabelsStd))
#     plt.plot(max_iters, test_score, label=activation)

# plt.xlabel('Max Iterations')
# plt.ylabel('Test Score')
# plt.legend()
# plt.show()


# Cc = [10, 25, 50, 100, 150, 200, 300]
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# for kernel in kernels:
#     test_score = []
#     for C in Cc:
#         model= svm.SVC(C=C, kernel=kernel, class_weight=None, probability=False)
#         model.fit(learningFeatures_scaled, learningLabelsStd)
#         test_score.append(model.score(testFeatures_scaled, testLabelsStd))
#     plt.plot(Cc, test_score, label=kernel)

# plt.xlabel('Max Iterations')
# plt.ylabel('Test Score')
# plt.legend()
# plt.show()


solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
test_score = []
for solver in solvers:
    model= LogisticRegression(solver=solver)
    model.fit(learningFeatures_scaled, learningLabelsStd)
    test_score.append(model.score(testFeatures_scaled, testLabelsStd))
    plt.plot(test_score, label=solver)

plt.xlabel('Max Iterations')
plt.ylabel('Test Score')
plt.legend()
plt.show()
