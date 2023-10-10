#!/usr/bin/env python

#***************************************
#author: TINA
#date: 2014_10_13 CSLT

# update by Zheng Li at XMUSPEECH
# 2020 07 26

#***************************************

import os, sys, math
import scipy as sp
import numpy as np
from numpy import linalg
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import f1_score,accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
import time

if __name__ == "__main__":

	fin_train = sys.argv[1]
	fin_test = sys.argv[2]
	fout_ = sys.argv[3]

	maxit = int(sys.argv[4])	
	curve = sys.argv[5]
	Cvalue = float(sys.argv[6])

	sys.stderr.write('Paras: ' + fin_train + ' ' + fin_test + ' ' + fout_ + '\n')

	# data preparation for training set
	# time.sleep(3000)

	train_data = []
	train_lable = []

	fin = open(fin_train, 'r')
	sort_ = []
	for line in fin:
		sort_.append(line)
	fin.close()
	sort_.sort()
	fin = open(fin_train, 'w')
	fin.writelines(sort_)
	fin.close()

	fin = open(fin_train, 'r')

	for line in fin:
		line = line.strip()
		wordList = line.split()
		tempList = []
		tempList = np.array(wordList[1:], dtype=float)
		train_data.append(tempList)
		train_lable.append(np.array(wordList[0], dtype=int))

	sys.stderr.write('train_data: ' + str(len(train_data)) + ' ' + 'train_lable: ' + str(len(train_lable)) + '\n')

	fin.close()

	# model training
	#scaler = preprocessing.StandardScaler().fit(train_data)
	#scaler.transform(train_data)
	# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e1,1,1e-1,1e-2,1e-3, 1e-4],
    #                  'C': [1e-3,1e-2,1e-1,1, 10, 100, 1000]}]
	# clf = GridSearchCV(svm.SVC(probability = True), tuned_parameters, cv=5)
	# clf = svm.SVC(C=0.1, kernel = curve, probability = True, max_iter = -1, class_weight = None, gamma="auto")
	# print("=====C = {}, curve = {}, maxit = {}".format(Cvalue,curve,maxit))
	clf = svm.NuSVC(nu=0.01, kernel = curve,probability = True, max_iter = maxit)
	clf.fit(train_data, train_lable)
	sys.stderr.write('Training Done!\n')

	# data preparation for test target test

	test_data = []
	test_lable = []
	
	fin = open(fin_test, 'r')
	sort_ = []
	for line in fin:
		sort_.append(line)
	fin.close()
	sort_.sort()
	fin = open(fin_test, 'w')
	fin.writelines(sort_)
	fin.close()
	
	fin = open(fin_test, 'r')
	
	for line in fin:
		line = line.strip()
		wordList = line.split()
		tempList = []
		tempList = np.array(wordList[1:], dtype=float)
		test_data.append(tempList)
		test_lable.append(np.array(wordList[0], dtype=int))
	sys.stderr.write('test_data: ' + str(len(test_data)) + ' ' + 'test_lable: ' + str(len(test_lable)) + '\n')

	test_data = test_data / np.linalg.norm(test_data,axis=1,keepdims=True)
	fin.close()

	# scaler.transform(test_data)

	# predict
	fout = open(fout_, 'w')
	correct = 0
	incorrect = 0
	for i in range(len(test_data)):
		prob = clf.predict_proba(test_data[i].reshape(1,-1))
		# print(prob)
		# time.sleep(1000)
		fout.write(str(prob) + '\n')	
	
	fout.close()

	# predict_test_labels = clf.predict(test_data)

#	for i in range(len(test_data)):
#		pre = clf.predict(test_data[i])
#		if pre[0] == test_lable[i]:
#			correct += 1
#		else:
#			incorrect += 1
#	sys.stderr.write('trials: ' + str(len(test_data)) + ' ' + str(correct) + ' ' + str(incorrect))

	# print 'Test Done'	
	# print(clf.best_params_)
	# print("test accuracy score: {0:.2f}".format(accuracy_score(predict_test_labels,test_lable)))