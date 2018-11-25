import pandas as pd 
import numpy as np
import argparse
import sys
import os
from sklearn.model_selection import KFold


def main(path_to_files, type_of_files, classifier):
	#read in the training and test data
	#setting dtype to none captures the actual class of the data
	data = read_in_data(path_to_files, type_of_files)

	kf = KFold(n_splits=5)
	split_n = 1
	for train_index, test_index in kf.split(data):
		X_train, X_test = data.iloc[train_index].iloc[:, :-1], data.iloc[test_index].iloc[:, :-1]
		Y_train, Y_test = data.iloc[train_index].iloc[:, -1], data.iloc[test_index].iloc[:, -1]

		print('split number', split_n)
		if classifier == 'Random Forest':
			RF_acc = rand_forest_classifier(X_train, Y_train, X_test, Y_test)
			print_accuracy('Random Forrest', RF_acc)
		elif classifier == 'Bernoulli Naive Bayes':
			BNB_acc = bernoulli_naive_bayes(X_train, Y_train, X_test, Y_test)
			print_accuracy('Bernoulli Naive Bayes', BNB_acc)
		elif classifier == 'Multinomial Naive Bayes':
			MNB_acc = multinomial_naive_bayes(X_train, Y_train, X_test, Y_test)
			print_accuracy('Multinomial Naive Bayes', MNB_acc)
		elif classifier == 'Logistic Reggression':
			LR_acc = logistic_regression(X_train, Y_train, X_test, Y_test)
			print_accuracy('Logistic Regression', LR_acc)
		else:
			MLP_acc = mlp_classifier(X_train, Y_train, X_test, Y_test)
			print_accuracy('Neural Nets', MLP_acc)
		
		print()	#newline
		split_n += 1

def mlp_classifier(X_train, Y_train, X_test, Y_test):
	'''
	train a Neural Net classifier and test its accuracy on 
	some given test data
	'''
	from sklearn.neural_network import MLPClassifier
	classifier = MLPClassifier(hidden_layer_sizes=(205,180), random_state = 2, learning_rate='invscaling')
	classifier.fit(X_train, Y_train)

	MLP_pred = classifier.predict(X_test)
	MLP_acc = np.mean(MLP_pred == Y_test)
	
	return MLP_acc		

def rand_forest_classifier(X_train, Y_train, X_test, Y_test):
	'''
	train a random forest classifier and test its accuracy on 
	some given test data
	'''
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
	classifier.fit(X_train, Y_train)

	RF_pred = classifier.predict(X_test)
	RF_acc = np.mean(RF_pred == Y_test)
	
	return RF_acc

def bernoulli_naive_bayes(X_train, Y_train, X_test, Y_test):
	'''
	train a bernoulli naive bayes learner, and test its accuracy
	on some given test data
	'''
	from sklearn.naive_bayes import BernoulliNB
	naive_bayes = BernoulliNB()
	naive_bayes.fit(X_train, Y_train)

	BNB_pred = naive_bayes.predict(X_test)
	BNB_acc = np.mean(BNB_pred == Y_test)
	
	return BNB_acc

def multinomial_naive_bayes(X_train, Y_train, X_test, Y_test):
	'''
	train a multinomial naive bayes learner, and test its accuracy
	on some given test data
	'''
	from sklearn.naive_bayes import MultinomialNB
	naive_bayes = MultinomialNB()
	naive_bayes.fit(X_train, Y_train)

	MNB_pred = naive_bayes.predict(X_test)
	MNB_acc = np.mean(MNB_pred == Y_test)
	
	return MNB_acc

def logistic_regression(X_train, Y_train, X_test, Y_test):
	'''
	train a logistic Regression model, and test its accuracy
	on some given test data
	'''
	from sklearn.linear_model.logistic import LogisticRegression
	classifier = LogisticRegression(solver='liblinear')
	classifier.fit(X_train, Y_train)

	LR_pred = classifier.predict(X_test)
	LR_acc = np.mean(LR_pred == Y_test)
	
	return LR_acc

def print_accuracy(test_name, acc):
	print('accuracy of {}: {:.2%}'.format(test_name, acc))

def read_in_data(path_to_files, type_of_files):
	'''
	read in data and transform it into numpy arrays
	return it as a panda dataframe for ease of use
	'''
	data = []
	for file in os.listdir(path_to_files):
		if type_of_files in file:
			fwp = os.path.join(path_to_files, file)
			data.append(np.genfromtxt(fwp, delimiter = " ", dtype=None))

	np_data = data[0]
	for arr in data[1:]:
		np_data = np.append(np_data, arr)
	return pd.DataFrame(np_data)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path_to_files')
	parser.add_argument('--type_of_files', '-tof', choices=['nmv', 'mv'], default='nmv')
	parser.add_argument('--classifier', '-cfr', choices=['Random Forest',
										'Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Logistic Regression',
										'Neural Nets'], default='Random Forest')

	parsed = parser.parse_args(sys.argv[1:])
	main(parsed.path_to_files, parsed.type_of_files, parsed.classifier)