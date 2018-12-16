import pandas as pd 
import numpy as np
import argparse
import sys
import os
from sklearn.model_selection import KFold


def main(path_to_training, path_to_evaluation, n_folds, type_of_files, classifier):
	#read in the training and test data
	#setting dtype to none captures the actual class of the data
	data = read_in_data(path_to_training, type_of_files)

	selection_model = select_features(data)

	#specify number of folds 
	kf = KFold(n_splits=n_folds)
	split_n = 1
	best_accuracy = 0
	best_classifier = None
	for train_index, test_index in kf.split(data):
		#seperate data into train/test
		X_train, X_test = data.iloc[train_index].iloc[:, :-1], data.iloc[test_index].iloc[:, :-1]
		Y_train, Y_test = data.iloc[train_index].iloc[:, -1], data.iloc[test_index].iloc[:, -1]

		#purge data based on the selection model calculated earlier 
		X_train = selection_model.transform(X_train)
		X_test = selection_model.transform(X_test)

		#test on the user specified ML algos
		print('split number', split_n)
		if 'Random_Forrest' in classifier :
			RF_acc, cf = rand_forest_classifier(X_train, Y_train, X_test, Y_test)
			print_accuracy('Random Forrest', RF_acc)
			if RF_acc > best_accuracy:
				best_accuracy = RF_acc
				best_classifier = cf
		if 'Bernoulli_Naive_Bayes' in classifier:
			BNB_acc, cf = bernoulli_naive_bayes(X_train, Y_train, X_test, Y_test)
			print_accuracy('Bernoulli_Naive_Bayes', BNB_acc)
			if BNB_acc > best_accuracy:
				best_accuracy = BNB_acc
				best_classifier = cf
		if 'Multinomial_Naive_Bayes' in classifier:
			MNB_acc, cf = multinomial_naive_bayes(X_train, Y_train, X_test, Y_test)
			print_accuracy('Multinomial_Naive_Bayes', MNB_acc)
			if MNB_acc > best_accuracy:
				best_accuracy = MNB_acc
				best_classifier = cf
		if 'Logistic_Regression' in classifier:
			LR_acc, cf = logistic_regression(X_train, Y_train, X_test, Y_test)
			print_accuracy('Logistic Regression', LR_acc)
			if LR_acc > best_accuracy:
				best_accuracy = LR_acc
				best_classifier = cf
		if 'Neural_Nets' in classifier:
			MLP_acc, cf = mlp_classifier(X_train, Y_train, X_test, Y_test)
			print_accuracy('Neural Nets', MLP_acc)
			if MLP_acc > best_accuracy:
				best_accuracy = MLP_acc
				best_classifier = cf
		
		print()	#newline
		split_n += 1

	#read in data to be evaluated and make it work with our models
	eval_data = read_in_data(path_to_evaluation, type_of_files)
	X_eval = eval_data.iloc[: , :-1]
	X_eval = selection_model.transform(X_eval)
	Y_eval = best_classifier.predict(X_eval)

	'''
	for testing
	#open the correct prediction file to see how well our model is doing
	with open('./PredictionCorrect.txt', 'r') as f:
		correct_eval = [int(i.strip()) for i in f.readlines() if i != '\n']
		acc_eval = np.mean(correct_eval == Y_eval)
		print_accuracy('eval file', acc_eval)
		'''

	with open('PredictionFile.txt', 'w') as f:
		for Y in Y_eval[:-1]:
			f.write('{}\n'.format(Y))
		f.write('{}'.format(Y_eval[-1]))

def mlp_classifier(X_train, Y_train, X_test, Y_test):
	'''
	train a Neural Net classifier and test its accuracy on 
	some given test data
	'''
	from sklearn.neural_network import MLPClassifier
	classifier = MLPClassifier(hidden_layer_sizes=(50, 45), random_state = 2, activation='logistic',
													 learning_rate='invscaling')
	classifier.fit(X_train, Y_train)

	MLP_pred = classifier.predict(X_test)
	MLP_acc = np.mean(MLP_pred == Y_test)
	
	return MLP_acc, classifier		

def rand_forest_classifier(X_train, Y_train, X_test, Y_test):
	'''
	train a random forest classifier and test its accuracy on 
	some given test data
	'''
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=80, max_features='log2')
	classifier.fit(X_train, Y_train)

	RF_pred = classifier.predict(X_test)
	RF_acc = np.mean(RF_pred == Y_test)
	
	return RF_acc, classifier

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
	
	return BNB_acc, naive_bayes

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
	
	return MNB_acc, naive_bayes

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
	
	return LR_acc, classifier

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
			data.append(np.genfromtxt(fwp, delimiter = " ", dtype=None, encoding=None, missing_values='?'))

	np_data = data[0]
	for arr in data[1:]:
		np_data = np.append(np_data, arr)
	return pd.DataFrame(np_data)

def select_features(data):
	'''
	select features, such that only the relevant features are left
	that is features which contribute the most to the classification 
	of some class
	'''
	#an attempt at feature selection, not sure if its the best
	#but it seems to work for now, does not improve accuracy much 
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.feature_selection import SelectFromModel

	X_train = data.iloc[:, :-1]
	Y_train = data.iloc[:, -1]

	clf = ExtraTreesClassifier(n_estimators=100, criterion='entropy')
	clf = clf.fit(X_train, Y_train)

	selection_model = SelectFromModel(clf, prefit=True)
	return selection_model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path_to_training', default='./TrainingFiles')
	parser.add_argument('path_to_evaluation', default='./EvaluationFiles')
	parser.add_argument('number_of_folds', default='5', type=int)
	parser.add_argument('--type_of_files', '-tof', choices=['nmv', 'ymv'], default='nmv')
	parser.add_argument('--classifier', '-cfr', choices=['Random_Forest',
										'Bernoulli_Naive_Bayes', 'Multinomial_Naive_Bayes', 'Logistic_Regression',
										'Neural_Nets'], default=['Random_Forrest'], action='append')

	parsed = parser.parse_args(sys.argv[1:])
	main(parsed.path_to_training, parsed.path_to_evaluation, parsed.number_of_folds, parsed.type_of_files, parsed.classifier)