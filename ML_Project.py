import pandas as pd 
import numpy as np
import argparse
import sys


def main(training_file):
	dataset = np.loadtxt(training_file, delimiter = " ")

	X_train = dataset[:,:-1]
	#X = dataset[:][:-1]
	Y_train = dataset[:,-1]

	RF_acc = rand_forest_classifier(X_train, Y_train, X_train, Y_train)

	print('accuracy of Random Forest: {:.2%}'.format(RF_acc))

def rand_forest_classifier(X_train, Y_train, X_test, Y_test):
	'''
	train a random forest classifier and test its accuracy on 
	some given test data
	'''
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators=100)
	classifier.fit(X_train, Y_train)

	RF_pred = classifier.predict(X_test)
	RF_acc = np.mean(RF_pred == Y_test)
	
	return RF_acc

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('training_file')

	parsed = parser.parse_args(sys.argv[1:])
	main(parsed.training_file)