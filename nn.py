import pandas as pd 
import numpy 
from sklearn.neural_network import MLPClassifier


dataset = numpy.loadtxt('train.nmv0.csv', delimiter = " ")


X = dataset[:,0:205]
Y = dataset[:,205]

model = MLPClassifier(hidden_layer_sizes=(205,180, 180), random_state = 2, learning_rate='invscaling')
model.fit(X, Y)

predictedClassArray = model.predict(X)

i = 0
total = 0
numTimesRight = 0


for instance in predictedClassArray:
	originalClass = Y[0]
	predictedClass = instance
	if(originalClass == predictedClass):
		numTimesRight += 1
	total += 1

accuracy = numTimesRight/total

print("original training set accuracy")

print(accuracy)

dataset = numpy.loadtxt('train.nmv1.csv', delimiter = " ")


X = dataset[:,0:205]
Y = dataset[:,205]

# model = MLPClassifier(hidden_layer_sizes=(205,30))
# model.fit(X, Y)

predictedClassArray = model.predict(X)

i = 0
total = 0
numTimesRight = 0


for instance in predictedClassArray:
	originalClass = Y[0]
	predictedClass = instance
	if(originalClass == predictedClass):
		numTimesRight += 1
	total += 1

accuracy = numTimesRight/total

print("another training set accuracy")

print(accuracy)



dataset = numpy.loadtxt('train.nmv2.csv', delimiter = " ")


X = dataset[:,0:205]
Y = dataset[:,205]

# model = MLPClassifier(hidden_layer_sizes=(205,30))
# model.fit(X, Y)

predictedClassArray = model.predict(X)

i = 0
total = 0
numTimesRight = 0


for instance in predictedClassArray:
	originalClass = Y[0]
	predictedClass = instance
	if(originalClass == predictedClass):
		numTimesRight += 1
	total += 1

accuracy = numTimesRight/total

print("another training set accuracy 2")

print(accuracy)

