# CS5350
This is a machine learning library developed by Ashley Lan for
CS5350/6350 in University of Utah

Include run.sh to start the program for each folder


#### For HW1-DecisoinTree:
Call ID3(S, Attribute, Label, splitName, columns, maxDepth, currDepth) to learn Decision Tree.\
S-whole data set\
Attribute-whole attributes, a dictonary including values for each attribute\
Label-a list contains all possible labels\
splitName-type of information gain we can have: "Entropy", "MajorityError", or "GiniIndex"\
columns-a list contains all columns header\
maxDepth-int of max depth for the tree\
currDepth-int of current depth for the tree

#### For HW2-Ensemble Learning and Linear Regression:
My program for running ensemble learning takes a longer time than expected.

Ensemble Learning- run each python file below to get part 1-5 answer\
AdaBoost.py
Bagging.py
BiasVariance.py
RandomForest.py
BiasVariance2e.py

Linear Regression-run GradientDescent.py to get answer

#### For HW3-Perceptron:
There is Standard Perceptron, Voted Perceptron and Average Perceptron to use.\
The parameters would be data, learning rate and number of epoch.\
StandardPerceptron(data, r, T) VotedPerceptron(data, r, T) AveragePerceptron(data, r, T)

#### For HW4-SVM:
primal.py is to run the SVM in primal form\
dual.py is to run the SVM in dual form\
The dual.py might take around 1 hour to run and print all the results
