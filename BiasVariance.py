import numpy as np
import ReadData as a
import DecisionTree as dt
import random

"""
Bagging algorithm
"""
def Bagging(S, T, m):
    #random.seed(20)
    trees = []
    for t in range(T):
        # draw samples without replacement
        #samples = random.sample(list(S), m)
        #samples = np.array(samples)
        samples = DrawSamples(S, m)
        # learn a decision tree
        tree = dt.ID3(samples, a.attributes2, a.labels2, "Entropy", a.columns2, 16, 0)
        trees.append(tree)    
            
    return trees


"""
Draw m uniformly without replacement
"""
def DrawSamples(S, m):
    samples = []
    for i in range(m):
        index = random.randint(0, len(S)-1)
        samples.append(S[index,:])
        
    samples = np.array(samples)
    return samples


"""
Get single and bagged trees
"""
def SingleAndBagged100Tree(S):
    baggedPredict = []
    for i in range(100):
        #print(i)
        
        trees = Bagging(S, 200, 500)
        baggedPredict.append(trees)
            
    #print(len(baggedPredict))
    singletrees100 = [bag[0] for bag in baggedPredict]
        
    return singletrees100, baggedPredict


"""
Get ground truth label
"""
def Truth(row):
    if row[-1] == 'yes':
        return 1
    else:
        return 0
    
    
"""
Bias Variance For Single Tree
yes represents 1, no represents 0
"""
def BiasVarianceForSingleTree(row, trees, columns):        
    numPredict = []
    for tree in trees:
        if dt.predict(row, tree, columns) == 'yes':
            numPredict.append(1)
        else:
            numPredict.append(0)
 
    n = len(numPredict)
    m = sum(numPredict)/n
    bias = (m - Truth(row))**2
    squareSum = [(x-m)**2 for x in numPredict]
    variance = 1/(n-1)*sum(squareSum)
    return bias, variance


"""
Bias Variance For Bagged Tree
yes represents 1, no represents 0
"""
def BiasVarianceForBaggedTree(row, baggedTrees,columns):
    numPredict = []
    for bag in baggedTrees:
        yes = 0
        no = 0
        for tree in bag:
            if dt.predict(row, tree, columns) == 'yes':
                yes += 1
            else:
                no += 1
                
        if yes > no:
            numPredict.append(1)
        else:
            numPredict.append(0)
    
    n = len(numPredict)
    m = sum(numPredict)/n
    bias = (m - Truth(row))**2
    squareSum = [(x-m)**2 for x in numPredict]
    variance = 1/(n-1)*sum(squareSum)
    return bias, variance



# Read in data and calculate
bankTrain = a.bankTrain
bankTest = a.bankTest

# For single decision tree learner
singleTrees, baggedTrees = SingleAndBagged100Tree(bankTrain)

#print(singleTrees[0].name)
#[print(t.name) for t in baggedTrees[0]]

bias_list = []
variance_list = []

bias_list_bag = []
variance_list_bag = []
for row in bankTest:
    b,v = BiasVarianceForSingleTree(row, singleTrees, a.columns2)
    bias_list.append(b)
    variance_list.append(v)
    
    b2,v2 = BiasVarianceForBaggedTree(row, baggedTrees, a.columns2)
    bias_list_bag.append(b2)
    variance_list_bag.append(v2)
    
aveBias = sum(bias_list)/len(bankTest)
aveVariance = sum(variance_list)/len(bankTest)
generalSquareError = aveBias + aveVariance

aveBiasBag = sum(bias_list_bag)/len(bankTest)
aveVarianceBag = sum(variance_list_bag)/len(bankTest)
generalSquareErrorBag = aveBiasBag + aveVarianceBag

print("2c:")
print("SingleTree:")
print("Bias: ", aveBias, " Variance: ", aveVariance, "GeneralSquareError", generalSquareError)
print("BaggedTree:")
print("Bias: ", aveBiasBag, " Variance: ", aveVarianceBag, "GeneralSquareError", generalSquareErrorBag)


