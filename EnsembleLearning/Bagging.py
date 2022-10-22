import numpy as np
import math
import random
import DecisionTree as dt
import matplotlib.pyplot as plt
import ReadData as a


"""
Bagging algorithm
"""
def Bagging(S, T, m):
    #random.seed(20)
    trees = []
    for t in range(T):
        # draw m uniformly with replacement
        samples = DrawSamples(S, m)
        # learn a decision tree
        tree = dt.ID3(samples, a.attributes2, a.labels2, "Entropy", a.columns2, 16, 0)
        trees.append(tree)    
            
    return trees


"""
Calculate errors based on learned trees
"""
def Error(S, trees):
    error = 0
    for row in S:
        yes = 0
        no = 0
        for tree in trees:
            if 'yes' == dt.predict(row, tree, a.columns2):
                yes += 1
            else:
                no += 1
        
        if yes > no:
            predic = 'yes'
            #predictions.append('yes')
        else:
            predic = 'no'
            #predictions.append('no')
    
        # find mismatch
        if row[-1] != predic:
            error += 1
    
    return error/len(S)

"""
Draw m uniformly with replacement
"""
def DrawSamples(S, m):
    samples = []
    for i in range(m):
        index = random.randint(0, len(S)-1)
        samples.append(S[index,:])
        
    samples = np.array(samples)
    return samples


# get training trees
bankTrain = a.bankTrain
bankTest = a.bankTest

train_error = []
test_error = []
for t in range(1, 51, 2):
    # change m here: len(bankTrain) or 1000
    trees = Bagging(bankTrain, t, len(bankTrain))
    train = Error(bankTrain, trees)
    train_error.append(train)
    test = Error(bankTest, trees)
    test_error.append(test)
    

for t in range(1, 6):
    trees = Bagging(bankTrain, t*100, len(bankTrain))
    train = Error(bankTrain, trees)
    train_error.append(train)
    test = Error(bankTest, trees)
    test_error.append(test)
    

# make plot
print("2b:")
num1 = [i for i in range(1, 51, 2)]   
num2 = [100,200,300,400,500] 
num = num1 + num2
plt.plot(num, train_error, label="Train")
plt.plot(num, test_error, label="Test")
plt.xlabel("Number of trees")
plt.ylabel("Error")
plt.title("Train and test Error based on number of trees when m=5000")
plt.legend()
plt.show()
