import numpy as np
import ReadData as a
import DecisionTreeFeature as dt
import random
import matplotlib.pyplot as plt

"""
Random Forest algorithm
"""
def RandomForest(S, T, m, featureSize):
    trees = []
    for t in range(T):
        # draw m uniformly with replacement
        samples = DrawSamples(S, m)
        # learn a decision tree, choose feature number
        tree = dt.ID3(samples, a.attributes2, a.labels2, "Entropy", a.columns2, 16, 0, featureSize)
        trees.append(tree)    
            
    return trees



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



# Read in data
bankTrain = a.bankTrain
bankTest = a.bankTest


print("2d:")

def draw(featureSize):
    train_error = []
    test_error = []
    for t in range(1, 51, 2):
        # change m here: len(bankTrain) or 1000
        trees = RandomForest(bankTrain, t, 1000, featureSize)
        train = Error(bankTrain, trees)
        train_error.append(train)
        test = Error(bankTest, trees)
        test_error.append(test)
        
    for t in range(1, 6):
        trees = RandomForest(bankTrain, t*100, 1000, featureSize)
        train = Error(bankTrain, trees)
        train_error.append(train)
        test = Error(bankTest, trees)
        test_error.append(test)
        
    # make plot
    print("featur size", featureSize)
    num1 = [i for i in range(1, 51, 2)]   
    num2 = [100,200,300,400,500] 
    num = num1 + num2
    plt.plot(num, train_error, label="Train")
    plt.plot(num, test_error, label="Test")
    plt.xlabel("Number of trees")
    plt.ylabel("Error")
    plt.title("Train and test Error based on number of random trees")
    plt.legend()
    plt.show()
    

    

f = [2,4,6]
for featureSize in f:
    draw(featureSize)





