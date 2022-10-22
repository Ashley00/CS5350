import numpy as np
import math
import DecisionTreeWeight as dt
import matplotlib.pyplot as plt
import ReadData as a

"""
AdaBoost algorithm with decision tree
"""
def AdaBoost(S, Attribute, Label, columns, T):
    # initialize D1
    D = [1/len(S)] * len(S)
    D = np.array(D)
  
    weakL = []
    votes = []
    # append weight as the last column in data set
    S_weight = np.zeros([len(S), len(columns)+1] , dtype=object)
    S_weight[:,0:len(columns)] = S
    S_weight[:,len(columns)] = D
    
    for j in range(T):
        # find a classifier
        tree = dt.ID3(S_weight, Attribute, Label, columns, 1, 0)
        #print(tree.name)
        
        # compute its vote
        error = dt.calcError(S_weight, tree, columns)
        #print(error)
        # np.log is base e
        vote = 1/2*np.log((1-error)/error)
        
        weakL.append(tree)
        votes.append(vote)
        
        # update the values of weights
        for row in S_weight:
            if row[-2] == dt.predict(row, tree, columns):
                row[-1] = row[-1] * math.exp(-vote)
            else:
                row[-1] = row[-1] * math.exp(vote)
        
        # normalize D
        Zt = sum(S_weight[:,-1])
        D = [d/Zt for d in S_weight[:,-1]]
        # update
        S_weight[:,-1] = D
        #print(sum(S_weight[:,-1]))
        #print(S_weight[:,-1][0:5])
        
    
    # final hypothesis
    predictions = [''] * len(S)
    
    positive = Label[0]
    negative = Label[1]
    for r in range(len(S)):
        final = 0
        for i in range(T):
            if dt.predict(row, weakL[i], columns) == positive:
                final += votes[i]*1
            else:
                final += votes[i]*(-1)
            
    
        if final >= 0:
            predictions[r] = positive
        else:
            predictions[r] = negative
        
    return predictions





train_predicts_a = AdaBoost(a.bankTrain, a.attributes2, a.labels2, a.columns2, 10)
#test_predicts_a = AdaBoost(bankTest, attributes2, labels2, columns2, 10)
#print(train_predicts_a[0:10])
    
    
    
"""            
train_list = []   
test_list = []         
for t in range(13):
    train_predicts_a = AdaBoost(bankTrain, attributes2, labels2, columns2, t)
    test_predicts_a = AdaBoost(bankTest, attributes2, labels2, columns2, t)

    train_error_a = 0
    test_error_a = 0
    for i in range(len(bankTrain)):
        if train_predicts_a[i] != bankTrain[i,len(columns2)-1]:
            train_error_a += 1
        if test_predicts_a[i] != bankTest[i,len(columns2)-1]:
            test_error_a += 1

    train_list.append(train_error_a/len(bankTrain))
    test_list.append(test_error_a/len(bankTest))
    print(t)
    #print(train_predicts_a[0:15])

print(train_list)
print(test_list)
t = [i for i in range(13)] 
plt.plot(t, train_list)   
"""

def error(predicts, S):
    train_correct = 0
    for i,row in enumerate(S):
        if predicts[i] == row[-1]:
            train_correct += 1
    train_error = 1-train_correct/len(S)
    
    return train_error
    

#train1 = error(train_predicts, bankTrain)
#test1 = error(test_predicts, bankTest)
#print(test1)


