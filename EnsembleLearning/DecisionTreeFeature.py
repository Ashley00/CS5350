import numpy as np
import math
from collections import Counter
import random 

"""
Node class for Tree construction
"""        
class Node:
    def __init__(self, name):
        self.name=name
        self.child = []
        self.branches = {}
        
        
"""
ID3 Algorithm
"""
def ID3(S, Attribute, Label, splitName, columns, maxDepth, currDepth, featureNum):
    currLabels = set(S[:, len(columns)-1])
    
    if len(currLabels) == 1:
        l = list(currLabels)[0]
        return Node(l)
    
    if len(Attribute.keys()) == 0:
        commonL = commonLabel(S, columns)
        return Node(commonL)
    # reach max depth, choose most common attribute
    if currDepth == maxDepth:
        commonL = commonLabel(S, columns)
        return Node(commonL)
    
    else:
        # add feature select
        if len(Attribute) > featureNum:
            key = random.sample(Attribute.keys(), featureNum)
            G = {k: Attribute[k] for k in key}
            A = split(S, G, Label, splitName, columns)
        else:
            A = split(S, Attribute, Label, splitName, columns)
        
        #print(A)
        root = Node(A)
        for branch in Attribute[A]:
            root.branches[branch] = ""
            S_v = S[S[:,columns.index(A)]==branch]
            if len(S_v) == 0:
                commonL = commonLabel(S, columns)
                root.branches[branch] = Node(commonL)
            else:                
                AttributeCopy = Attribute.copy()
                del AttributeCopy[A]
               
                root.branches[branch] = ID3(S_v, AttributeCopy, Label, splitName, columns, maxDepth, currDepth+1, featureNum)
        return root
    
 
"""
Split the dataset based on different information gain method
"""
def split(S, Attribute, Label, name, columns):
    ig = 0
    maxA = 0
    splitA = list(Attribute.keys())[0]
    for A in Attribute:
        if name == "Entropy":
            ig = IG(S, Attribute, A, entropy, columns)
        elif name == "MajorityError":
            ig = IG(S, Attribute, A, majorityerror, columns)
        elif name == "GiniIndex":
            ig = IG(S, Attribute, A, gini, columns)
        # debug: print("IG:",ig)    
        # find the attribute with max InformationGain    
        if ig > maxA:
            maxA = ig
            splitA = A
    return splitA
    

"""
Calculate Information Gain
"""
def IG(S, Attribute, A, calc, columns):    
    gain = calc(S, Attribute, columns)
    
    for v in Attribute[A]:
        S_v = S[S[:, columns.index(A)]==v]
        gain = gain - len(S_v)/len(S)*calc(S_v, Attribute, columns)
           
    return gain
    
"""
Calculate entropy
"""        
def entropy(S, Attribute, columns):
    label = Counter(S[:,len(columns)-1])
    l = [(i, label[i]/len(S)) for i in label]
    
    entropyS = 0
    for probability in l:
        if probability[1] == 0:
            continue
        entropyS = entropyS - probability[1]*math.log(probability[1],2)
    #print(entropyS)    
    return entropyS

"""
Calculaate Majority Error
"""
def majorityerror(S, Attribute, columns):
    if len(S[:,len(columns)-1]) == 0:
        me = 1
    else:
        label = Counter(S[:,len(columns)-1])
        me = 1 - label.most_common(1)[0][1]/len(S)
    
    return me
    
"""
Calculate Gini Index
"""    
def gini(S, Attribute, columns):
    label = Counter(S[:,len(columns)-1])
    l = [(i, label[i]/len(S)) for i in label]
    
    giniS = 1
    for probability in l:
        giniS = giniS - probability[1]**2
    return giniS

    
"""
Find most common labels in dataset
"""
def commonLabel(S, columns):   
    c = Counter(S[:, len(columns)-1])
    return c.most_common(1)[0][0]



"""
Read csv file
"""
def readCSV(fileName):
    data = []
    with open (fileName, "r") as f:
        for line in f:
            terms = line.strip().split(",")
            data.append(terms)
    data = np.array(data)
    return data
    
"""
Predict the data label
"""    
def predict(data, tree, columns):
    currNode = tree
    # find leaf node 
    while len(currNode.branches) != 0:
        name = currNode.name
        attriName = data[columns.index(name)]
        currNode = currNode.branches[attriName]
    return currNode.name

"""
Calculate error
"""
def calcError(S, tree, columns):
    correct = 0
    for row in S:
        if row[len(columns)-1] == predict(row, tree, columns):
            correct = correct + 1
    error = 1 - correct/len(S)
    #print(correct, len(S), error)
    return error
