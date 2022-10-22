import numpy as np
import math
from collections import Counter


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
def ID3(S, Attribute, Label, columns, maxDepth, currDepth):
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
        A = split(S, Attribute, Label, columns)
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
               
                root.branches[branch] = ID3(S_v, AttributeCopy, Label, columns, maxDepth, currDepth+1)
        return root
    
 
"""
Split the dataset based on different information gain method
"""
def split(S, Attribute, Label, columns):
    ig = 0
    maxA = 0
    splitA = list(Attribute.keys())[0]
    for A in Attribute:        
        ig = IG(S, Attribute, A, entropy, columns)
      
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
        #S_v, D_v= subset(S, D, columns, A, v)
        S_v = S[S[:, columns.index(A)]==v]
        gain = gain - len(S_v)/len(S)*calc(S_v, Attribute, columns)
           
    return gain

    
def subset(S, D, columns, A, v):
    S_v = []
    D_v = []
    for i,row in enumerate(S):
        if row[columns.index(A)]==v:
            S_v.append(row)
            D_v.append(D[i])
    
    return S_v, D_v
    
    
"""
Calculate entropy
"""        
def entropy(S, Attribute, columns):
    l = {'yes': 0.0, 'no': 0.0}
    yes = 0.0
    no = 0.0
    for row in S:
        if row[-2] == 'no':
            no = no + row[-1]
        else:
            yes = yes + row[-1]
    
    l['yes'] = yes
    l['no'] = no
    #print(list(l.items()))
    entropyS = 0
    for label,prob in l.items():
        if prob == 0:
            continue
        entropyS = entropyS - prob*math.log(prob,2)        
    #entropyS = 0-yes*math.log(yes,2)-no*math.log(no,2)
    """
    label = Counter(S[:,len(columns)-1])
    l = [(i, label[i]/len(S)) for i in label]
    
    entropyS = 0
    for probability in l:
        if probability[1] == 0:
            continue
        entropyS = entropyS - probability[1]*math.log(probability[1],2)
    #print(entropyS)    
    """
    return entropyS


    
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
        # prediction compare to actual value
        if row[-2] == predict(row, tree, columns):
            correct = correct + row[-1]
    error = 1 - correct
    #print(correct, len(S), error)
    return error

