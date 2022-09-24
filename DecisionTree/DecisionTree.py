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
def ID3(S, Attribute, Label, splitName, columns, maxDepth, currDepth):
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
               
                root.branches[branch] = ID3(S_v, AttributeCopy, Label, splitName, columns, maxDepth, currDepth+1)
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




# For Question 2: Car dataset

columns = ["buying","maint","doors","persons","lug_boot","safety","label"]
labels = ["unacc", "acc", "good", "vgood"]
attributes = {"buying": ["vhigh", "high", "med", "low"], "maint": ["vhigh", "high", "med", "low"],
              "doors": ["2", "3", "4", "5more"], "persons": ["2", "4", "more"],
              "lug_boot": ["small", "med", "big"], "safety": ["low", "med", "high"]}


car = readCSV("car_train.csv")
carTest = readCSV("car_test.csv")

"""
# Debug part: construct trees
carTree = ID3(carTest, attributes, labels, "MajorityError", columns, 3, 0)
errorE = calcError(carTest, carTree, columns)
print(errorE)

print(carTree.name)
print(list(carTree.branches))
for i in list(carTree.branches):
    node = carTree.branches[i]
    print(node.name)
    print(list(node.branches))
    
    for j in list(node.branches):
        node2 = node.branches[j]
        print(node2.name)
        print(list(node2.branches))
     
"""

# predict for training set
print("CarTraingPredict:")
for i in range(1,7):
    print("Max Depth is ", i, ": ")
    carTreeE = ID3(car, attributes, labels, "Entropy", columns, i, 0)
    errorE = calcError(car, carTreeE, columns)  
    carTreeM = ID3(car, attributes, labels, "MajorityError", columns, i, 0)
    errorM = calcError(car, carTreeM, columns)  
    carTreeG = ID3(car, attributes, labels, "GiniIndex", columns, i, 0)
    errorG = calcError(car, carTreeG, columns)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG)   


# predict for testing set
print("CarTestingPredict:")
for i in range(1,7):
    print("Max Depth is ", i, ": ")
    carTreeE = ID3(carTest, attributes, labels, "Entropy", columns, i, 0)
    errorE = calcError(carTest, carTreeE, columns)  
    carTreeM = ID3(carTest, attributes, labels, "MajorityError", columns, i, 0)
    errorM = calcError(carTest, carTreeM, columns)  
    carTreeG = ID3(carTest, attributes, labels, "GiniIndex", columns, i, 0)
    errorG = calcError(carTest, carTreeG, columns)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG)      

    


# For Question 3a: Bank Predict

columns2 = ['age','job','marital','education','default','balance',
            'housing','loan','contact','day','month','duration',
            'campaign','pdays','previous','poutcome','y']
labels2 = ['yes', 'no']
attributes2 = {
    'age':['h','l'],
    'job':['admin.','unknown','unemployed','management','housemaid',
        'entrepreneur','student','blue-collar','self-employed','retired',
        'technician','services'],
    'marital':['married','divorced','single'],
    'education':['unknown','secondary','primary','tertiary'],
    'default':['yes','no'],
    'balance':['h','l'],
    'housing':['yes','no'],
    'loan':['yes','no'],
    'contact':['unknown','telephone','cellular'],
    'day':['h','l'],
    'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration':['h','l'],
    'campaign':['h','l'],
    'pdays':['h','l'],
    'previous':['h','l'],
    'poutcome':['unknown','other','failure','success']
}    
        

bankTrain = readCSV("bank_train.csv")  
bankTest = readCSV("bank_test.csv")

# Preprocess for dataset
threshold = {'age':0.0,'balance':0.0,'day':0.0,'duration':0.0,'campaign':0.0,'pdays':0.0,'previous':0.0}        
for name in list(threshold.keys()):
    subCol = bankTrain[:,columns2.index(name)]
    subCol = subCol.astype(np.float64)
    media = np.median(subCol)
    threshold[name] = media

for row in bankTrain:
    for name in list(threshold.keys()):
        i = columns2.index(name)
        if float(row[i]) < threshold[name]:
            row[i] = 'l'
        else:
            row[i] = 'h'

for row in bankTest:
    for name in list(threshold.keys()):
        i = columns2.index(name)
        if float(row[i]) < threshold[name]:
            row[i] = 'l'
        else:
            row[i] = 'h'

"""
#Debug part
carTreeE = ID3(bankTrain, attributes2, labels2, "MajorityError", columns2, 3, 0)
errorE = calcError(bankTrain, carTreeE, columns2)
print(errorE)

print(carTreeE.name)
print(list(carTreeE.branches))

for i in list(carTreeE.branches):
    node = carTreeE.branches[i]
    print(node.name)
    print(list(node.branches))

  
    #print(node.branches)
    for j in list(node.branches):
        node2 = node.branches[j]
        print(node2.name)
        print(list(node2.branches))
"""
          
# predict for training set
print("3a-BankTraingPredict:")
for i in range(1,17):
    print("Max Depth is ", i, ": ")
    bankTreeE = ID3(bankTrain, attributes2, labels2, "Entropy", columns2, i, 0)
    errorE = calcError(bankTrain, bankTreeE, columns2)  
    bankTreeM = ID3(bankTrain, attributes2, labels2, "MajorityError", columns2, i, 0)
    errorM = calcError(bankTrain, bankTreeM, columns2)  
    bankTreeG = ID3(bankTrain, attributes2, labels2, "GiniIndex", columns2, i, 0)
    errorG = calcError(bankTrain, bankTreeG, columns2)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG)             



# predict for testing set
print("3a-BankTestingPredict:")
for i in range(1,17):
    print("Max Depth is ", i, ": ")
    bankTreeE = ID3(bankTest, attributes2, labels2, "Entropy", columns2, i, 0)
    errorE = calcError(bankTest, bankTreeE, columns2)  
    bankTreeM = ID3(bankTest, attributes2, labels2, "MajorityError", columns2, i, 0)
    errorM = calcError(bankTest, bankTreeM, columns2)  
    bankTreeG = ID3(bankTest, attributes2, labels2, "GiniIndex", columns2, i, 0)
    errorG = calcError(bankTest, bankTreeG, columns2)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG) 


# For Question 3b: Bank Predict

# Handle unknown entry, find majority in training set
job = Counter(bankTrain[:, columns2.index('job')])
jobUnknown = job.most_common(1)[0][0]
edu = Counter(bankTrain[:, columns2.index('education')])
eduUnknown = edu.most_common(1)[0][0]
contact = Counter(bankTrain[:, columns2.index('contact')])
contactUnknown = contact.most_common(1)[0][0]
po = Counter(bankTrain[:, columns2.index('poutcome')])
poUnknown = po.most_common(2)[1][0]
 

for row in bankTrain:
    if row[columns2.index('job')] == 'unknown':
        row[columns2.index('job')] = jobUnknown
    if row[columns2.index('education')] == 'unknown':
        row[columns2.index('education')] = eduUnknown
    if row[columns2.index('contact')] == 'unknown':
        row[columns2.index('contact')] = contactUnknown  
    if row[columns2.index('poutcome')] == 'unknown':
        row[columns2.index('poutcome')] = poUnknown


for row in bankTest:
    if row[columns2.index('job')] == 'unknown':
        row[columns2.index('job')] = jobUnknown
    if row[columns2.index('education')] == 'unknown':
        row[columns2.index('education')] = eduUnknown
    if row[columns2.index('contact')] == 'unknown':
        row[columns2.index('contact')] = contactUnknown  
    if row[columns2.index('poutcome')] == 'unknown':
        row[columns2.index('poutcome')] = poUnknown

# predict for training set
print("3b-BankTraingPredict:")
for i in range(1,17):
    print("Max Depth is ", i, ": ")
    bankTreeE = ID3(bankTrain, attributes2, labels2, "Entropy", columns2, i, 0)
    errorE = calcError(bankTrain, bankTreeE, columns2)  
    bankTreeM = ID3(bankTrain, attributes2, labels2, "MajorityError", columns2, i, 0)
    errorM = calcError(bankTrain, bankTreeM, columns2)  
    bankTreeG = ID3(bankTrain, attributes2, labels2, "GiniIndex", columns2, i, 0)
    errorG = calcError(bankTrain, bankTreeG, columns2)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG)

# predict for testing set
print("3b-BankTestingPredict:")
for i in range(1,17):
    print("Max Depth is ", i, ": ")
    bankTreeE = ID3(bankTest, attributes2, labels2, "Entropy", columns2, i, 0)
    errorE = calcError(bankTest, bankTreeE, columns2)  
    bankTreeM = ID3(bankTest, attributes2, labels2, "MajorityError", columns2, i, 0)
    errorM = calcError(bankTest, bankTreeM, columns2)  
    bankTreeG = ID3(bankTest, attributes2, labels2, "GiniIndex", columns2, i, 0)
    errorG = calcError(bankTest, bankTreeG, columns2)
    print("Entropy: ", errorE, " MajorityError: ", errorM, " GiniIndex: ", errorG)
