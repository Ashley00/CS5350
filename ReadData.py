import numpy as np
import DecisionTreeWeight as dt
import random
import matplotlib.pyplot as plt

# Read in bank dataset
bankTrain = dt.readCSV("Bank/train.csv")  
bankTest = dt.readCSV("Bank/test.csv")

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


#key = random.sample(attributes2.keys(), 2)
#G = {k: attributes2[k] for k in key}
#print(len(G))

# Preprocess for bank dataset
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
