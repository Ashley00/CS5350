import numpy as np


"""
Calculate the sign of dot product
"""
def sgn(w, x):
    d = w.T.dot(x)
    if d < 0:
        return 0
    else:
        return 1


"""
Calculate the prediction error on test data
"""
def PredictError(test, w):
    error = 0
    
    for row in test:
        x = row[0:len(row)-1]
        if sgn(w, x) != row[-1]:
            error += 1
            
    return error/len(test)


"""
Calculate the average prediction error on test data using Voted Perceptron
"""
def AvePredictError(test, w_list, c_list):
    error = 0
    
    
    for row in test:
        x = row[0:len(row)-1]
        
        # average prediction for one example
        predic = 0
        for i in range(len(w_list)):
            w = w_list[i]
            c = c_list[i]
            if sgn(w, x) == 1:
                predic += c
            else:
                predic -= c
        if predic > 0:
            predic = 1
        else:
            predic = 0
            
        if predic != row[-1]:
            error += 1
            
    return error/len(test)



"""
Standard Perceptron Algorithm
"""
def StandardPerceptron(data, r, T):
    # feature size
    n = len(data[0]) - 1
    # initialize w0
    w = np.zeros(n)
    
    for i in range(T):
        # shuffle the data
        np.random.shuffle(data)
        # for each training example
        for row in data:
            x = row[0:len(row)-1]
            y_predict = sgn(w, x)
            if y_predict != row[-1]:
                if row[-1] == 1:
                    w = w + r*x
                else:
                    w = w - r*x
                       
    return w

"""
Voted Perceptron Algorithm
"""
def VotedPerceptron(data, r, T):
    # feature size
    n = len(data[0]) - 1
    # initialize w0
    w = np.zeros(n)
    m = 0
    
    w_list = []
    c_list = []
    for i in range(T):
        for row in data:
            x = row[0:len(row)-1]
            y_predict = sgn(w, x)
            if y_predict != row[-1]:
                if row[-1] == 1:
                    w = w + r*x
                else:
                    w = w - r*x
                    
                w_list.append(w)   
                c_list.append(1)
                m = m + 1
                
            else:
                c_list[m-1] = c_list[m-1] + 1
    
    return w_list, c_list


"""
Average Perceptron Algorithm
"""
def AveragePerceptron(data, r, T):
    # feature size
    n = len(data[0]) - 1
    # initialize w0
    w = np.zeros(n)
    a = np.zeros(n)
    
    for i in range(T):
        # for each training example
        for row in data:
            x = row[0:len(row)-1]
            y_predict = sgn(w, x)
            if y_predict != row[-1]:
                if row[-1] == 1:
                    w = w + r*x
                else:
                    w = w - r*x
                    
            # update a
            a = a + w
                       
    return a






"""
Some preprocess for training and testing data
"""
train = np.genfromtxt('bank-note/train.csv', delimiter = ',')
test = np.genfromtxt('bank-note/test.csv', delimiter = ',')

# bias term can be treated as a constant feature folded into w
# first element in w is b
one = np.ones([len(train)])
data = np.zeros([len(train), len(train[0,:])+1])
data[:,0] = one
data[:, 1:6] = train
   
one_test = np.ones([len(test)])
data_test = np.zeros([len(test), len(test[0,:])+1])
data_test[:,0] = one_test
data_test[:,1:6] = test

data_b = data.copy()

"""
2a:
"""
r = 0.05
T = 10
w = StandardPerceptron(data, r, T)
error = PredictError(data_test, w)

print("2a:")
print("Weight is", w)
print("Prediction error is", error)
  

"""
2b:
"""
r = 0.05
T = 10
w_list, c_list = VotedPerceptron(data_b, r, T)
error_b = AvePredictError(data_test, w_list, c_list)

print("2b:")
for i in range(len(w_list)):
    print('Weight:', w_list[i], 'Count:', c_list[i])

print("Prediction error is", error_b)

"""
2c:
"""
r = 0.05
T = 10
w_c = AveragePerceptron(data_b, r, T)
error_c = PredictError(data_test, w_c)

print("2c:")
print("Weight is", w_c)
print("Prediction error is", error_c)
  
