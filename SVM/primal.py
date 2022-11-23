import numpy as np

"""
Some preprocess for training and testing data
"""
train_original = np.genfromtxt('bank-note/train.csv', delimiter = ',')
test_original = np.genfromtxt('bank-note/test.csv', delimiter = ',')

# convert 0 to -1
np.place(train_original[:,4], train_original[:,4]==0, [-1])
np.place(test_original[:,4], test_original[:,4]==0, [-1])

# bias term can be treated as a constant feature folded into w
# first element in w is b
one = np.ones([len(train_original)])
train = np.zeros([len(train_original), len(train_original[0,:])+1])
train[:,0] = one
train[:, 1:6] = train_original
   
one_test = np.ones([len(test_original)])
test = np.zeros([len(test_original), len(test_original[0,:])+1])
test[:,0] = one_test
test[:,1:6] = test_original

"""
Calculate the sign of dot product
"""
def sgn(w, x):
    d = w.T.dot(x)
    if d < 0:
        return -1
    else:
        return 1


"""
Calculate the prediction error on data set
"""
def PredictError(test, w):
    error = 0
    
    for row in test:
        x = row[0:len(row)-1]
        if sgn(w, x) != row[-1]:
            error += 1
            
    return error/len(test)

"""
primal form of SVM
"""
def primalSVM(train, schedule, C):
    # feature size
    n = len(train[0]) - 1
    # initialize w0
    weights = np.zeros(n)
    T = 100
    r0 = 0.01
    a = 1

    N = len(train)
    i = 0

    #summ = 0
    for t in range(T):
        # for each epoch, shuffle the data set first
        np.random.shuffle(train)
    
        for row in train:
            # update learning rate
            if schedule == 1:
                r = r0/(1+r0*i/a)
            else:
                r0 = 0.001
                r = r0/(1+i)
                
            #summ += r
            i += 1
            w = np.copy(weights)
            w[0] = 0
    
            x = row[0:len(row)-1]
            y = row[-1]
            d = y*weights.T.dot(x)
            if d <= 1:
                weights = weights - r*w + r*C*N*y*x
            else:
                weights = weights - r*w
    
    return weights

"""
Problem 2:
"""
Clist = [100/873, 500/873, 700/873]    

# 2a, schedule is 1, r = r0/(1+r0*i/a)
print('2a:')
for c in Clist:
    w = primalSVM(train, 1, c)
            
    error_train_a = PredictError(train, w)
    error_test_a = PredictError(test, w)
    print('C is:', c)
    print('train error:', error_train_a, 'test error:', error_test_a)
    print('weight:', w)


# 2b, schedule is 2, r = r0/(1+i)
print('2b:')
for c in Clist:
    w = primalSVM(train, 2, c)
            
    error_train_a = PredictError(train, w)
    error_test_a = PredictError(test, w)
    print('C is:', c)
    print('train error:', error_train_a, 'test error:', error_test_a)
    print('weight:', w)
