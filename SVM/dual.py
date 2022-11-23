import numpy as np
import scipy
from scipy.optimize import minimize

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
Dual form of SVM objective function
Simplify the double for loop using matrix operation
"""
def dualObjective(alpha, x, y):
    ayx = np.multiply(np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1))), x)
    prod = np.dot(ayx, ayx.T)
    summ = 0.5 * np.sum(prod)- np.sum(alpha)
    return summ


"""
Recover parameter b
"""
def dualFindB(alpha, w, C, X, y):
    b = 0
    num = 0
    for i in range(len(alpha)):
        if alpha[i] > 0 and alpha[i] < C:
            b += y[i] - w.dot(X[i])
            num += 1
    return b/num


"""
dual form of SVM
"""
def dualSVM(train_original, C):

    X = train_original[:, 0:4]
    y = train_original[:,4]
    xy = (X.dot(X.T))*(y.dot(y.T))

    cons = {'type':'eq', 'fun': lambda a: np.sum(a*y)}
    bounds = [(0, C)] * len(X)
    alpha0 = np.zeros((1, len(X)))
    result = minimize(dualObjective, alpha0, args=(X,y), method='SLSQP', bounds=bounds, constraints=cons)
    alphas = result.x

    # get weights
    w = np.sum(np.multiply(np.multiply(np.reshape(alphas,(-1,1)), np.reshape(y, (-1,1))), X), axis=0)    
    # get b
    b = dualFindB(alphas, w, C, X, y)
    
    return w, b
    
"""
Calculate GaussianKernel
"""
def GaussianKernel(xi, xj, gamma):
    xin = np.sum(xi**2, axis=1)
    xjn = np.sum(xj**2, axis=1)
    norm = xin.reshape(-1,1) + xjn.reshape(1,-1) - 2*xi.dot(xj.T)
    return np.exp(-norm/gamma)

"""
GK between two examples
"""
def GaussianKernel2(xi, xj, gamma):
    result = np.exp(-np.linalg.norm(xi-xj)**2 / gamma)
    return result
 
    
def dualObjectiveGK(alpha, y, k):
    ay = np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1)))
    prod = np.dot(ay, ay.T)
    summ = 0.5 * np.sum(np.multiply(prod, k))- np.sum(alpha)
    return summ


def dualFindBGK(alpha, C, X, y, gamma):
    b = 0
    num = 0
    for i in range(len(alpha)):
        if alpha[i] > 0:
            x = X[i,:]
            result = 0
            for j in range(len(X)):
                result += alpha[i]*y[i]*GaussianKernel2(x, X[j,:], gamma)
            b += result
            num += 1
    return b/num


def dualSVMGK(train_original, C, gamma):
    X = train_original[:, 0:4]
    y = train_original[:,4]
    k = GaussianKernel(X, X, gamma)# x and x?

    cons = {'type':'eq', 'fun': lambda a: np.sum(a*y)}
    bounds = [(0, C)] * len(X)
    alpha0 = np.zeros((1, len(X)))
    result = minimize(dualObjectiveGK, alpha0, args=(y,k), method='SLSQP', bounds=bounds, constraints=cons)
    alphas = result.x
    
    #idx = np.where((alphas > 0) & (alphas < C))
    #idx = np.where(alphas > 0)
    #K = GaussianKernel(X, X[idx], gamma)
    #blist = y[idx] - (alphas*y).dot(K)
    
    #b = blist.mean()
    b = dualFindBGK(alphas, C, X, y, gamma)
    return alphas, b

def PredictErrorGK(alphas, test, b, gamma):
    X = test[:,0:len(test[0])-1]
    y = test[:, -1]
    error = 0
    for i in range(len(y)):
        predict = 0
        xi = X[i,:]
        for j in range(len(y)):
            xj = X[j,:]
            predict += alphas[j]*y[j]*GaussianKernel2(xi, xj, gamma)
        
        predict += b
        
        sgn = 0
        if predict > 0:
            sgn = 1
        else:
            sgn = -1
            
        if sgn != y[i]:
            error += 1
    return error/len(y)

"""
Problem 3:
"""
Clist = [100/873, 500/873, 700/873]    
gammas = [0.1, 0.5, 1, 5, 100]


print('3a:') 
for c in Clist:
    w, b = dualSVM(train_original, c)
    w = np.insert(w, 0, b)
    error_train = PredictError(train, w)
    error_test = PredictError(test, w)
    print('C is:', c)
    print('weight:', w)
    print('train error:', error_train, 'test error:', error_test)



print('3b:')
for c in Clist:
    print('C:', c)
    for g in gammas:
        alphas, b = dualSVMGK(train_original, c, g)
        error = PredictErrorGK(alphas, train, b, g)
        error_test = PredictErrorGK(alphas, test, b, g)
        print('gamma:', g, 'train error:', error, 'test error:', error_test)


print('3c:')
support = []
for c in Clist:
    print('C:', c)
    for g in gammas:
        alphas, b = dualSVMGK(train_original, c, g)
        idx = np.where(alphas > 0)
        print('gamma:', g, 'support vectors:', len(idx[0]))
        
        if c == 500/873:
            support.append(alphas > 0)
     
print('Shared support vectors:')
for i in range(len(gammas)-1):
    count = 0
    for j in range(len(support[i])):
        if support[i][j] == support[i+1][j] and support[i][j] == True:
            count += 1
    
    print(f"{gammas[i]}, {gammas[i+1]}: {count}")
