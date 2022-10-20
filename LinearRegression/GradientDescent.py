import numpy as np
import math
import matplotlib.pyplot as plt
import random

"""
Compute cost 
"""
def cost(X, y, w):
    SSE = 0
    for i in range(len(X)):
        SSE += (y[i] - w.dot(X[i,:]))**2
    
    return SSE/2


"""
Compute gradient of J(w) at w^t
"""
def gradient(X, y, w):
    vec = []
    for i in range(len(w)):
        wj_sum = 0
        for j in range(len(X)):
            error = y[j] - w.dot(X[j,:])
            wj_sum = wj_sum + error*X[j,i]
        wj_sum = -wj_sum
        vec.append(wj_sum)
        
    vec = np.array(vec)    
    return vec



"""
Get norm of difference between w_t and w_t-1
alternative way is to call np.linalg.norm
"""
def getNorm(new, old):
    diff = new - old
    norm = 0
    for i in range(len(diff)):
        norm += diff[i]**2
    norm = math.sqrt(norm)
    return norm


"""
Batch Gradient Descent Algorithm
"""
def BatchGradientDescent(data, T, r):
    #w0 = np.array([1]*len(data))
    bound = len(data[0])-1
    X = data[:,0:bound]
    y = data[:, bound]
    
    # initialize w^0 to 0
    weight = np.zeros(bound)
    #weight=[-1,-1,1,-1]
    #weight = np.array(weight)
    cost_list = []
    for t in range(T):
        # compute cost
        J = cost(X, y, weight)
        cost_list.append(J)
        #print(J)
        # compute gradient of w^t
        gradi = gradient(X, y, weight)
        #print(gradi)
        
        # update w
        weight_new = weight - r*gradi
        #print(weight_new)
        norm = getNorm(weight, weight_new)
        weight = weight_new
        
        #print("N:",norm)
        if norm < 0.00001:
            print('Learned weight vector is:', [float('{0:0.3f}'.format(i)) for i in weight_new])
            print('Learning rate:', r, '. At iteration:', t)
            return weight_new, cost_list
       

"""
Stochastic Gradient Descent
"""
def StochasticGradientDescent(data, T, r):
    bound = len(data[0])-1
    X = data[:,0:bound]
    y = data[:, bound]
    random.seed(20)
    # initialize w^0 to 0
    weight = np.zeros(bound)
    cost_list = []
    ite = 0
    min_cost = 22
    return_w = []
    while ite < T:
            i = random.randint(0, len(data)-1)
            weight_new = []
            for j in range(len(weight)):
                
                error = y[i] - weight.dot(X[i,:])
                weight_new_j = weight[j] + r*error*X[i,j]
                weight_new.append(weight_new_j)
            
            weight_new = np.array(weight_new)
            weight = weight_new
            J = cost(X, y, weight_new)
            cost_list.append(J)
            ite += 1
            
            # get the optimal weight
            if J < min_cost:
                min_cost = J
                return_w = weight_new
            #print(J)
    
    return return_w, cost_list


# Read in dataset
train = np.genfromtxt('concrete/train.csv', delimiter = ',')
test = np.genfromtxt('concrete/test.csv', delimiter = ',')
# append 53 one as the first column in dataset, so the first element in w vector is b, but hard to converge,
# so finally choose weight veector of length 7
"""
one = np.ones([len(train)])
data = np.zeros([len(train), 9])
data[:,0] = one
data[:,1:9] = train
"""

# 4(a):
print('4a')
w, cost_list = BatchGradientDescent(train, 4000, 0.015)
t = [i for i in range(3534)]
plt.plot(t, cost_list)
plt.xlabel('step')
plt.ylabel('Cost')
plt.title('Cost function value of training data at each step')
plt.show()

test_cost = cost(test[:,0:7], test[:,7], w)
print('Cost of test data:', test_cost)

# 4(b):
print('4b')
weight_b, cost_list_b = StochasticGradientDescent(train, 500000, 0.005)
print("Learned weight vector is", [float('{0:0.3f}'.format(i)) for i in weight_b])
print("Learning rate is 0.005", "Steps is 500000")
t_b = [i for i in range(500000)]
plt.plot(t_b, cost_list_b)
plt.xlabel('step')
plt.ylabel('Cost')
plt.title('Cost function value of training data at each step')
plt.show()

test_cost_b = cost(test[:,0:7], test[:,7], weight_b)
print('Cost of test data:', test_cost_b)

# 4(c):
print('4c')
X=train[:,0:7]
y=train[:,7]
xdot = X.T.dot(X)
ww = np.linalg.inv(xdot).dot(X.T).dot(y)
print('Analytical form weight vector is:',[float('{0:0.3f}'.format(i)) for i in ww])


# code for checking part1.5
def StochasticGradientDescent2(data, T, r):
    bound = len(data[0])-1
    X = data[:,0:bound]
    y = data[:, bound]
    
    # initialize w^0 to 0
    weight = np.zeros(bound)  
    
    for i in range(len(data)):
            w_update = []
            gra = []
            for j in range(len(weight)):
                
                error = y[i] - weight.dot(X[i,:])
                gradi = error*X[i,j]
                gra.append(gradi)                
                weight_new_j = weight[j] + r*gradi
                
                w_update.append(weight_new_j)
            
            w_update = np.array(w_update)
            weight = w_update
            print("gradient:",gra)
            print("w_new:",w_update)
            J = cost(X, y, w_update)
    return

data2 = [[1,1,-1,2,1],[1,1,1,3,4],[1,-1,1,0,-1],[1,1,2,-4,-2],[1,3,-1,-1,0]]
data2 = np.array(data2)

#StochasticGradientDescent2(data2, 10, 0.1)
#BatchGradientDescent(data2, 250, 0.02)
X=data2[:,0:4]
y=data2[:,4]
xdot = X.T.dot(X)
ww = np.linalg.inv(xdot).dot(X.T).dot(y)
