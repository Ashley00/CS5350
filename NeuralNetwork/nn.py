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
Sigmoid function
"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
Derivative of sigmoid function
"""
def sigmoid_de(s):
    return sigmoid(s)*(1-sigmoid(s))


#print(sigmoid_de(np.dot(weights[2], nodes[2-1].reshape([-1,1])))[0][0])

"""
Forward pass for one example, calculate each node
"""
def forward_pass(x, weights, nodes, network, layer_num):
    nodes[0] = x
        
    for j in range(1, layer_num-1):
        nodes[j][1:] = sigmoid(np.dot(weights[j], nodes[j-1].reshape([-1,1])))
            
    out_i = layer_num - 1
    nodes[out_i] = np.dot(weights[out_i], nodes[out_i-1])
        
    return nodes[out_i][0][0]
    
"""
Backward for one example, calculate derivatives regarding weights
"""
def backward(ystar, weights, weights_de, nodes, network, layer_num):
    dLdy = nodes[-1][0][0] - ystar
    dydw = np.transpose(nodes[2])
    weights_de[-1] = dLdy*dydw
    #store = []
    hiddenz = network[-2]-1
    # step by step update
    for i in reversed(range(1, layer_num-1)):
        if i == 2:
            dydz2 = weights[-1][0][1:]
            #store = dLdy*dydz2
            sd = sigmoid_de(np.dot(weights[2], nodes[i-1].reshape([-1,1])))
            #print(nodes[i-1].reshape(-1))
            for j in range(hiddenz):
                dz2dw = sd[j][0]*nodes[i-1].reshape(-1) # need to reshape
                dLdw = dLdy*dydz2[j]*dz2dw
                weights_de[2][j] = dLdw
               
            
        if i == 1:
            sig = sigmoid_de(np.dot(weights[1], nodes[0].reshape([-1,1])))
            for j in range(hiddenz):
                dLdw = [weights_de[2][k][0] for k in range(hiddenz)]
                w = [weights[2][k][j+1] for k in range(hiddenz)]
                dLdw = np.array(dLdw)
                w = np.array(w)
             
                dLdz1 = sum(dLdw*w)
                dz1dw = sig[j][0]*nodes[i-1].reshape(-1)
                weights_de[1][j] = dLdz1*dz1dw


"""
Update weights
"""
def updateW(weights, weights_de, gamma, layer_num):
    for i in range(1, layer_num):
        weights[i] = weights[i] - gamma*weights_de[i]
    
    
    
"""
Stochastic gradient descent for training neural network
"""
def SGD(train, weights, weights_de, nodes, network, layer_num, gamma0, d):
    # 50 epoch
    for t in range(50):
        # for each epoch, shuffle the data set first
        np.random.shuffle(train)
        
        for row in train:
            store = []
            x = row[0:len(row)-1]
            store.append(x)
            store = np.array(store)
            ystar = row[-1]
            
            y = forward_pass(store, weights, nodes, network, layer_num)
            backward(ystar, weights, weights_de, nodes, network, layer_num)
            # update gamma
            gamma = gamma0/(1+gamma0*t/d)
            # update weight
            updateW(weights, weights_de, gamma, layer_num)
    
    return weights

"""
Calculate the error
"""
def predictError(test, weights, nodes, network, layer_num):
    error = 0
    
    for row in test:
        store = []
        x = row[0:len(row)-1]
        store.append(x)
        store = np.array(store)
        
        y = forward_pass(store, weights, nodes, network, layer_num)
        
        if y < 0:
            y = -1
        else:
            y = 1
        
        if y != row[-1]:
            error += 1
            
    return error/len(test)



"""
2a
"""
# test forward pass and backward with problem3
print('2a: back-propagation verify')
widtha = 3
x3 = np.array([[1,1,1]])
networka = [3, widtha, widtha, 1]
layer_numa = len(networka)
weightsa = [[], np.array([[-1,-2,-3], [1,2,3]]), np.array([[-1,-2,-3], [1,2,3]]), np.array([[-1,2,-1.5]])]
nodesa = [np.ones([networka[i], 1]) for i in range(4)]
ya = forward_pass(x3, weightsa, nodesa, networka, layer_numa) # -2.436
print('Using forward pass get y is ',ya)
print('nodes is')
print(nodesa)
ystar = 1       
weights_dea = [[], np.array([[0,0,0], [0,0,0]], dtype=float), np.array([[0,0,0], [0,0,0]], dtype=float), np.array([[0,0,0]])]
backward(ystar, weightsa, weights_dea, nodesa, networka, layer_numa)    
print('Using back propagation derivative is ')
print(weights_dea)
print('All results is the same as paper problem 3, which verify my implementation is correct')


"""
2b
"""
def setup2b(units):
    # represents the number of units in hidden layer
    width = units
    # overall nn structure, number of units in each layer
    network = [5, width, width, 1]
    layer_num = len(network)
    weights = [[] for i in range(layer_num)]
    weights_de = [[] for i in range(layer_num)]
    # easier to utilize matrix operation in this structure
    nodes = [np.ones([network[i], 1]) for i in range(4)]


    # initialize weights and derivitave of weights
    for i in range(1,layer_num-1):
        wi = np.random.normal(0,1,(network[i]-1, network[i-1]))
        weights[i] = wi
        weights_de[i] = np.zeros([network[i]-1, network[i-1]], dtype=float)
        out_i = layer_num - 1
        weights[out_i] = np.random.normal(0,1,(network[out_i], network[out_i-1]))
        weights_de[out_i] = np.zeros([network[out_i], network[out_i-1]])


    weights = SGD(train, weights, weights_de, nodes, network, layer_num, 0.1, 1)
    error_train = predictError(train, weights, nodes, network, layer_num)
    error_test = predictError(test, weights, nodes, network, layer_num)
    print('width:', units, 'train_error:', error_train, 'test_error:', error_test)
    
print('2b: Initialize weights from standard Gaussian distribution')
width_list = [5,10,25,50,100]
for units in width_list:
    setup2b(units)
    


"""
2c
"""
def setup2c(units):
    # represents the number of units in hidden layer
    width = units
    # overall nn structure, number of units in each layer
    network = [5, width, width, 1]
    layer_num = len(network)
    weights = [[] for i in range(layer_num)]
    weights_de = [[] for i in range(layer_num)]
    # easier to utilize matrix operation in this structure
    nodes = [np.ones([network[i], 1]) for i in range(4)]


    # initialize weights and derivitave of weights
    for i in range(1,layer_num-1):
        wi = np.zeros([network[i]-1, network[i-1]], dtype=float)
        weights[i] = wi
        weights_de[i] = np.zeros([network[i]-1, network[i-1]], dtype=float)
        out_i = layer_num - 1
        weights[out_i] = np.zeros([network[out_i], network[out_i-1]], dtype=float)
        weights_de[out_i] = np.zeros([network[out_i], network[out_i-1]])


    weights = SGD(train, weights, weights_de, nodes, network, layer_num, 0.1, 1)
    error_train = predictError(train, weights, nodes, network, layer_num)
    error_test = predictError(test, weights, nodes, network, layer_num)
    print('width:', units, 'train_error:', error_train, 'test_error:', error_test)
#change b and gamma    
print('2c: Initialize weights with 0')
width_list = [5,10,25,50,100]
for units in width_list:
    setup2c(units)
