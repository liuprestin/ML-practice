# tutorial from http://adventuresinmachinelearning.com/neural-networks-tutorial/
#


# math desciption of a neuron
# f(z) = 1 / (1 + exp(-z))

import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8,8,0.1)
f = 1 / (1 + np.exp(-x))

plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
# feels like the logit plot again... its the sigmoid - to represent between 0 and 1
#
#

# for each node (perceptron) we take the average of all (input x weights)
# where the  weights are the values we are trying to be optimizing for
# In other words - many inputs for one output.
#
# y = f(XW + B) - instead use a matrix to have mulitple inputs
# where x is a row of inputs
# where w is is one column of weights
# B is the bias
# (wow this is'nt very complex)
#
# W and B is what we are iterating for.
#
# for example for a node with one input and one output:

w1 = 0.5
w2 = 1.0
w3 = 2.0
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'
for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x*w))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)
plt.show()
# ^ this shows the affects of ajusting the weights

# by using a bias term we can ajust values as if it were an if control statement
# reminds me of amplifiers in electronics

w = 5.0
b1 = -8.0
b2 = 0.0
b3 = 8.0
l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'
for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x*w+b)))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_wb(x)')
plt.legend(loc=2)
plt.show()

# https://causeyourestuck.io/2017/06/12/neural-network-scratch-theory/
# perceptrons are layered so that we can solve problems
# input - hidden - output layers
#
# forward propagation
# where the out - of the input layers is the IN for hidden layers (and for forth for outputlayer)
#
# backprogation
# loss function:
# J = 1/2(y' - y)^2  where y' is the desired output
#
# so we can get the GLOBAL MINIMA - so this is a gradient descent method to reach our FITTING answer.
# (looks like its from least squares method.)
#
# so we do the derivative:
#
# dJ/dB2 = dJ/dY * dY/dB2
# = (Y - Y')* f'(HW_2 + B_2)
# = omega
#
# where we use the sigmoid function:
# f'(x) = d/dx(1/(1 + exp(-x))) = (exp(-x))/(1+exp(-2))^2
#
# we take the dirivative with respect to all the parameters we seek:
#
# Where t is transpose
#
# dJ/dB2 = omega_2 = (Y - Y*) * f'(HW_2 + B2)
# dJ/dB1 = omega_1 = omega_2 * W2^t * f'(XW_1 + B1)
# dJ/dW2 = H^t * omega_2
# dJ/dW1 = X^t * omega_1


# Now for the iteration part - we want to modify the parameters as we
# iterate to what we want
#
# where alpha is the learning rate
#
# B2 = B2 - alpha * dJ/dB2
# B1 = B1 - alpha * dJ/dB1
# W2 = W2 - alpha * dJ/dW2
# W1 = W1 - alpha * dJ/dW1

# W is a matrix of all weights between layers
# it represents all combinations of inputs to the next layers

# let w1 be the array of weights for layer 1
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1,3))
w2[0,:] = np.array([0.5, 0.5, 0.5])

# now for the bias for each layer
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

# the sigmoid function
def f(x):
    return 1 / ( 1 + np.exp(-x))

# now for the feed forward function
# given  the number of layers in a NN , x input , we deal with weights and biases
# this is a manual iterative approach.
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        #Setup the input array which the weights will be multiplied by for each layer
        #If it's the first layer, the input array will be the x input vector
        #If it's not the first layer, the input to the next layer will be the
        #output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((w[l].shape[0],))
        #loop through the rows of the weight array
        for i in range(w[l].shape[0]):
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            #add the bias
            f_sum += b[l][i]
            #finally use the activation function to calculate the
            #i-th output i.e. h1, h2, h3
            h[i] = f(f_sum)
    return h

w = [w1, w2]
b = [b1, b2]
x = [1.5, 2.0, 3.0]
simple_looped_nn_calc(3, x, w, b)

# interesting for speeding up the algorithm - we add variable z?
# seems that we just use the dot product
#
# be aware that the computer can simplify our operations (would be nice to know how todo them by hand of course)

def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h

# what does a mathmatical model ask?
# what do the variables represent?
# why are the operations that way?
# what are the feedback loops within such a model?
# how is the model operatied?
# how do we get the solution to the weights? (newton's method, gradient descent?)
# how do we know how to ADJUST the weights? (ie. the error function)
# what kind of data/variables are you worknig with? what mathematics would be associated with that?
# what are some assumptions that are made with the data that may affect the oppertations that you perform with such a set?
#  what are all the steps in the process? how will it addup to a full algorithm?
#  for a particular method - what daatsets / setup does it need?

# for backprogation we can use gradient descent.
# gradient descent example:
x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

def df(x):
    y = 4 * x**3 - 9 * x**2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

print("The local minimum occurs at %f" % x_new)
# sidenote: analytically the solution can be found using calculus - but that isn't common
#  so computational methods maybe needed instead. As some functions may not have a solution - but an aproximation.
#  https://math.stackexchange.com/questions/1713372/how-do-we-calculate-the-gradient-from-numerical-data
#  https://geometrictools.com/Documentation/FiniteDifferences.pdf
#  https://en.wikipedia.org/wiki/Numerical_differentiation

# we can also minimize the cost functions
#  ie. how we adjust the weights - sum of squares error?

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[1])
plt.show()
# MNIST digit 1

# to make the data useful for us:
# 1. scale the data
# 2. split the data into test and training sets
digits.data[0,:]

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
X[0, :]
# splitting the dataset
from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# output layer setup - as we're going to reprent the result as a vector
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
y_train[0], y_v_train[0]

# define the structure of the neural network:
# 64 nodes for 64px image, 10 output layer nodes - for the prediction
nn_structure = [64, 30, 10]

# sigmoid function f(x)
# derivative of said sigmoid function:

def f_deriv(x):
    return f(x) * (1 - f(x))

# so now how do we train it?
#
# overall algorithm using gradient descent solution:
#
# - set W and B to zeros
# - for samples 1 to m:
# -- feed forward pass through all n layer - store activation function out: h
# -- caluculate omege for output layer
# -- backprogation for omega_2 to nl - 1
# -- update W and b for each layer
# -- gradient descent

import numpy.random as r
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

# init - set values to zero
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

# need feed forward function for the gradient descent

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

# final function:
# we're gonna set iteration value - instead of variance
def train_nn(nn_structure, X, y, iter_num=1000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

# now to see if the model is good:
#
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
accuracy_score(y_test, y_pred)*100
