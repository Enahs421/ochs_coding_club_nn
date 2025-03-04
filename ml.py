import numpy as np
from nnfs.datasets import spiral_data as sd
from matplotlib import pyplot as plt

accbarsize=100

class Layer:
    def __init__(self, ninputs, nneurons):
        self.ninputs = ninputs
        self.nneurons = nneurons

        self.weights = np.random.uniform(low=-1, high=1, size=(ninputs, nneurons))
        self.biases = np.zeros((1, nneurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def softmax(inputs):
    exp_inputs = np.exp(inputs)
    sum_exp_inputs = np.sum(exp_inputs)

    return exp_inputs/sum_exp_inputs

def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))

def relu(inputs):
    return np.maximum(0, inputs)

def cce(prediction, target):
    loss = -np.sum(target * np.log(prediction)) / target.shape[0]
    return loss

def accuracy(pis, tis):
    if(pis == tis):
        return 1
    else:
        return 0

nsamples = 10000
nclasses = 3

X, y = sd(samples=nsamples, classes=nclasses)

hlayer1 = Layer(2, 64)
hlayer2 = Layer(64, 64)
hlayer3 = Layer(64, 64)
olayer = Layer (64,3)

accuracy_count = 0

y_tries = np.array([])

for i in range(len(X)):
    inputs = X

    hlayer1.forward(inputs[i])
    hlayer2.forward(sigmoid(hlayer1.output))
    hlayer3.forward(sigmoid(hlayer2.output))
    olayer.forward(sigmoid(hlayer3.output))

    outputs = softmax(olayer.output)

    arr = np.zeros(nclasses)
    arr[y[i]]=1
    loss = cce(outputs, arr)

    accuracy_count += accuracy(np.argmax(outputs), np.argmax(arr))
    y_tries = np.append(y_tries, np.argmax(outputs))

    #print(f'Outputs: {outputs}')    
    #print(f'Loss: {loss}')

accuracy = accuracy_count/len(X)
accbarfill = accuracy * accbarsize

accbar = "["
for i in range(accbarsize):
    j = i+1
    if(j<=accbarfill):
        accbar = accbar + "@"
    else:
        accbar = accbar + '-'
accbar += ']'

print(f'Accuracy: {accuracy*100}% -', accbar)

input("Press enter to show graphs: ")

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()

plt.scatter(X[:,0], X[:,1], c=y_tries, cmap='brg')
plt.show()