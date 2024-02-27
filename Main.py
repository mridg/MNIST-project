import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup():
    # first layer
    w1 = np.random.rand(20, 784) - 0.5
    b1 = np.random.rand(20, 1) - 0.5
    # second layer
    w2 = np.random.rand(10, 20) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2

def forwardprop(w1, b1, w2, b2, A):
    z1 = w1.dot(A) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

def backprop(w1, w2, b1, b2, z1, z2, a1, a2, input, ans, learningrate):
    m = 1 / len(input)

    dz2 = a2 - ans
    dw2 = m * dz2.dot(a1.T)
    db2 = m * np.sum(dz2)

    w2f = w2 - learningrate * dw2
    b2f = b2 - learningrate * db2

    dz1 = w2.T.dot(dz2) * sigmoidprime(z1)
    dw1 = m * dz1.dot(input.T)
    db1 = m * np.sum(dz1)

    w1f = w1 - learningrate * dw1
    b1f = b1 - learningrate * db1

    return w1f, b1f, w2f, b2f

def ReLU(input):
    return np.maximum(0, input)

def ReLUprime(input):
    return input > 0 #this works because true converts to 1 and false converts to 0

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidprime(input):
    return sigmoid(input) * (1 - sigmoid(input))

def accuracy(z2, answer, correct):
    guess = np.argmax(z2)
    ans = np.argmax(answer)
    if guess == ans:
        correct += 1
    return correct

def main():
    """Write your mainline logic below this line (then delete this line)."""
    data = pd.read_csv("train.csv")
    data = np.array(data) #turns any array like object into an array! how wonderful! I want a greasy almond sandwich
    correct = 0
    learningrate = 1

    w1, b1, w2, b2 = setup() 
    for i in range(len(data)):
        reading = np.delete(data[i].T , 0,  axis=0) #initial greyscale inputs
        input = np.reshape(reading, (len(reading), 1))

        ans = np.zeros(10)
        ans[data[i][0]] = 1 #answer array for checking our results 
        ans.shape += (1,)

        z1, a1, z2, a2 = forwardprop(w1, b1, w2, b2, input)
        w1, b1, w2, b2 = backprop(w1, w2, b1, b2, z1, z2, a1, a2, input, ans, learningrate) 

        correct = accuracy(z2, ans, correct)
        if i % 1000 == 0:
            precent = correct / (1000) * 100
            print(f"Iteration {i} Accuracy is {precent:0.2f}%")
            correct = 0 

if __name__ == "__main__":
    main()