import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup():
    # first layer
    w1 = np.random.rand(20, 784) - 0.5
    #w1 = np.random.randn(10, 784) #10 row 784 col
    b1 = np.random.rand(20, 1) - 0.5
    #b1 = np.random.randn(10, 1) #10 row 1 col
    # second layer
    w2 = np.random.rand(10, 20) - 0.5
    #w2 = np.random.randn(10, 10) #10 row 10 col
    b2 = np.random.rand(10, 1) - 0.5
    #b2 = np.random.randn(10, 1) #10 row 1 col

    return w1, b1, w2, b2

def forwardprop(w1, b1, w2, b2, A):
    #z1 = sigmoid(np.matmul(w1, A) + b1) #10 row 1 col
    z1 = w1.dot(A) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    #z2 = sigmoid(np.matmul(w2, z1) + b2) #10 row 1 col

    return z1, a1, z2, a2

#TODO: check if the code still work when the matrix dimensions change 
#TODO: check if adding more layers will help !! unlikely but see what it does 
#TODO: fun experiment, try learning rate scheduling

def backprop(w1, w2, b1, b2, z1, z2, a1, a2, input, ans, learningrate):
    #dw2 = np.matmul(np.matmul(z1, sigmoidprime(z2).T), (z2 - ans)) * 2 #10 row 1 col
    #dw2 = 2 * (z2 - ans) @ sigmoidprime(w2 @ z1 + b2).T @ z1 # might be times z1?
    #db2 = np.matmul((z2 - ans), sigmoidprime(z2)) * 2 #10 row 1 col
    #db2 = 2 * (z2 - ans) * sigmoidprime(w2 @ z1 + b2) 
    m = 1 / len(input)

    dz2 = a2 - ans
    dw2 = m * dz2.dot(a1.T)
    db2 = m * np.sum(dz2)

    w2f = w2 - learningrate * dw2
    b2f = b2 - learningrate * db2

    #dw1 = np.matmul(input, np.matmul(sigmoidprime(z1).T, (w2 - w2f))) * 2
    #dw1 = (2 * (z2 - ans) * sigmoidprime(w2 @ z1 + b2) * z1 * sigmoidprime(w1 @ input + b1)) @ input.T
    #dw1 = 2 * (z1 - (np.log(ans / (1 - ans))/ w2f)) @ sigmoidprime(w1 @ input + b1) @ input.T

    #db1 = np.matmul(sigmoidprime(z1).T, (b2 - b2f)) * 2
    #db1 = 2 * (z2 - ans) @ sigmoidprime(w2 @ z1 + b2).T @ sigmoidprime(w1 @ input + b1)
    #db1 = 2 * (z1 - (np.log(ans / (1 - ans))/ w2f)) @ sigmoidprime(w1 @ input + b1)

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