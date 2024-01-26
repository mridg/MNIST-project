import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup():
    # first layer
    w1 = np.zeros((10, 784))
    #w1 = np.random.randn(10, 784) #10 row 784 col
    b1 = np.zeros((10, 1))
    #b1 = np.random.randn(10, 1) #10 row 1 col
    # second layer
    w2 = np.zeros((10, 10))
    #w2 = np.random.randn(10, 10) #10 row 10 col
    b2 = np.zeros((10, 1))
    #b2 = np.random.randn(10, 1) #10 row 1 col

    return w1, b1, w2, b2

def forwardprop(w1, b1, w2, b2, A):
    z1 = sigmoid(np.matmul(w1, A) + b1) #10 row 1 col
    z2 = sigmoid(np.matmul(w2, z1) + b2) #10 row 1 col

    return z1, z2

#TODO: check if the code still work when the matrix dimensions change 
#TODO: check if adding more layers will help !! unlikely but see what it does 
#TODO: fun experiment, try learning rate scheduling

def backprop(w1, w2, b1, b2, z1, z2, input, ans, learningrate):
    dw2 = np.matmul(np.matmul(z1, sigmoidprime(z2).T), (z2 - ans)) * 2 #10 row 1 col
    db2 = np.matmul((z2 - ans), sigmoidprime(z2)) * 2 #10 row 1 col

    w2f = w2 - learningrate * dw2
    b2f = b2 - learningrate * db2

    dw1 = np.matmul(input, np.matmul(sigmoidprime(z1).T, (w2 - w2f))) * 2
    db1 = np.matmul(sigmoidprime(z1).T, (b2 - b2f)) * 2

    w1f = w1 - learningrate * dw1.T
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
    learningrate = 0.1

    #print(data.shape)
    # xpoints = np.array([1, 8])
    # ypoints = np.array([3, 10])

    # plt.plot(xpoints, ypoints)
    # plt.show()

    w1, b1, w2, b2 = setup() 
    for i in range(len(data)):
        reading = np.delete(data[i].T , 0,  axis=0) #initial greyscale inputs
        input = np.reshape(reading, (len(reading), 1))

        ans = np.zeros(10)
        ans[data[i][0]] = 1 #answer array for checking our results 

        z1, z2 = forwardprop(w1, b1, w2, b2, input)
        w1, b1, w2, b2 = backprop(w1, w2, b1, b2, z1, z2, input, ans, learningrate) 

        correct = accuracy(z2, ans, correct)
        if i % 1000 == 0:
            precent = correct / (1000) * 100
            print(f"Iteration {i} Accuracy is {precent:0.2f}%")
            correct = 0 

if __name__ == "__main__":
    main()