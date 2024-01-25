import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def forwardprop(weight, bias, input):
#     z = weight.dot(input) + bias # issue with the dot product
#     z = ReLU(z)
#     return z
def setup():
    # first layer
    w1 = np.random.randn(10, 784) #10 row 784 col
    b1 = np.random.randn(10, 1) #10 row 1 col
    # second layer
    w2 = np.random.randn(10, 10) #10 row 10 col
    b2 = np.random.randn(10, 1) #10 row 1 col

    return w1, b1, w2, b2

def test1234():
    input = np.random.randn(1, 784)
    sigmoidz1 = np.random.randn(1, 10)
    cost = np.random.randn(10, 10)
    
    output = np.matmul(input.T, np.matmul(sigmoidz1, cost))
    return output

def forwardprop(w1, b1, w2, b2, A):
    z1 = sigmoid(np.matmul(w1, A) + b1) #10 row 1 col
    z2 = sigmoid(np.matmul(w2, z1) + b2) #10 row 1 col

    return z1, z2

def backprop(w1, w2, b1, b2, z1, z2, input, ans, learningrate):
    dw2 = np.matmul(np.matmul(z1, sigmoidprime(z2).T), (z2 - ans)) * 2 #10 row 1 col
    db2 = np.matmul((z2 - ans), sigmoidprime(z2)) * 2 #10 row 1 col

    w2f = w2 - learningrate * dw2.T
    b2f = b2 - learningrate * db2

    dw1 = np.matmul(input, np.matmul(sigmoidprime(z1).T, (w2 - w2f))) * 2
    db1 = np.matmul(sigmoidprime(z1).T, (b2 - b2f)) * 2

    w1f = w1 - learningrate * dw1.T
    b1f = b1 - learningrate * db1

    return w1f, b1f, w2f, b2f

def ReLU(input):
    return np.maximum(0, input)

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidprime(input):
    return sigmoid(input) * (1 - sigmoid(input))

def accuracy(z2, answer, correct):
    guess = np.argmax(z2)
    ans = np.argmax(answer)
    if guess == ans:
        correct = correct + 1
    
    return correct

def main():
    """Write your mainline logic below this line (then delete this line)."""
    data = pd.read_csv("train.csv")
    data = np.array(data) #turns any array like object into an array! how wonderful! I want a greasy almond sandwich
    correct = 0

    #print(data.shape)
    # xpoints = np.array([1, 8])
    # ypoints = np.array([3, 10])

    # plt.plot(xpoints, ypoints)
    # plt.show()


    w1, b1, w2, b2 = setup() 
    for i in range(10000):
        reading = np.delete(data[i].T , 0,  axis=0) #initial greyscale inputs
        input = np.reshape(reading, (len(reading), 1))

        ans = np.zeros(10)
        ans[data[i][0]] = 1 #answer array for checking our results 

        z1, z2 = forwardprop(w1, b1, w2, b2, input)
        w1, b1, w2, b2 = backprop(w1, w2, b1, b2, z1, z2, input, ans, 0.01) 

        correct = accuracy(z2, ans, correct)
        precent = correct / (i + 1) * 100
        print(f"Accuracy is {precent:0.2f}%")











if __name__ == "__main__":
    main()
