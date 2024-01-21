import numpy as np
import pandas as pd
import matplotlib as math

# def forwardprop(weight, bias, input):
#     z = weight.dot(input) + bias # issue with the dot product
#     z = ReLU(z)
#     return z
def setup():
    # first layer
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    # second layer
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)

    return w1, b1, w2, b2

def forwardprop(A):
    w1, b1, w2, b2 = setup()
    z1 = ReLU(np.matmul(w1, A) + b1)
    z2 = sigmoid(np.matmul(w2, z1) + b2)

    return z1, z2

def backprop(w1, w2, b1, b2, z1, z2, input, ans, learningrate):
    dw2 = z1 * sigmoidprime(z2) * 2 * (z2 - ans)
    db2 = 1 * sigmoidprime(z2) * 2 * (z2 - ans)

    w2f = w2 - learningrate * dw2
    b2f = b2 - learningrate * db2

    dw1 = input * sigmoidprime(z1) * 2 * (w2 - w2f) #not sure check math on this 
    db1 = 1 * sigmoidprime(z1) * 2 * (b2 - b2f) #check math not sure 

    w1f = w1 - learningrate * dw1
    b1f = b1 - learningrate * db1

def ReLU(input):
    return np.maximum(0, input)

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidprime(input):
    return sigmoid(input) * (1 - sigmoid(input))

def checkanswer():
    pass

def main():
    """Write your mainline logic below this line (then delete this line)."""
    data = pd.read_csv("train.csv")
    data = np.array(data) #turns any array like object into an array! how wonderful! I want a greasy almond sandwich
    #data = data.T #transpose to be in the correct form,so now the first row is all the answers
    #print(data)

    #print(data.shape)

    for i in range(10):
        input = np.delete(data[i].T , 0,  axis=0) #initial greyscale inputs

        ans = np.zeros(10)
        ans[data[i][0]] = 1 #answer array for checking our results 

        #print(input.size)
        print(forwardprop(input))



        

    # for i in range(len(data)):
    #     forwardprop(data[i])
    #     z1, z2 = forwardprop(i)
    #     print(z2)




    # ans = data[0] #can't use this form, we want this to be in binary so we can appropriately edit w and b

    # a = np.empty([len(ans), 10])

    # for i in range(len(ans)):
    #     a[i, ans[i]] = 1

    # w1, b1, w2, b2 = setup()

    # learningrate = 0.5

    # i = np.delete(data, 0, axis=0)

    # for j in range(100):
    #     z1, z2 = forwardprop(w1, b1, w2, b2, i)
    #     dw1, db1, dw2, db2 = backwardprop(w2, i, z1, z2, a)

    #     w1 = w1 - learningrate * dw1
    #     b1 = b1 - learningrate * db1

    #     w2 = w2 - learningrate * dw2
    #     b2 = b2 - learningrate * db2

    # print(z2)









if __name__ == "__main__":
    main()
