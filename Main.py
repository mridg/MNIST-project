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
    z1 = ReLU(w1 * A + b1)
    print(z1.shape)

    #z2 = w2 * z1 + b2
    #z2 = sigmoid(z2)

    return z1 #, z2

def backwardprop(w2, i, z1, z2, key):
    m = len(z2)
    dz2 = z2 - key.T
    print(dz2.shape)
    dw2 =  1 / m * dz2.dot(z1.T)
    print(dw2.shape)
    db2 = 1 / m * np.sum(dz2, axis=1)
    print(db2.shape)

    dz1 = ((w2).dot(dz2)).dot(sigmoidprime(z1).T) #this shape is supposed to be 10x42000
    print(dz1.shape)
    dw1 = 1 #1 / m * dz1.dot(i.T)
    #print(dw1.shape)
    db1 = 1 #1 / m & np.sum(dz1, axis=0)
    #print(db1.shape)

    return dw1, db1, dw2, db2

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
    data = data.T #transpose to be in the correct form,so now the first row is all the answers
    #print(data)

    print(data[0][1].shape)
    #for i in range(len(data)):
        #forwardprop(data[i])
        # z1, z2 = forwardprop(i)
        # print(z2)




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
