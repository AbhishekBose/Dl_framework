import numpy as np

class Softmax:
    # A standard fully connected layer with softmax activation

    def __init__(self,input_len, nodes):
        #dividing the weight by input len to reduce variance of our initial values
        self.weights = np.random.randn(input_len,nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self,input):
        '''
        Performs forward pass of the softmax layer using the given input.
        Returns 1d numpy array with class probabilities
        '''
        input = input.flatten()
        input_len, nodes = self.weights.shape
        totals = np.dot(input,self.weights) + self.biases
        exp = np.exp(totals)
        return exp / np.sum(exp, axis = 0)

if __name__ == "__main__":
    img = np.random.randn(4,4,1)
    softmax = Softmax(4*4*1,10)
    print(softmax.forward(img))
    

