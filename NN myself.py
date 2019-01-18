class NeuralNetwork():
    def __init__(self):
        np.random.seed(0)
        self.weights=np.random.random((3,1))*2-1
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def __sigmoid_derivative(self,x):
        return x*(1-x)
    def activate(self,input):
        return self.__sigmoid(np.dot(input,weights))
    def train(self,train_input,train_output,number_of_iterations):
        for i in range(number_of_iterations):
            output=self.activate(train_input)
            self.weights+=np.dot(train_input.T,(train_output-output)*self.__sigmoid_derivative(output))
    

if __name__=='__main__':
    nn=NeuralNetwork()
    print('The weights at initial period:',nn.weights)
    train_inputs =np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs =np.array([[0, 1, 1, 0]]).T
    nn.train(train_inputs,train_outputs,100000)
    print('\n\n',nn.weights)
    print(nn.activate(np.array([1,0,1])))
