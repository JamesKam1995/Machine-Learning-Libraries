"""
Implementation of a FC network using python standard libraries

"""
import numpy as np

class FCLayer():
    def __init__(self, input_size, output_size, activation):
        """
        Args:
            input_size (int): Input shape of the layer 
            output_size (int): Output of the layer
            activation (str): activation function 
        """
        self.weight = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation

        #Adam optim
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_bias = np.zeros((1, output_size))
        self.v_bias = np.zeros((1, output_size))

        #Define Hyperparamter
        self.beta1 = 0.9 
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward(self, X):
        self.x = X
        z = np.dot(self.x, self.weight) + self.bias

        #apply activation function
        if self.activation == 'relu':
            self.output = np.maximum(0, z)
        
        elif self.activation == 'softmax':
            exp_values = np.exp(z - np.max(z, axis= -1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        else:
            print(f"the activation function does not exist!")

        return self.output
    
    def backward(self, d_values, learning_rate, t):
        if self.activation == 'softmax':
            for i, gradient in enumerate(d_values):
                if len(gradient.shape) == 1:
                    gradient = gradient.reshape(1, -1)
                jacobian_matirx = np.diagflat(gradient) - np.dot(gradient, gradient.T)
                d_values[i] = np.dot(jacobian_matirx, self.output[i])

        #derivative of ReLU function
        elif self.activation == 'relu':
            d_values = d_values * (self.output > 0)

        #calculate the derviative with respect to the weight and bias
        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis = 0, keepdims=True)

        #clip the derviative
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        #calculate the gradient with respect to the input
        d_inputs = np.dot(d_values, self.weight.T)

        #Update the weight and biases
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_biases

        #Update weights using m and v values
        m_weight = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        v_weight = self.beta1 * self.v_weights + (1 - self.beta1) * (d_weights**2)
        m_hat_weights = m_weight / (1 - self.beta1 ** t)
        v_hat_weights = v_weight / (1 - self.beta2 ** t)
        self.weight -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        #Update biases using m and v values
        m_biases = self.beta1 * self.m_bias + (1 - self.beta1) * d_biases
        v_biases = self.beta1 * self.v_bias + (1 - self.beta1) * (d_biases**2)
        m_hat_biases = m_biases / (1 - self.beta1 ** t)
        v_hat_biases = v_biases / (1 - self.beta2 ** t)
        self.bias -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        return d_inputs