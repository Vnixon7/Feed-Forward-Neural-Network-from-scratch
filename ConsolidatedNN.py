import numpy as np
import matplotlib.pyplot as plt


def get_activation_List():
    print('All activation functions\n__________________\nsigmoid\ntanh\nsoftmax\nlinear\nswish')


class Neural_Network(object):
    def __init__(self, model_name, data, labels, step, activation_function):
        np.random.seed(42)
        self.model_name = model_name
        self.data = data
        self.labels = labels.reshape(len(labels[0]), 1)
        self.weights = np.random.rand(len(data[0]), 1)
        self.bias = np.random.rand(1)
        self.step = step
        self.activation_function = activation_function
        self.slope = 0
        self.cost = 0

    def activation(self, x):

        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(- x))
        if self.activation_function == 'tanh':
            return np.tanh(x)

        if self.activation_function == 'softmax':
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        if self.activation_function == 'linear':
            return self.step * x

        if self.activation_function == 'swish':
            return x / (1 - np.exp(-x))

    def activation_der(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def train(self, iterations, cost_visual):
        for i in range(iterations):

            inputs = self.data
            # Forward pass step 1: initial config of weights and bias
            # y = input * weight + bias
            IWB = np.dot(inputs, self.weights) + self.bias

            # Forward pass step 2: applying activation function to sum
            output = self.activation(IWB)

            # Backward pass step 1: Cost function
            self.cost = output - self.labels
            if cost_visual:
                print(f'______ITERATION_{i}______')
                print('Loss: ', np.round(self.cost.sum(), 4))

            # Backwards pass step 2: Derivatives
            der_output = self.activation_der(output)
            delta = self.cost * der_output
            inputs = self.data.T
            self.slope = self.weights - self.step * np.dot(inputs, delta)
            self.weights -= self.step * np.dot(inputs, delta)
            for i in delta:
                self.bias -= self.step * i

    def predict(self, plot):
        acc = 0
        print('________', self.model_name, '_________')
        # forward pass to get prediction
        result = self.activation(np.dot(self.data, self.weights) + self.bias)

        for i, point in enumerate(result):
            print(f'Result {i + 1} Prediction: ', np.round(point, 3), ' | ', f'Result {i + 1} Actual: ', self.labels[i])
            if point > 0.5 and self.labels[i] == 1 or point < 0.5 and self.labels[i] == 0:
                acc += 1
        print('Accuracy: ', acc / len(result))
        print('Cost: ', '\n',np.round(self.cost,3))

        if plot:
            plt.scatter(result, self.labels)
            plt.plot(self.slope)
            plt.show()


if __name__ == '__main__':
    get_activation_List()
    data = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    labels = np.array([[1, 0, 0, 1, 1]])
    nn_sig = Neural_Network(model_name='Sigmoid', data=data, labels=labels, step=0.5, activation_function='sigmoid')
    nn_tanh = Neural_Network(model_name='tanh', data=data, labels=labels, step=0.5, activation_function='tanh')
    nn_softmax = Neural_Network(model_name='softmax', data=data, labels=labels, step=0.5, activation_function='softmax')
    nn_linear = Neural_Network(model_name='linear', data=data, labels=labels, step=0.5, activation_function='linear')
    nn_sig.train(iterations=100, cost_visual=False)
    nn_sig.predict(plot=False)
    nn_tanh.train(iterations=100, cost_visual=False)
    nn_tanh.predict(plot=False)
    nn_linear.train(iterations=100, cost_visual=False)
    nn_linear.predict(plot=False)
    nn_softmax.train(iterations=100, cost_visual=False)
    nn_softmax.predict(plot=False)


