import numpy as np


class NN_Model():
    def __init__(self, input_size, loss):
        self.parameters = {}
        self.grads = {}
        self.values = {}
        self.input_size = input_size
        self.output_size = None
        self.num_layers = 0
        self.loss = loss

    def add_layer(self, units, activation=None):
        self.num_layers += 1

        w = np.random.randn(units, self.output_size)
        b = np.zero(units)
        self.output_size = units

        self.parameters[f'W{self.num_layers}'] = w
        self.parameters[f'b{self.num_layers}'] = b
        self.parameters[f'a{num_layers}'] = activation

    def predict(self, x):
        output = self._forward(x)
        return output
        
    
    def train(self, x, y, lr=0.1, lambda_=0, batch_size=None, epochs=1):
        indx = np.arange(x.shape[1])
        if batch_size != None:
            for epoch in range(epochs):
                permut = np.random.permutation(indx)
                x_shuffled = x[permut]
                y_shuffled = y[permut]

                batch_start = 0
                batch_end = batch_size
                while batch_end <= x.shape[1]:
                    x_batch = x_shuffled[batch_start:batch_end, :]
                    y_batch = y_shuffled[batch_start:batch_end, :]

                    output = self._forward(x_batch)

                    get_loss_grad =  getattr(self, f'_get_{self.loss}_grad')
                    self.grads[f'a{self.num_layers}'] = get_loss_grad(output, y)

                    self._backward(lambda_)
                    self._optimize(lr=lr)

                    batch_start += batch_size
                    batch_end += batch_size
        
    def _forward(self, x):
        self.values['h0'] = x.copy()
        self.values['a0'] = x.copy()
        for num_layer in range(1, self.num_layers+1):
            self.values[f'h{num_layer}'] = self.values[f'h{num_layer-1}'] @ self.parameters[f'W{num_layer}'].T
            self.values[f'h{num_layer}'] += self.parameters[f'b{num_layer}']
            activation = self.parameters[f'a{num_layer}']
            if activation != None:
                act_method = getattr(NN_Model, activation)
                self.values[f'a{num_layer}'] = act_method(self.values[f'h{num_layer}'])
            else:
                self.values[f'a{num_layer}'] = self.values[f'h{num_layer}'].copy()
        
        return self.values[f'a{self.num_layers}']

    def _backward(self, loss_grad, lambda_=0):
        self.grads[f'a{self.num_layers}'] = loss_grad
        
        for num_layer in range(self.num_layers, 0, -1):
            activation = self.parameters[f'a{self.num_layer}']
            self.grads[f'h{num_layer}'] = getattr(
                self,
                f'_get_{activation}_grad')(self.values[f'a{num_layer}']) * self.grads[f'a{num_layer}']
            
            self.grads[f'W{num_layer}'] = self.grads[f'h{num_layer}'] @ self.values[f'h{num_layer-1}']
            self.grads[f'b{num_layer}'] = np.sum(self.grads[f'h{num_layer}'], axis=0)
            
            self.grads[f'W{num_layer}'] += 2 * lambda_ * self.parameters[f'W{num_layer}']

            self.grads[f'a{num_layer-1}'] = self.grads[f'h{num_layer}'] @ self.parameters[f'W{num_layer}']

    def _optimize(self, lr):
        for num_layer in range(self.num_layers, 0, -1):
            self.parameters[f'W{num_layer}'] -= lr * self.grads[f'W{num_layer}']
            self.parameters[f'b{num_layer}'] -= lr * self.grads[f'W{num_layer}']

    @staticmethod
    def softmax(cls, output, axis=1):
        max_deg = np.max(output, axis=axis, keepdims=True)
        output_shift = output - max_deg
        return  np.exp(output_shift) / np.sum(output_shift, axis=1, keepdims=True)
    
    @staticmethod
    def _get_softmax_grad(output):
        return output * (1 - output)

    @staticmethod
    def L2Loss(pred, target):
        loss = np.sum((pred - target)**2) / (2*pred.size)
        return loss
    
    @staticmethod
    def _get_L2Loss_grad(pred, target):
        return (pred - target) / pred.size
        
    @staticmethod
    def NLLoss(pred, target):
        loss = np.sum(target * (-np.log(pred))) / len(target)
        return loss
    
    @staticmethod
    def _get_NLLoss(pred, target, epsilon=1e-15):
        return -target * 1 / ((pred + epsilon) * len(target))
    
    @staticmethod
    def sigmoid(output):
        return 1. / (1 + np.exp(-output))

    @staticmethod
    def _get_sigmoid_grad(output):
        return output * (1 - output)

    @staticmethod
    def BinaryCrossEntropy(pred, target):
        loss = np.sum(-(np.dot(target, np.log(pred)) + np.dot(1-target, np.log(1-pred)))) / len(target)
        return loss
    
    @staticmethod
    def _get_BinaryCrossEntropy_grad(pred, target, epsilon=1e-15):
        target_mat = target.reshape(target.shape[0], -1)
        grad = -target_mat / pred + (1 - target_mat) / (1 - pred)
        return grad / len(target)