import numpy as np


class NN_Model():
    """ Class for simple Neural Networks which can solve classification and regression problems
    
        Args:
            parameters (dict): stores weight, bias, and name of activation function corresponding
                to relative model's layer;
                keys of the dictionary of the following kind:
                    W{int} - weights to compute the {int}th layer
                    b{int} - bias parameters to compute the {int}th layer
                    a{int} - activation function called for {int}th layer
            values (dict): for each layer stores their value before and after applying activation function
            output_size (int): last dimentsion of the last layer of the model
            num_layers (int): number of layers in the model
            loss (string): name of the loss which will be applied to the output of the model
    """

    def __init__(self,  num_features, loss):
        """Initialize instance of NN_model

            Args:
                num_features (int): number of input features
                loss (string): loss which will be used for the output of model;
                    Possible values - L2Loss, NLLoss, BinaryCrossEntropy
        """
        self.parameters = {}
        self.grads = {}
        self.values = {}
        self.output_size = num_features
        self.num_layers = 0
        self.loss = loss

    def add_layer(self, units, activation=None):
        """Add layer to the end of the model

            Args:
                units (int): size of the layer (number of unit/neurons in the layer)
                activation (string): activation to apply for the units in the layer, if None no activations is applied
        """
        self.num_layers += 1

        w = np.random.randn(units, self.output_size)
        b = np.zeros(units)
        self.output_size = units

        self.parameters[f'W{self.num_layers}'] = w
        self.parameters[f'b{self.num_layers}'] = b
        self.parameters[f'a{self.num_layers}'] = activation

    def predict(self, x):
        """Compute output of the model

            Args:
                x (nd.array): input to the model

            Returns:
                output (nd.array): output of the model
        """
        output = self._forward(x)
        return output
        
    
    def train(self, x, y, lr=0.1, lambda_=0, batch_size=None, epochs=1, verbose=0):
        """Train model

            Args:
                x (nd.array): input to the model of the shape (N, ...);
                     N - number of examples in traing dataset;
                y (nd.array): true output of the model with shape (N, ...)
                lr (int): learning rate for optimization method
                lambda_ (int): regularization coeff
                batch_size (int): size of batch feeded to the model
                epochs (int): number of training epochs
                verbose (int): if equals to 1, it will output loss for every 10 epochs
        """
        loss = 1e10
        indx = np.arange(x.shape[0])
        if batch_size == None:
            batch_size = x.shape[0]
        
        print('Training started')
        for epoch in range(1, epochs+1):
            permut = np.random.permutation(indx)
            x_shuffled = x[permut, ...]
            y_shuffled = y[permut, ...]
            if len(y_shuffled.shape) == 1:
                y_shuffled = y_shuffled.reshape(len(y_shuffled), -1)

            batch_start = 0
            batch_end = batch_size
            num_batches = x.shape[0] // batch_size
            sum_loss = 0
            while batch_end <= x.shape[0]:
                x_batch = x_shuffled[batch_start:batch_end, :]
                y_batch = y_shuffled[batch_start:batch_end, :]

                output = self._forward(x_batch)

                get_loss_grad =  getattr(self, f'_get_{self.loss}_grad')

                loss_grad, loss = get_loss_grad(output, y_batch)
                sum_loss += loss

                self._backward(loss_grad, lambda_)
                self._optimize(lr=lr)

                batch_start += batch_size
                batch_end += batch_size
            
            if verbose==1:
                    print(f'Epoch #{epoch}: {sum_loss / num_batches}')
        
        print('Training ended')
        
    def _forward(self, x):
        """Forward pass of the model. During forward pass it stores values of layers

            Args:
                x (nd.array): batch of examples which model forwards
            
            Returns:
                Output of the model
        """
        self.values['h0'] = x.copy()
        self.values['a0'] = x.copy()
        for num_layer in range(1, self.num_layers+1):
            self.values[f'h{num_layer}'] = self.values[f'a{num_layer-1}'] @ self.parameters[f'W{num_layer}'].T
            self.values[f'h{num_layer}'] += self.parameters[f'b{num_layer}']
            activation = self.parameters[f'a{num_layer}']
            if activation != None:
                act_method = getattr(NN_Model, activation)
                self.values[f'a{num_layer}'] = act_method(self.values[f'h{num_layer}'])
            else:
                self.values[f'a{num_layer}'] = self.values[f'h{num_layer}'].copy()
        
        return self.values[f'a{self.num_layers}']

    def _backward(self, loss_grad, lambda_=0):
        """Computes and stores gradients

            Args:
                loss_grad (nd.array): gradients of loss with respect to the model output
                lambda_ (int): regularization coeff
        """
        self.grads[f'a{self.num_layers}'] = loss_grad
        
        for num_layer in range(self.num_layers, 0, -1):
            activation = self.parameters[f'a{num_layer}']
            if activation == None:
                self.grads[f'h{num_layer}'] = self.grads[f'a{num_layer}'].copy()
            else:        
                self.grads[f'h{num_layer}'] = getattr(
                    self,
                    f'_get_{activation}_grad')(self.values[f'h{num_layer}']) * self.grads[f'a{num_layer}']
            
            self.grads[f'W{num_layer}'] = self.grads[f'h{num_layer}'].T @ self.values[f'a{num_layer-1}']
            self.grads[f'b{num_layer}'] = np.sum(self.grads[f'h{num_layer}'], axis=0)
            
            self.grads[f'W{num_layer}'] += 2 * lambda_ * self.parameters[f'W{num_layer}']

            self.grads[f'a{num_layer-1}'] = self.grads[f'h{num_layer}'] @ self.parameters[f'W{num_layer}']

    def _optimize(self, lr):
        """Applies gradients to update the parameters

            Args: 
                lr (int): learning rate
        """
        for num_layer in range(self.num_layers, 0, -1):
            self.parameters[f'W{num_layer}'] -= lr * self.grads[f'W{num_layer}']
            self.parameters[f'b{num_layer}'] -= lr * self.grads[f'b{num_layer}']

    @staticmethod
    def softmax(output, axis=1):
        """Computes softmax along the specific axis

            Args:
                output (nd.array): array for which will be applied softmax function
                axis (int): axis a long which will be computed softmax
            
            Returns:
                Computed result
        """
        max_deg = np.max(output, axis=axis, keepdims=True)
        output_shift = output - max_deg
        return  np.exp(output_shift) / np.sum(output_shift, axis=1, keepdims=True)
    
    @staticmethod
    def _get_softmax_grad(output):
        """Computes gradients of softmax function with respect to the input

            Args:
                output (nd.array): array to which applies softmax function
            
            Returns:
                Computed gradients
        """
        act = NN_Model.softmax(output)
        return act * (1 - act)

    @staticmethod
    def L2Loss(pred, target):
        """Computes L2Loss

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed L2Loss
        """
        loss = np.sum((pred - target)**2) / (2*pred.size)
        return loss
    
    @staticmethod
    def _get_L2Loss_grad(pred, target):
        """Computes gradients of L2Loss with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed gradients and loss
        """
        return (pred - target) / pred.size, NN_Model.L2Loss(pred, target)
        
    @staticmethod
    def NLLoss(pred, target):
        """Computes Negative Loss Likelihood

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value (true value in format of one-hot vector)
            
            Returns:
                Computed NLLoss
        """
        loss = np.sum(target * (-np.log(pred))) / len(target)
        return loss
    
    @staticmethod
    def _get_NLLoss_grad(pred, target, epsilon=1e-15):
        """Computes gradients of L2Loss with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value (true value in format of one-hot vector)
                epsilon (float): value added to the denominator to prevent division by 0 
            
            Returns:
                Computed gradients and loss
        """
        return -target * 1 / ((pred + epsilon) * len(target)), NN_Model.NLLoss(pred, target)
    
    @staticmethod
    def sigmoid(output):
        """Computes sigmoid function

            Args:
                output (nd.array): input array
            
            Returns:
                Computed sigmoid function
        """
        return 1. / (1 + np.exp(-output))

    @staticmethod
    def _get_sigmoid_grad(output):
        """Computes gradients of sigmoid function with respect to the input

            Args:
                output (nd.array): array to which applies softmax function
            
            Returns:
                Computed gradients
        """
        act = NN_Model.sigmoid(output)
        return act * (1 - act)

    @staticmethod
    def BinaryCrossEntropy(pred, target):
        """Computes BinaryCrossEntropy loss

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed BinaryCrossEntropy loss
        """
        target_mat = target.reshape(target.shape[0], -1)
        loss = np.sum(-(target_mat @ np.log(pred).T + (1-target_mat) @ np.log(1-pred).T)) / len(target)
        return loss
    
    @staticmethod
    def _get_BinaryCrossEntropy_grad(pred, target, epsilon=1e-15):
        """Computes gradients of Binary Cross Entropy loss with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
                epsilon (float): value added to the denominator to prevent division by 0 
            
            Returns:
                Computed gradients and loss
        """
        target_mat = target.reshape(target.shape[0], -1)
        grad = -target_mat / pred + (1 - target_mat) / (1 - pred)
        return grad / len(target), NN_Model.BinaryCrossEntropy(pred, target)