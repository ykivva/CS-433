import numpy as np


class NNModel():
    """ Class for simple Neural Networks which can solve classification and regression problems
    
        Args:
            parameters (dict): stores weight, bias, and name of activation function corresponding
                to relative model's layer;
                keys of the dictionary of the following kind:
                    W{int} - weights to compute the {int}th layer
                    b{int} - bias parameters to compute the {int}th layer
                    a{int} - activation function called for {int}th layer
            values (dict): for each layer stores their value before and after applying activation function
            features_out (int): last dimentsion of the last layer of the model
            num_layers (int): number of layers in the model
            loss (string): name of the loss which will be applied to the output of the model
    """

    def __init__(self,  features_in):
        """Initialize instance of NNModel

            Args:
                features_in (int): number of input features
                loss (string): loss which will be used for the output of model;
                    Possible values - l2, cross_entropy, logistic_reg
        """
        self.parameters = {}
        self.grads = {}
        self.values = {}
        self.features_out = features_in
        self.num_layers = 0

    def add_layer(self, units, activation=None):
        """Add layer to the end of the model

            Args:
                units (int): size of the layer (number of unit/neurons in the layer)
                activation (string): activation to apply for the units in the layer, if None no activations is applied
        """
        self.num_layers += 1

        w = np.random.randn(units, self.features_out)
        b = np.zeros(units)

        self.grads[f'mW{self.num_layers}'] = np.zeros((units, self.features_out))
        self.grads[f'mb{self.num_layers}'] = np.zeros(units)
        self.grads[f'W{self.num_layers}'] = np.zeros((units, self.features_out))
        self.grads[f'b{self.num_layers}'] = np.zeros(units)

        self.features_out = units

        self.parameters[f'W{self.num_layers}'] = w
        self.parameters[f'b{self.num_layers}'] = b
        self.parameters[f'a{self.num_layers}'] = activation

    def zero_grad(self):
        for i in range(1, self.num_layers+1):
            self.grads[f'mW{self.num_layers}'] -= self.grads[f'mW{self.num_layers}']
            self.grads[f'mb{self.num_layers}'] -= self.grads[f'mb{self.num_layers}']
            self.grads[f'W{self.num_layers}'] -= self.grads[f'W{self.num_layers}']
            self.grads[f'b{self.num_layers}'] -= self.grads[f'b{self.num_layers}']



    def predict(self, x):
        """Compute output of the model

            Args:
                x (nd.array): input to the model

            Returns:
                output (nd.array): output of the model
        """
        output = self._forward(x)
        return output
        
    
    def train(self, x, y, lr=0.1, lambda_=0, batch_size=None, epochs=1, verbose=0, loss_fun='l2', momentum=0):
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
        self.zero_grad()

        loss = 1e10
        indx = np.arange(x.shape[0])
        if batch_size == None:
            batch_size = x.shape[0]
        
        print('Training started')
        decay = 1
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
            decay *= 3
            while batch_end <= x_shuffled.shape[0]:
                x_batch = x_shuffled[batch_start:batch_end, ...]
                y_batch = y_shuffled[batch_start:batch_end, ...]

                output = self._forward(x_batch)

                get_loss_grad =  getattr(NNModel, f'_get_{loss_fun}_grad')

                loss_grad, loss = get_loss_grad(output, y_batch)
                sum_loss += loss

                self._backward(loss_grad, lambda_)
                self._optimize(lr=lr/decay, momentum=0)

                batch_start += batch_size
                batch_end += batch_size
            
            if verbose==1:
                bar  = (epoch*20//epochs)*"#" + " " * (20 - (epoch*20//epochs))
                print(f'\r>Epoch #{epoch}:\t[{bar}]; Loss: {sum_loss / num_batches}', end='')
        
        print('\nTraining ended\n')
        
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
                act_method = getattr(NNModel, activation)
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
                    NNModel,
                    f'_get_{activation}_grad')(self.values[f'h{num_layer}']) * self.grads[f'a{num_layer}']
            
            self.grads[f'W{num_layer}'] = self.grads[f'h{num_layer}'].T @ self.values[f'a{num_layer-1}']
            self.grads[f'b{num_layer}'] = np.sum(self.grads[f'h{num_layer}'], axis=0)
            
            self.grads[f'W{num_layer}'] += 2 * lambda_ * self.parameters[f'W{num_layer}']

            self.grads[f'a{num_layer-1}'] = self.grads[f'h{num_layer}'] @ self.parameters[f'W{num_layer}']

    def _optimize(self, lr, momentum=0):
        """Applies gradients to update the parameters

            Args: 
                lr (int): learning rate
        """
        for num_layer in range(1, self.num_layers+1):
            self.grads[f'mW{num_layer}'] = momentum * self.grads[f'mW{num_layer}'] - lr * self.grads[f'W{num_layer}']
            self.grads[f'mb{num_layer}'] = momentum * self.grads[f'mb{num_layer}'] - lr * self.grads[f'b{num_layer}']

            self.parameters[f'W{num_layer}'] += self.grads[f'mW{num_layer}']
            self.parameters[f'b{num_layer}'] += self.grads[f'mb{num_layer}']

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
        act = NNModel.softmax(output)
        return act * (1 - act)

    @staticmethod
    def l2(pred, target):
        """Computes l2

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed l2
        """
        loss = np.sum((pred - target)**2) / (2*pred.size)
        return loss
    
    @staticmethod
    def _get_l2_grad(pred, target):
        """Computes gradients of l2 with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed gradients and loss
        """
        return (pred - target) / pred.size, NNModel.l2(pred, target)
        
    @staticmethod
    def cross_entropy(pred, target):
        """Computes Negative Log Likelihood

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value (true value in format of one-hot vector)
            
            Returns:
                Computed cross_entropy
        """
        loss = np.sum(target * (-np.log(pred))) / len(target)
        return loss
    
    @staticmethod
    def _get_cross_entropy_grad(pred, target, epsilon=1e-9):
        """Computes gradients of negative log likelihood with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value (true value in format of one-hot vector)
                epsilon (float): value added to the denominator to prevent division by 0 
            
            Returns:
                Computed gradients and loss
        """
        return -target * 1 / ((pred + epsilon) * len(target)), NNModel.cross_entropy(pred, target)
    
    @staticmethod
    def sigmoid(output, lower_bound=-30, upper_bound=30):
        """Computes sigmoid function

            Args:
                output (nd.array): input array
            
            Returns:
                Computed sigmoid function
        """
        output_bounded = output.copy()
        output_bounded[output>upper_bound] = upper_bound
        output_bounded[output<lower_bound] = lower_bound
        return 1. / (1 + np.exp(-output_bounded))

    @staticmethod
    def _get_sigmoid_grad(output):
        """Computes gradients of sigmoid function with respect to the input

            Args:
                output (nd.array): array to which applies softmax function
            
            Returns:
                Computed gradients
        """
        act = NNModel.sigmoid(output)
        return act * (1 - act)

    @staticmethod
    def logistic_reg(pred, target):
        """Computes logistic_reg loss

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
            
            Returns:
                Computed logistic_reg loss
        """
        pred_sq = np.squeeze(pred)
        target_sq = np.squeeze(target)
        loss = np.sum(-(np.dot(target_sq, np.log(pred_sq)) + np.dot((1-target_sq), np.log(1-pred_sq)))) / len(target_sq)
        return loss
    
    @staticmethod
    def _get_logistic_reg_grad(pred, target, epsilon=1e-9):
        """Computes gradients of Binary Cross Entropy loss with respect to the pred

            Args:
                pred (nd.array): prediction of the model
                target (nd.array): true value
                epsilon (float): value added to the denominator to prevent division by 0 
            
            Returns:
                Computed gradients and loss
        """
        grad = -target / (pred + epsilon) + (1 - target) / (1 - pred + epsilon)
        return grad / len(target), NNModel.logistic_reg(pred, target)
    