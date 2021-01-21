################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1.0/(1.0+np.exp(-x + 1e-6))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(0.0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        sig = self.sigmoid(self.x)
        return sig * (1 - sig)

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        # """
        return 1 - self.tanh(self.x) ** 2


    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        dx = np.zeros(self.x.shape)
        dx[self.x > 0] = 1
        return dx


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        # (10000*1024) * (1024x128) = (10000*128)
        # (1*128)
        self.bias_velocity = np.zeros((1, out_units))
        self.v_w = np.zeros((in_units, out_units))
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units, out_units)  # You can experiment with initialization.

        self.v_b = np.zeros((1, out_units))                                    
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.prev_dw = np.zeros(self.w.shape)
        self.prev_db = np.zeros(self.b.shape)

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        #12000 x 1024  * 1024 x 128 = 12000 x 128
        self.x = x
        self.a = np.matmul(x, self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # (1024x128) * 
        # delta (128, 10)
        # w (128, 10)
        # x (128, 128)
        # print("delta", delta.shape)
        # print("w", self.w.shape)
        # print("x", self.x.shape)
        # w_ij = w_ij + alpha delta_j z_i
        # i to j

        self.prev_dw = self.d_w
        self.prev_db = self.d_b

        self.d_x = np.matmul(delta, self.w.T)
        self.d_w = np.matmul(self.x.T, delta) / len(delta)
        self.d_b = np.mean(delta, axis=0)

        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.momentum_gamma = config['momentum_gamma']
        self.L2_penalty = config['L2_penalty']

        # layers = [Layer(1024, 128), Activation(tanh), Layer(128, 10)]]
            
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))
        



    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """        
        #            input->hidden                       hidden->output
        #layers = [Layer(1024, 128), Activation(tanh), Layer(128, 10)]]

        self.x = x
        self.targets = targets

        # CHECKING NUMERICAL APPROXIMATION

        # output = x

        # epsilon = 1e-2


        # print("i = 0")
        # print("weight change 1")
        # self.layers[0].w[0][0] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[0].w[0][0] -= 2 * epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[0].w[0][0] += epsilon

        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print("weight change 2")
        # self.layers[0].w[0][1] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[0].w[0][1] -= 2 * epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[0].w[0][1] += epsilon
        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # print("bias")
        # self.layers[0].b[0][0] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[0].b[0][0] -= 2 * epsilon


        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[0].b[0][0] += epsilon




        # print("i = 2")
        # print("weight change 1")
        # self.layers[2].w[0][0] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[2].w[0][0] -= 2 * epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[2].w[0][0] += epsilon


        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print("weight change 2")
        # self.layers[2].w[0][1] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[2].w[0][1] -= 2 * epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[2].w[0][1] += epsilon
        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # print("bias")
        # self.layers[2].b[0][0] += epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # add_loss = self.loss(self.y, targets)
        # print("add", add_loss)

        # self.layers[2].b[0][0] -= 2 * epsilon

        # output = x
        # for i in range(len(self.layers)):
        #     output = self.layers[i].forward(output)

        # self.y = self.softmax(output)

        # # loss = self.softmax_loss(self.y, targets)
        # sub_loss = self.loss(self.y, targets)
        # print("sub", sub_loss)

        # print("subtracted", (add_loss - sub_loss))
        # print("divided", ((add_loss - sub_loss)/(2*epsilon)))

        # self.layers[2].b[0][0] += epsilon

        output = x
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)

        self.y = self.softmax(output)

        loss = self.loss(self.y, targets)



        return loss

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
                                                         
        #            input->hidden           i            hidden->output
        # layers = [Layer(1024, 128),   Activation(tanh),  Layer(128, 10)]]

        delta = self.targets - self.y
        for i in range(len(self.layers) - 1, -1, -1):
            delta = self.layers[i].backward(delta)

            if isinstance(self.layers[i], Layer):
                
                if self.momentum:
                    self.layers[i].v_w = self.layers[i].d_w + self.momentum_gamma*self.layers[i].v_w
                    #L1 regularization
                    self.layers[i].v_w -= (self.L2_penalty * self.layers[i].w)
                    #L1 regularization
                    # for r in range(self.layers[i].v_w.shape[0]):
                    #     for c in range(self.layers[i].v_w.shape[1]):
                    #         if self.layers[i].w[r][c] > 0.0:
                    #             self.layers[i].v_w[r][c] -= self.L2_penalty
                    #         elif self.layers[i].w[r][c] < 0.0:
                    #             self.layers[i].v_w[r][c] += self.L2_penalty
                    self.layers[i].w += self.learning_rate * self.layers[i].v_w
                
                    self.layers[i].v_b = self.layers[i].d_b + self.momentum_gamma*self.layers[i].v_b
                    #L2 regularization
                    self.layers[i].v_b -= (self.L2_penalty * self.layers[i].b)
                    #L1 regularization
                    # for j in range(self.layers[i].v_b.shape[1]):
                    #     if self.layers[i].b[0][j] > 0.0:
                    #         self.layers[i].v_b[0][j] -= self.L2_penalty
                    #     elif self.layers[i].b[0][j] < 0.0:
                    #         self.layers[i].v_b[0][j] += self.L2_penalty
                    
                    self.layers[i].b += self.learning_rate * self.layers[i].v_b
                else:
                    self.layers[i].w += self.learning_rate * self.layers[i].d_w
                    self.layers[i].b += self.learning_rate * self.layers[i].d_b

        return delta

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        ps = np.empty(x.shape)
        for i in range(x.shape[0]):
            ps[i] = np.exp(x[i] - np.max(x[i]))
            ps[i] /= ps[i].sum()
        return ps

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        # Logits = the predicted value out of the last layer
        sum = np.sum(np.log(logits + 1e-6) * (targets))
        loss = -sum / (len(targets))
        # Regularization L2 
        regularization_loss = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                # regularization_loss += np.sum(np.square(layer.w))
                regularization_loss += np.sum(np.abs(layer.w))
        regularization_loss *= (self.L2_penalty/(2*len(targets)))
        loss += regularization_loss
        return loss

#with regulariation
# Test Loss 1.0303845390241253
# Test Accuracy 0.7318300553165334


# Test Loss 0.9725193107653298
# Test Accuracy 0.7377842655193608

# Regularization 1e^3
# Test Loss 0.7592699236229427
# Test Accuracy 0.7780039950829748

# Regularization 1e^6
# Test Loss 1.016977917683449
# Test Accuracy 0.7315611555009219

# L1 Regularization
# Test Loss 0.7994522117549617
# Test Accuracy 0.7710125998770744