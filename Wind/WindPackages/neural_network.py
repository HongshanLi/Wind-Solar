import numpy as np
import math
from six.moves import xrange

class NeuralNetwork():
    """ Neural Network Class from Hongshan."""
    
    def __init__(self, shape=list()):
        
        self.shape = shape
        self.current_error = 0
        
        self.layers = []
        for i in range(len(self.shape)):
            self.layers.append(np.zeros([1, self.shape[i]]))

        self.Theta = [] # weight of each layer
        for i in range(len(shape)-1):
            self.Theta.append(np.random.rand(shape[i], shape[i+1]))
        
        self.Dlocal_output_Dweight = [] 
        i = 0
        for layer in self.layers[1:]:
            self.Dlocal_output_Dweight.append(np.zeros([self.shape[i], self.shape[i+1]]))
            i+=1
         
        self.Dlocal_output_Dlocal_input = [] # dlayer[n] /dlayer[n-1]
        i=1
        for layer in self.layers[2:]:
            self.Dlocal_output_Dlocal_input.append(np.zeros([shape[i], shape[i+1]]))
            i+=1

        self.dE_dTheta = []
        i=0
        for layer in self.layers[1:]:
            self.dE_dTheta.append(np.zeros([shape[i], shape[i+1]]))
            i+=1               
    
    def forward(self, x):
        num_layers = len(self.shape)
        
        def sigmoid(array):
            return 1.0 / (1.0 + np.exp(array))
        
        x = x.reshape(-1, self.shape[0])
        self.layers[0] = x
        i = 1
        for w in self.Theta:
            if i < num_layers - 1:
                x = sigmoid(np.dot(x, w))
            if i == num_layers -1:    
                x = np.dot(x, w)
            self.layers[i] = x
            i+=1
        return x  

    def backward(self, target, loss="MSE"):
        # update self.Dlocal_output_Dlocal_input
        # Local output and the weight matrix are the input
        
        def vectorize_local_output_and_weight(local_output, weight):
            b = local_output.mean(axis=0)
            w = weight
            v_b = b.flatten("C")
            v_b = np.concatenate([v_b]*w.shape[0])
            v_w = w.flatten("C")
            return v_b, v_w
        
        # update self.Dlocal_output_Dlocal_input except the last layer
        # Activation function are sigmoid except at the last layer
        i=0
        for w, layer in zip(self.Theta[1:-1], self.layers[2:-1]):
            v_local_output, v_weight = vectorize_local_output_and_weight(layer, w)
            M = np.multiply(np.multiply(v_local_output, v_local_output -1), -v_weight)
            M = M.reshape(w.shape[0], w.shape[1])
            self.Dlocal_output_Dlocal_input[i] = M
            i+=1
 
        # update the last entry of self.Dlocal_output_Dlocal_input
        # Take into account that the activation is linear in the last module
        self.Dlocal_output_Dlocal_input[-1]=self.Theta[-1]
 
        # update self.Dlocal_output_Dweight
        # Local output and previous local input are the input
        
        def vectorize_local_output_and_local_input(local_output, local_input):
            b = local_output.mean(axis=0)
            a = local_input.mean(axis=0)
            v_b = b.flatten("C")
            v_b = np.concatenate([v_b]*a.shape[0])
            a = a.reshape(1,-1)
            v_a = np.concatenate([a]*b.shape[0])
            v_a = v_a.flatten("F")
            return v_b, v_a

        # update self.Dlocal_output_Dweight except for the last module
        i = 0
        for layer, next_layer in zip(self.layers[:-2], self.layers[1:-1]):
            v_local_output, v_local_input = vectorize_local_output_and_local_input(next_layer, layer) 
            M = np.multiply(np.multiply(v_local_output, v_local_output-1), -v_local_input)
            M = M.reshape(layer.shape[1], next_layer.shape[1])
            self.Dlocal_output_Dweight[i] = M
            i+=1

        # update self.Dlocal_output_Dweight for the last module
        a = self.layers[-2].mean(axis=0).reshape(1, -1)
        v_a = np.concatenate([a]*self.shape[-1])
        v_a = v_a.flatten("F").reshape(self.shape[-2], self.shape[-1])
        self.Dlocal_output_Dweight[-1] = v_a

        # back-propogates dE/d(last_layer)
        if loss == "MSE":
            dE_dLast_layer = self.layers[-1] - target
            self.current_error = np.abs(dE_dLast_layer.mean())
            dE_dLast_layer = dE_dLast_layer.mean(axis=0).reshape(1, -1)
            dE_dLast_layer = dE_dLast_layer.sum(axis=1).reshape(1, 1)
            dE_dLast_layer = np.concatenate([dE_dLast_layer]*self.shape[-1], axis=0)
        
        # if loss is cross-entropy, then we need to define a few helper functions to
        # compute softmax of the last layer, and the derivative of the cross-entropy
        # with repect to the last layer
        def softmax(array):
            return 0
            
        def cross_entropy_prime(prediction, label):
            # takes softmax of the prediction and label inside
            # label needs to be onehot encoded
            # compute derivative of the cross-entropy of the softmax 
            # with respect to the prediction 
            return 0    
                   
        # For sake of chain, I need to define a function that computes the dot-product of a list of 
        # compatible numpy arraies
        
        def compute_chain(list_of_arries):
            x = list_of_arries[0]
            if len(list_of_arries) > 1:
                for y in list_of_arries[1:]:
                    x = np.dot(x, y)
                return x
            else:
                return x

        def dE_dweight(dE, dw):
            v_dE = dE.flatten()
            v_dE = np.concatenate([v_dE]*dw.shape[0])
            v_dw = dw.flatten()
            dE_dw = np.multiply(v_dE, v_dw).reshape(dw.shape[0], dw.shape[1])
            return dE_dw
        
        i=0
        for dw in self.Dlocal_output_Dweight:
            C = self.Dlocal_output_Dlocal_input[i:]
            C.append(dE_dLast_layer)
            C = compute_chain(C)
            dE_dw = dE_dweight(C, dw)
            self.dE_dTheta[i] = dE_dw
            i+=1

    def updateParams(self, eta):
        i=0
        for w, dw in zip(self.Theta, self.dE_dTheta):
            self.Theta[i] = w - eta*dw
            i+=1